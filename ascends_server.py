from __future__ import annotations

import logging
from pathlib import Path
from joblib import dump
from datetime import datetime
import shutil
import json
import re
from typing import Optional, Dict, Any, List
from fastapi import Request, Form, UploadFile, File
from fastapi import Query
from urllib.parse import quote
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import time
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None  # xgb optional; we'll fallback if absent

from uuid import uuid4

import dcor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ascends.core.explain import (
    explain_model as core_explain,
    save_importance_plot,
    save_default_shap_plot,
)
from ascends.gui_plotting import (
    plot_metric_bars as _plot_metric_bars,
    save_confusion_plot,
    save_parity_plot,
    train_img_dir,
)
from ascends.gui_predict_routes import create_predict_router

logger = logging.getLogger("ascends.gui")

app = FastAPI(title="ASCENDS GUI", version="0.1.0")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
WORKSPACE_DIR = BASE_DIR / "workspace"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
PREVIEW_NROWS = 5

def _ws_dir(ws_id: str) -> Path:
    """Workspace directory for a given session id."""
    return WORKSPACE_DIR / ws_id

TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _safe_csv_filename(original: str) -> str:
    stem = Path(original).stem[:50] or "upload"
    return f"{stem}-{uuid4().hex[:8]}.csv"


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


async def _save_csv(file: UploadFile) -> Path:
    name = _safe_csv_filename(file.filename or "data.csv")
    dest = UPLOADS_DIR / name
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large ({len(content) // (1024 * 1024)} MB). Maximum allowed is 50 MB.")
    dest.write_bytes(content)
    return dest


def _manifest_path(ws_id: str) -> Path:
    return _ws_dir(ws_id) / "manifest.json"


def _corr_dirs(ws_id: str) -> tuple[Path, Path]:
    """Return (data_dir, img_dir) for correlation artifacts."""
    data_dir = _ws_dir(ws_id) / "corr"
    img_dir = STATIC_DIR / "workspace" / ws_id / "corr"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, img_dir


def _compute_correlations(
    df: pd.DataFrame, target: str, inputs: List[str], metrics: List[str], task: str
) -> Dict[str, pd.DataFrame]:
    y = df[target].values
    X = df[inputs]
    out: Dict[str, pd.DataFrame] = {}

    # Pearson
    if "pearson" in metrics:
        vals = []
        for c in inputs:
            x = df[c].values
            if np.std(x) == 0 or np.std(y) == 0:
                vals.append((c, 0.0))
            else:
                r = np.corrcoef(x, y)[0, 1]
                vals.append((c, float(r)))
        pearson_df = pd.DataFrame(vals, columns=["feature", "score"]).sort_values(
            by="score", key=lambda s: np.abs(s), ascending=False
        )
        out["pearson"] = pearson_df

    # Spearman (Pearson on ranks; no SciPy dependency)
    if "spearman" in metrics:
        vals = []
        y_rank = pd.Series(y).rank(method="average").values
        for c in inputs:
            x_rank = df[c].rank(method="average").values
            if np.std(x_rank) == 0 or np.std(y_rank) == 0:
                vals.append((c, 0.0))
            else:
                r = np.corrcoef(x_rank, y_rank)[0, 1]
                vals.append((c, float(r)))
        spearman_df = pd.DataFrame(vals, columns=["feature", "score"]).sort_values(
            by="score", key=lambda s: np.abs(s), ascending=False
        )
        out["spearman"] = spearman_df

    # Mutual information
    if "mi" in metrics:
        if task == "c":
            # Classification target: expect integer labels
            y_disc = pd.Series(y).astype("category").cat.codes.values
            mi_vals = mutual_info_classif(X.values, y_disc, random_state=0)
        else:
            mi_vals = mutual_info_regression(X.values, y, random_state=0)
        mi_df = pd.DataFrame({"feature": inputs, "score": mi_vals}).sort_values(
            by="score", ascending=False
        )
        out["mi"] = mi_df

    # Distance correlation
    if "dcor" in metrics:
        vals = []
        y_f = np.asarray(y, dtype=np.float64)
        # Optional speed cap for very large datasets
        n = len(y_f)
        if n > 5000:
            rng = np.random.RandomState(0)
            idx = rng.choice(n, 5000, replace=False)
            y_f = y_f[idx]
            X_sub = X.iloc[idx, :]
        else:
            X_sub = X
        for c in inputs:
            x = np.asarray(X_sub[c].values, dtype=np.float64)
            try:
                s = float(dcor.distance_correlation(x, y_f))
            except Exception:
                s = 0.0
            vals.append((c, s))
        dcor_df = pd.DataFrame(vals, columns=["feature", "score"]).sort_values(
            by="score", ascending=False
        )
        out["dcor"] = dcor_df

    return out


def _prepare_corr_dataframe(csv_path: str, target: str, inputs: List[str]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(csv_path)
    cols = list(inputs) + [target]
    # Keep only requested columns that actually exist
    existing = [c for c in cols if c in raw.columns]
    df = raw.loc[:, existing].copy()
    # Coerce to numeric for all used columns
    for c in existing:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rows_before = len(df)
    # Drop any rows with NaNs in used columns
    df = df.dropna(axis=0, how="any")
    rows_after = len(df)
    # Ensure floating dtype for all used columns (dcor prefers float arrays)
    df = df.astype("float64")
    dropped = rows_before - rows_after
    # Skip constant input columns
    skipped: List[str] = []
    good_inputs: List[str] = []
    for c in inputs:
        if c in df.columns:
            if df[c].nunique(dropna=True) <= 1:
                skipped.append(c)
            else:
                good_inputs.append(c)
    # Final column order: good inputs + target (if present)
    keep_cols = [c for c in good_inputs if c in df.columns] + ([target] if target in df.columns else [])
    df = df.loc[:, keep_cols]
    info: Dict[str, Any] = {
        "rows_in": rows_before,
        "rows_used": rows_after,
        "rows_dropped": dropped,
        "skipped_inputs": skipped,
        "used_inputs": good_inputs,
    }
    return df, info
def _save_manifest(ws_id: str, data: Dict[str, Any]) -> None:
    """Save the manifest for a given workspace ID."""
    d = _ws_dir(ws_id)
    d.mkdir(parents=True, exist_ok=True)
    _manifest_path(ws_id).write_text(json.dumps(data, indent=2), encoding="utf-8")

def _load_manifest(ws_id: str) -> Dict[str, Any]:
    """Load the manifest for a given workspace ID."""
    p = _manifest_path(ws_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


@app.get("/favicon.svg")
async def _favicon_svg():
    # Serve the SVG to requests for /favicon.svg
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
async def _apple_touch_icon():
    # iOS prefers PNG, but serving SVG avoids 404 and is acceptable as a placeholder.
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "port": 7777}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/ui-lab", response_class=HTMLResponse)
async def ui_lab(request: Request) -> HTMLResponse:
    """TS + utility-first UI proof page."""
    return templates.TemplateResponse("ui_lab.html", {"request": request})


# Helper to preserve order & uniqueness
def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# Replace /train GET with context that loads manifest using ws_id or cookie
# Replace the /train GET to load manifest by ws_id (from query or cookie)
# Replace the /train GET with a version that logs what it sees
@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request, ws_id: Optional[str] = None) -> HTMLResponse:
    ws = ws_id or request.query_params.get("ws_id")
    ctx: Dict[str, Any] = {"request": request, "ws_id": ws}
    if ws:
        mf = _load_manifest(ws) or {}
        shap_view = str(mf.get("shap_view", "ascends")).lower()
        if shap_view not in {"ascends", "default"}:
            shap_view = "ascends"
        ctx.update({
            "csv_path": mf.get("csv_path"),
            "all_columns": mf.get("columns", []),
            "selected": mf.get("selected", []),
            "inputs": mf.get("inputs", []),
            "target": mf.get("target"),
            "shap_view": shap_view,
        })

        # DESIGN DECISION:
        # Keep ASCENDS custom plot as the default UI because users found it more readable.
        # If "default" plot is requested but unavailable, fallback to ASCENDS plot.
        shap_png = STATIC_DIR / "workspace" / ws / "train" / f"shap_importance_{shap_view}.png"
        legacy_png = STATIC_DIR / "workspace" / ws / "train" / "shap_importance.png"
        fallback_png = STATIC_DIR / "workspace" / ws / "train" / "shap_importance_ascends.png"
        shap_csv = _ws_dir(ws) / "train" / "shap_importance.csv"
        if shap_png.exists():
            ctx["shap_img_url"] = f"/static/workspace/{ws}/train/{shap_png.name}?ts={int(time.time())}"
        elif shap_view == "default" and fallback_png.exists():
            ctx["shap_img_url"] = f"/static/workspace/{ws}/train/{fallback_png.name}?ts={int(time.time())}"
        elif legacy_png.exists():
            ctx["shap_img_url"] = f"/static/workspace/{ws}/train/{legacy_png.name}?ts={int(time.time())}"
        if shap_csv.exists():
            try:
                df_shap = pd.read_csv(shap_csv).head(10)
                ctx["shap_rows"] = df_shap.values.tolist()
            except Exception:
                pass
    # Always include saved runs for the bottom-right pane
    ctx["saved_runs"] = _list_saved_runs()
    return templates.TemplateResponse("train.html", ctx)


@app.post("/train/shap", response_class=HTMLResponse)
async def train_shap(
    request: Request,
    ws_id: str = Form(...),
    max_samples: int = Form(300),
    shap_view: str = Form("ascends"),
) -> HTMLResponse:
    """Compute SHAP/permutation importance for the latest trained model in this workspace."""
    mf = _load_manifest(ws_id) or {}
    ctx: Dict[str, Any] = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": mf.get("csv_path"),
        "all_columns": mf.get("columns", []),
        "selected": mf.get("selected", []),
        "inputs": mf.get("inputs", []),
        "target": mf.get("target"),
        "saved_runs": _list_saved_runs(),
    }
    shap_view = str(shap_view or "ascends").lower()
    if shap_view not in {"ascends", "default"}:
        shap_view = "ascends"
    ctx["shap_view"] = shap_view

    rec = LAST_TRAIN.get(ws_id)
    if not rec:
        ctx["train_error"] = "No trained model found in this workspace. Train first, then run SHAP."
        return templates.TemplateResponse("train.html", ctx)

    # Keep showing existing train outputs if present
    ctx["metrics_train"] = rec.get("metrics_train")
    ctx["metrics_test"] = rec.get("metrics_test")
    ctx["parity_img_url"] = rec.get("parity_img_url")

    csv_path = rec.get("csv_path")
    inputs = rec.get("inputs", [])
    target = rec.get("target")
    task = rec.get("params", {}).get("task", "r")
    est = rec.get("estimator")
    if not csv_path or not est or not inputs or not target:
        ctx["train_error"] = "Insufficient training context for SHAP. Re-train and try again."
        return templates.TemplateResponse("train.html", ctx)

    try:
        df = pd.read_csv(csv_path)
        required = [c for c in inputs if c in df.columns] + ([target] if target in df.columns else [])
        if target not in required:
            raise ValueError("Target column missing in source CSV.")
        df2 = df[required].dropna(axis=0, how="any")
        X = df2[inputs]
        y = df2[target]
        task_name = "classification" if str(task).lower() == "c" else "regression"
        expl = core_explain(
            model=est,
            X=X,
            y=y,
            task=task_name,
            max_samples=max(50, int(max_samples)),
            random_state=42,
        )
    except Exception as e:
        ctx["train_error"] = f"SHAP failed: {e}"
        return templates.TemplateResponse("train.html", ctx)

    # Save artifacts
    data_dir = _ws_dir(ws_id) / "train"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = _train_img_dir(ws_id)
    csv_out = data_dir / "shap_importance.csv"
    report_out = data_dir / "shap_report.json"
    png_ascends = img_dir / "shap_importance_ascends.png"
    png_default = img_dir / "shap_importance_default.png"

    imp_df = expl["importance_df"]
    imp_df.to_csv(csv_out, index=False)

    save_importance_plot(
        imp_df,
        png_ascends,
        method=str(expl.get("method", "shap")),
        top_n=20,
    )
    default_ready = False
    if str(expl.get("method", "")).lower() == "shap":
        try:
            save_default_shap_plot(
                model=est,
                X=X,
                out_png=png_default,
                max_samples=max(50, int(max_samples)),
                random_state=42,
                max_display=20,
            )
            default_ready = True
        except Exception as e:
            warn = str(expl.get("warning") or "").strip()
            extra = f"Default SHAP view failed ({e}); using ASCENDS view."
            expl["warning"] = f"{warn} {extra}".strip() if warn else extra

    report_out.write_text(
        json.dumps(
            {
                "method": expl.get("method"),
                "warning": expl.get("warning"),
                "n_samples": expl.get("n_samples"),
                "csv_path": str(csv_out),
                "png_ascends_path": str(png_ascends),
                "png_default_path": str(png_default) if default_ready else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_png = png_default if (shap_view == "default" and default_ready) else png_ascends
    ctx["shap_img_url"] = f"/static/workspace/{ws_id}/train/{selected_png.name}?ts={int(time.time())}"
    ctx["shap_rows"] = imp_df.head(10).values.tolist()
    if expl.get("warning"):
        ctx["shap_warning"] = expl["warning"]

    # Persist UI preference for next Train page load.
    mf["shap_view"] = shap_view
    _save_manifest(ws_id, mf)

    return templates.TemplateResponse("train.html", ctx)


@app.post("/train/shap/view", response_class=HTMLResponse)
async def train_shap_view(
    request: Request,
    ws_id: str = Form(...),
    shap_view: str = Form("ascends"),
) -> HTMLResponse:
    """Switch displayed SHAP image without recomputing model explanation."""
    mf = _load_manifest(ws_id) or {}
    shap_view = str(shap_view or "ascends").lower()
    if shap_view not in {"ascends", "default"}:
        shap_view = "ascends"
    mf["shap_view"] = shap_view
    _save_manifest(ws_id, mf)

    ctx: Dict[str, Any] = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": mf.get("csv_path"),
        "all_columns": mf.get("columns", []),
        "selected": mf.get("selected", []),
        "inputs": mf.get("inputs", []),
        "target": mf.get("target"),
        "saved_runs": _list_saved_runs(),
        "shap_view": shap_view,
    }

    rec = LAST_TRAIN.get(ws_id)
    if rec:
        ctx["metrics_train"] = rec.get("metrics_train")
        ctx["metrics_test"] = rec.get("metrics_test")
        ctx["parity_img_url"] = rec.get("parity_img_url")

    img_dir = _train_img_dir(ws_id)
    selected_png = img_dir / f"shap_importance_{shap_view}.png"
    fallback_png = img_dir / "shap_importance_ascends.png"
    legacy_png = img_dir / "shap_importance.png"
    if selected_png.exists():
        ctx["shap_img_url"] = f"/static/workspace/{ws_id}/train/{selected_png.name}?ts={int(time.time())}"
    elif fallback_png.exists():
        ctx["shap_img_url"] = f"/static/workspace/{ws_id}/train/{fallback_png.name}?ts={int(time.time())}"
        if shap_view == "default":
            ctx["shap_warning"] = "Default SHAP view is not available for this run. Showing ASCENDS view."
    elif legacy_png.exists():
        ctx["shap_img_url"] = f"/static/workspace/{ws_id}/train/{legacy_png.name}?ts={int(time.time())}"

    shap_csv = _ws_dir(ws_id) / "train" / "shap_importance.csv"
    if shap_csv.exists():
        try:
            df_shap = pd.read_csv(shap_csv).head(10)
            ctx["shap_rows"] = df_shap.values.tolist()
        except Exception:
            pass

    return templates.TemplateResponse("train.html", ctx)


@app.post("/train/select", response_class=HTMLResponse)
async def train_select(
    request: Request,
    ws_id: str = Form(...),
    action: str = Form(...),
    columns: Optional[List[str]] = Form(None),
    rm_inputs: Optional[List[str]] = Form(None),
    target_choice: Optional[str] = Form(None),
) -> HTMLResponse:
    """Handle Train tab selection state (columns/inputs/target)."""
    mf = _load_manifest(ws_id)
    if not mf:
        return templates.TemplateResponse(
            "train.html",
            {
                "request": request,
                "ws_id": ws_id,
                "train_error": "Invalid session. Please upload/select data from Correlation tab first.",
                "saved_runs": _list_saved_runs(),
            },
        )

    all_columns: List[str] = list(mf.get("columns", []))
    selected = set(mf.get("selected", []))
    inputs = set(mf.get("inputs", []))
    target = mf.get("target")

    chosen_cols = columns or []
    to_remove = rm_inputs or []

    if action == "select_all":
        selected = set(all_columns)
    elif action == "select_none":
        selected = set()
    elif action == "to_inputs":
        selected = set(chosen_cols)
        for c in chosen_cols:
            if c in all_columns:
                inputs.add(c)
        if target in inputs:
            inputs.discard(target)
    elif action == "remove_inputs":
        for c in to_remove:
            inputs.discard(c)
    elif action == "set_target":
        selected = set(chosen_cols)
        if target_choice and target_choice in all_columns:
            target = target_choice
            if target in inputs:
                inputs.discard(target)

    ordered_inputs = sorted(inputs, key=lambda c: all_columns.index(c)) if all_columns else list(inputs)
    ordered_selected = sorted(selected, key=lambda c: all_columns.index(c)) if all_columns else list(selected)

    mf["inputs"] = ordered_inputs
    mf["target"] = target
    mf["selected"] = ordered_selected
    _save_manifest(ws_id, mf)

    ctx: Dict[str, Any] = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": mf.get("csv_path"),
        "all_columns": all_columns,
        "inputs": ordered_inputs,
        "target": target,
        "selected": ordered_selected,
        "saved_runs": _list_saved_runs(),
    }
    return templates.TemplateResponse("train.html", ctx)

# Save the current trained model and artifacts into runs/<name>/
@app.post("/train/save", response_class=HTMLResponse)
async def train_save(
    request: Request,
    ws_id: str = Form(...),
    save_name: Optional[str] = Form(None),
) -> HTMLResponse:
    ctx: Dict[str, Any] = {"request": request, "ws_id": ws_id}
    # Load manifest to re-populate panes
    mf = _load_manifest(ws_id) or {}
    ctx.update({
        "csv_path": mf.get("csv_path"),
        "all_columns": mf.get("columns", []),
        "selected": mf.get("selected", []),
        "inputs": mf.get("inputs", []),
        "target": mf.get("target"),
    })

    rec = LAST_TRAIN.get(ws_id)
    if not rec:
        ctx["train_error"] = "No trained model available to save. Please Train first."
        ctx["saved_runs"] = _list_saved_runs()
        return templates.TemplateResponse("train.html", ctx)

    # Determine run name
    base = save_name or (rec["params"].get("model", "model") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_name = _unique_run_name(base)  # also creates the directory atomically
    out_dir = RUNS_DIR / run_name

    # Save model
    try:
        dump(rec["estimator"], out_dir / "model.joblib")
    except Exception as e:
        ctx["train_error"] = f"Failed to save model: {e}"
        shutil.rmtree(out_dir, ignore_errors=True)
        ctx["saved_runs"] = _list_saved_runs()
        return templates.TemplateResponse("train.html", ctx)

    # Save metrics.csv (supports both regression and classification metric keys)
    try:
        import pandas as _pd
        train_metrics = dict(rec.get("metrics_train", {}))
        test_metrics = dict(rec.get("metrics_test", {}))
        dfm = _pd.DataFrame(
            [
                {"split": "Train", **train_metrics},
                {"split": "Test", **test_metrics},
            ]
        )
        dfm.to_csv(out_dir / "metrics.csv", index=False)
    except Exception as e:
        ctx["train_error"] = f"Failed to write metrics.csv: {e}"
        shutil.rmtree(out_dir, ignore_errors=True)
        ctx["saved_runs"] = _list_saved_runs()
        return templates.TemplateResponse("train.html", ctx)

    # Save manifest.json for the run
    manifest = {
        "name": run_name,
        "created_at": rec["timestamp"],
        "task": rec["params"].get("task", "r"),
        "model": rec["params"].get("model"),
        "seed": rec["params"].get("seed"),
        "test_size": rec["params"].get("test_size"),
        "tune": rec["params"].get("tune"),
        "inputs": rec.get("inputs", []),
        "target": rec.get("target"),
        "csv_path": rec.get("csv_path"),
        "ws_id": ws_id,
    }
    try:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as e:
        ctx["train_error"] = f"Failed to write manifest.json: {e}"
        shutil.rmtree(out_dir, ignore_errors=True)
        ctx["saved_runs"] = _list_saved_runs()
        return templates.TemplateResponse("train.html", ctx)

    # Copy train visualization (parity/confusion) if present
    try:
        ws_train_dir = STATIC_DIR / "workspace" / ws_id / "train"
        parity_img = ws_train_dir / "parity.png"
        confusion_img = ws_train_dir / "confusion.png"
        shap_img = ws_train_dir / "shap_importance_ascends.png"
        if parity_img.exists():
            shutil.copyfile(parity_img, out_dir / "parity.png")
        if confusion_img.exists():
            shutil.copyfile(confusion_img, out_dir / "confusion.png")
        if shap_img.exists():
            shutil.copyfile(shap_img, out_dir / "shap_importance.png")
    except Exception:
        pass

    # Generate report.html
    try:
        _generate_report(run_name, out_dir, rec, ws_id)
    except Exception as e:
        logger.warning("Report generation failed for %s: %s", run_name, e)

    ctx["save_ok"] = f"Saved run: {run_name}"
    ctx["saved_runs"] = _list_saved_runs()
    return templates.TemplateResponse("train.html", ctx)

@app.get("/train/report", response_class=HTMLResponse)
async def train_report_preview(request: Request, ws_id: str = Query(...)) -> HTMLResponse:
    """Render a live report from LAST_TRAIN without requiring Save."""
    rec = LAST_TRAIN.get(ws_id)
    if not rec:
        return HTMLResponse(content="No trained model found. Please train a model first.", status_code=404)
    return HTMLResponse(content=_render_report_html("(unsaved)", rec, ws_id))


def _render_report_html(run_name: str, rec: Dict[str, Any], ws_id: str, out_dir: Optional[Path] = None) -> str:
    """Render report.html from a LAST_TRAIN record and return HTML string.

    When out_dir is provided (saved run), plot src paths are relative to the
    run directory so the report works when opened directly from the filesystem.
    """
    from ascends.core.interpret import interpret_run

    task = rec["params"].get("task", "r")
    task_label = "Classification" if task == "c" else "Regression"
    train_metrics = dict(rec.get("metrics_train") or {})
    test_metrics = dict(rec.get("metrics_test") or {})
    inputs = rec.get("inputs", [])
    target = rec.get("target", "")
    n_train = rec.get("n_train") or 0
    n_test = rec.get("n_test") or 0

    metric_keys = list(dict.fromkeys(list(train_metrics.keys()) + list(test_metrics.keys())))

    # Load SHAP importance if available
    importance_rows = []
    importance_df = None
    shap_csv = _ws_dir(ws_id) / "train" / "shap_importance.csv"
    if shap_csv.exists():
        try:
            importance_df = pd.read_csv(shap_csv)
            importance_rows = importance_df.head(15).values.tolist()
        except Exception:
            pass

    # Load target values for MAE context
    target_values = None
    if task in ("r", "regression"):
        try:
            csv_path = rec.get("csv_path")
            if csv_path:
                df_tmp = pd.read_csv(csv_path, usecols=[target])
                target_values = df_tmp[target].dropna().tolist()
        except Exception:
            pass

    raw_insights = interpret_run(
        task=task,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        n_train=n_train,
        n_test=n_test,
        target_values=target_values,
        importance_df=importance_df,
    )

    def _level(text: str) -> str:
        low = text.lower()
        if any(w in low for w in ("overfitting", "imbalance", "leakage", "worse", "low", "poor", "large error", "small training", "heavily relies")):
            return "warn"
        if any(w in low for w in ("very good", "excellent", "consistent", "low error", "good")):
            return "good"
        return ""

    insights = [{"text": t, "level": _level(t)} for t in raw_insights]

    # Plot paths: relative for saved reports, absolute URLs for live preview
    plot_files = []
    if out_dir is not None:
        for fname, label in [("parity.png", "Parity Plot"), ("confusion.png", "Confusion Matrix"), ("shap_importance.png", "Feature Importance")]:
            if (out_dir / fname).exists():
                plot_files.append({"src": fname, "label": label})
    else:
        ws_train_dir = STATIC_DIR / "workspace" / ws_id / "train"
        for fname, label, url_name in [
            ("parity.png", "Parity Plot", "parity.png"),
            ("confusion.png", "Confusion Matrix", "confusion.png"),
            ("shap_importance_ascends.png", "Feature Importance", "shap_importance_ascends.png"),
        ]:
            if (ws_train_dir / fname).exists():
                plot_files.append({"src": f"/static/workspace/{ws_id}/train/{url_name}", "label": label})

    return templates.get_template("report.html").render(
        run_name=run_name,
        task_label=task_label,
        model=rec["params"].get("model", ""),
        target=target,
        created_at=rec.get("timestamp", ""),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        metric_keys=metric_keys,
        n_train=n_train,
        n_test=n_test,
        insights=insights,
        plot_files=plot_files,
        importance_rows=importance_rows,
        inputs=inputs,
        test_size=rec["params"].get("test_size", ""),
        seed=rec["params"].get("seed", ""),
    )


def _generate_report(run_name: str, out_dir: Path, rec: Dict[str, Any], ws_id: str) -> None:
    """Render report.html into the run directory.

    For saved runs, plot src attributes use relative paths so the report
    is portable (works when opened directly from the filesystem).
    """
    # Build a copy of rec with plot paths pointing to the run directory
    rec_saved = dict(rec)
    html = _render_report_html(run_name, rec_saved, ws_id, out_dir=out_dir)
    (out_dir / "report.html").write_text(html, encoding="utf-8")


@app.get("/runs/{run_name}/report.html", response_class=HTMLResponse)
async def serve_report(run_name: str) -> HTMLResponse:
    """Serve the saved report.html for a run."""
    report_path = RUNS_DIR / run_name / "report.html"
    if not report_path.exists():
        return HTMLResponse(content="Report not found.", status_code=404)
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))


# Delete a saved run directory
@app.post("/train/delete", response_class=HTMLResponse)
async def train_delete(
    request: Request,
    run_name: str = Form(...),
    ws_id: Optional[str] = Form(None),
) -> HTMLResponse:
    ctx: Dict[str, Any] = {"request": request}
    target_dir = RUNS_DIR / run_name
    if target_dir.exists() and target_dir.is_dir():
        try:
            shutil.rmtree(target_dir)
            ctx["save_ok"] = f"Deleted run: {run_name}"
        except Exception as e:
            ctx["train_error"] = f"Failed to delete run {run_name}: {e}"
    else:
        ctx["train_error"] = f"Run not found: {run_name}"
    # Keep workspace context after delete so Train/SHAP panels remain stable.
    if ws_id:
        return RedirectResponse(url=f"/train?ws_id={quote(ws_id)}", status_code=303)

    # Fallback when no workspace context exists.
    ctx["saved_runs"] = _list_saved_runs()
    return templates.TemplateResponse("train.html", ctx)

@app.get("/correlation", response_class=HTMLResponse)
async def correlation_page(request: Request, ws_id: Optional[str] = None) -> HTMLResponse:
    ctx: Dict[str, Any] = {"request": request}
    if ws_id:
        mf = _load_manifest(ws_id)
        if mf:
            ctx.update(
                {
                    "ws_id": ws_id,
                    "csv_path": mf.get("csv_path"),
                    "all_columns": mf.get("columns", []),
                    "inputs": mf.get("inputs", []),
                    "target": mf.get("target"),
                    "selected": mf.get("selected", []),
                }
            )
            csvp = mf.get("csv_path")
            if csvp:
                try:
                    df = pd.read_csv(csvp, nrows=PREVIEW_NROWS)
                    ctx["preview_headers"] = list(df.columns)
                    ctx["preview_rows"] = (
                        df.astype(object).where(pd.notnull(df), None).values.tolist()
                    )
                except Exception:
                    pass
    # If previous correlation artifacts exist, pre-populate the chart/table
    corr = mf.get("corr", {}) if ws_id else {}
    artifacts = corr.get("artifacts") or {}
    if artifacts:
        available = list(artifacts.keys())
        active_metric = corr.get("active_metric") or (available[0] if available else None)
        chart_url = artifacts.get(active_metric, {}).get("png")
        # Load top rows from CSV with same sorting/top_k logic
        metric_rows = []
        try:
            csvp = artifacts.get(active_metric, {}).get("csv")
            if csvp:
                dfm = pd.read_csv(csvp)
                if active_metric in {"pearson", "spearman"}:
                    dfm = dfm.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
                else:
                    dfm = dfm.sort_values(by="score", ascending=False)
                tk = corr.get("top_k")
                if tk and tk > 0:
                    dfm = dfm.head(tk)
                metric_rows = dfm.values.tolist()
        except Exception:
            pass
        ctx.update({
            "available_metrics": available,
            "active_metric": active_metric,
            "chart_url": chart_url,
            "metric_rows": metric_rows,
        })
    return templates.TemplateResponse("correlation.html", ctx)

@app.post("/correlation/run", response_class=HTMLResponse)
async def correlation_run(
    request: Request,
    ws_id: str = Form(...),
    metrics: Optional[List[str]] = Form(None),
    top_k: Optional[str] = Form(None),
    task: str = Form("r"),
    view: str = Form("wide"),
) -> HTMLResponse:
    mf = _load_manifest(ws_id)
    if not mf:
        return templates.TemplateResponse("correlation.html", {"request": request, "error": "Invalid session. Please re-upload CSV."})
    cols = mf.get("columns", [])
    inputs = mf.get("inputs", [])
    target = mf.get("target")
    if not target:
        return templates.TemplateResponse("correlation.html", {"request": request, "ws_id": ws_id, "csv_path": mf.get("csv_path"), "all_columns": cols, "inputs": inputs, "target": target, "selected": mf.get("selected", []), "error": "Please set a target column before running correlation."})
    if not inputs:
        return templates.TemplateResponse("correlation.html", {"request": request, "ws_id": ws_id, "csv_path": mf.get("csv_path"), "all_columns": cols, "inputs": inputs, "target": target, "selected": mf.get("selected", []), "error": "Please select at least one input feature."})

    # Ensure corr section exists even for older workspaces
    corr_section = mf.setdefault("corr", {})

    chosen_metrics = metrics or ["pearson", "spearman", "mi", "dcor"]
    # Parse top_k safely: treat empty string as None
    top_k_val = None
    if top_k is not None and str(top_k).strip() != "":
        try:
            top_k_val = int(str(top_k).strip())
            if top_k_val <= 0:
                raise ValueError("top_k must be > 0")
        except Exception as e:
            # Re-render with a friendly error
            ctx = {
                "request": request,
                "ws_id": ws_id,
                "csv_path": mf.get("csv_path"),
                "all_columns": cols,
                "inputs": inputs,
                "target": target,
                "selected": mf.get("selected", []),
                "error": f"Invalid Top-K: {e}",
            }
            try:
                df = pd.read_csv(mf.get("csv_path"), nrows=PREVIEW_NROWS)
                ctx["preview_headers"] = list(df.columns)
                ctx["preview_rows"] = df.astype(object).where(pd.notnull(df), None).values.tolist()
            except Exception:
                pass
            return templates.TemplateResponse("correlation.html", ctx)

    # Store settings
    corr_section["metrics"] = chosen_metrics
    corr_section["top_k"] = top_k_val
    corr_section["task"] = task
    corr_section["view"] = view
    # Prepare data and write cleaning log
    data_dir, img_dir = _corr_dirs(ws_id)
    df_clean, info = _prepare_corr_dataframe(mf["csv_path"], target, inputs)
    mf["corr"]["used_inputs"] = info.get("used_inputs", [])
    (data_dir / "corr_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    # Base context (so we never reference ctx before assignment)
    ctx = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": mf.get("csv_path"),
        "all_columns": cols,
        "inputs": inputs,
        "target": target,
        "selected": mf.get("selected", []),
        "corr_info": info,
    }
    # Add preview to ctx
    try:
        df_prev = pd.read_csv(mf.get("csv_path"), nrows=PREVIEW_NROWS)
        ctx["preview_headers"] = list(df_prev.columns)
        ctx["preview_rows"] = df_prev.astype(object).where(pd.notnull(df_prev), None).values.tolist()
    except Exception:
        pass
    try:
        used_inputs = mf["corr"].get("used_inputs", inputs)
        if not used_inputs:
            ctx["error"] = "No usable input features after cleaning (all constant or dropped)."
            return templates.TemplateResponse("correlation.html", ctx)

        chosen_metrics = mf["corr"].get("metrics", ["pearson", "spearman", "mi", "dcor"])
        results = _compute_correlations(df_clean, target, used_inputs, chosen_metrics, mf["corr"].get("task", "r"))

        data_dir, img_dir = _corr_dirs(ws_id)
        artifacts: Dict[str, Dict[str, str]] = {}
        for m, dfm in results.items():
            csv_path = data_dir / f"{m}.csv"
            dfm.to_csv(csv_path, index=False)
            png_path = img_dir / f"{m}.png"
            _plot_metric_bars(dfm, m, target, info["rows_used"], png_path, top_k=mf["corr"].get("top_k"))
            artifacts[m] = {"csv": str(csv_path), "png": f"/static/workspace/{ws_id}/corr/{m}.png"}

        mf["corr"]["artifacts"] = artifacts
        _save_manifest(ws_id, mf)

        active_metric = next(iter(artifacts.keys())) if artifacts else None
        chart_url = artifacts[active_metric]["png"] if active_metric else None

        top_df = results[active_metric].copy() if active_metric else pd.DataFrame(columns=["feature", "score"])
        tk = mf["corr"].get("top_k")
        if active_metric in {"pearson", "spearman"}:
            top_df = top_df.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
        else:
            top_df = top_df.sort_values(by="score", ascending=False)
        if tk and tk > 0:
            top_df = top_df.head(tk)
        ctx["available_metrics"] = list(artifacts.keys())
        ctx["active_metric"] = active_metric
        ctx["chart_url"] = chart_url
        ctx["metric_rows"] = top_df.values.tolist()

        return templates.TemplateResponse("correlation.html", ctx)

    except Exception as e:
        ctx["error"] = f"Correlation computation failed: {e}"
        return templates.TemplateResponse("correlation.html", ctx)


@app.post("/correlation/view", response_class=HTMLResponse)
async def correlation_view(
    request: Request,
    ws_id: str = Form(...),
    metric: str = Form(...),
) -> HTMLResponse:
    mf = _load_manifest(ws_id)
    if not mf or "corr" not in mf or not mf["corr"].get("artifacts"):
        return templates.TemplateResponse("correlation.html", {"request": request, "error": "No correlation artifacts available. Please run correlation first."})

    cols = mf.get("columns", [])
    inputs = mf.get("inputs", [])
    target = mf.get("target")
    corr = mf["corr"]
    artifacts = corr.get("artifacts") or {}
    available = list(artifacts.keys())
    if not available:
        return templates.TemplateResponse("correlation.html", {"request": request, "error": "No metrics available."})
    if metric not in available:
        metric = available[0]

    # Build base ctx (with preview)
    ctx = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": mf.get("csv_path"),
        "all_columns": cols,
        "inputs": inputs,
        "target": target,
        "selected": mf.get("selected", []),
        "available_metrics": available,
        "active_metric": metric,
        "chart_url": artifacts[metric]["png"],
    }
    try:
        df_prev = pd.read_csv(mf.get("csv_path"), nrows=PREVIEW_NROWS)
        ctx["preview_headers"] = list(df_prev.columns)
        ctx["preview_rows"] = df_prev.astype(object).where(pd.notnull(df_prev), None).values.tolist()
    except Exception:
        pass

    # Load top rows for the selected metric
    try:
        dfm = pd.read_csv(artifacts[metric]["csv"])
        if metric in {"pearson", "spearman"}:
            dfm = dfm.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
        else:
            dfm = dfm.sort_values(by="score", ascending=False)
        tk = corr.get("top_k")
        if tk and tk > 0:
            dfm = dfm.head(tk)
        ctx["metric_rows"] = dfm.values.tolist()
    except Exception:
        ctx["metric_rows"] = []

    # Persist the last-viewed metric
    corr["active_metric"] = metric
    _save_manifest(ws_id, mf)

    return templates.TemplateResponse("correlation.html", ctx)

@app.post("/correlation/upload", response_class=HTMLResponse)
async def correlation_upload(request: Request, csvfile: UploadFile = File(...)) -> HTMLResponse:
    fname = (csvfile.filename or "").lower()
    if not fname.endswith(".csv"):
        return templates.TemplateResponse(
            "correlation.html", {"request": request, "error": "Please upload a .csv file."}
        )
    saved = await _save_csv(csvfile)
    try:
        df = pd.read_csv(saved, nrows=PREVIEW_NROWS)
        headers = list(df.columns)
        rows = df.astype(object).where(pd.notnull(df), None).values.tolist()
    except Exception as e:
        return templates.TemplateResponse(
            "correlation.html", {"request": request, "error": f"Failed to read CSV: {e}"}
        )
    ws_id = uuid4().hex
    manifest = {
        "csv_path": str(saved),
        "columns": headers,
        "inputs": [],
        "target": None,
        "selected": []
    }
    _save_manifest(ws_id, manifest)
    ctx: Dict[str, Any] = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": str(saved),
        "preview_headers": headers,
        "preview_rows": rows,
        "all_columns": headers,
        "inputs": [],
        "target": None,
        "selected": [],
    }
    return templates.TemplateResponse("correlation.html", ctx)


@app.post("/correlation/select", response_class=HTMLResponse)
async def correlation_select(
    request: Request,
    ws_id: str = Form(...),
    action: str = Form(...),
    cols: Optional[List[str]] = Form(None),
    inputs: Optional[List[str]] = Form(None),
    target_choice: Optional[str] = Form(None),
) -> HTMLResponse:
    mf = _load_manifest(ws_id)
    if not mf:
        return templates.TemplateResponse(
            "correlation.html", {"request": request, "error": "Invalid session. Please re-upload CSV."}
        )

    columns: List[str] = list(mf.get("columns", []))
    cur_inputs = set(mf.get("inputs", []))
    cur_target = mf.get("target")

    cur_selected = set(mf.get("selected", []))

    chosen_cols = cols or []
    inputs_sel = inputs or []

    if action == "select_all":
        cur_selected = set(columns)

    elif action == "select_none":
        cur_selected = set()

    elif action == "add_inputs":
        cur_selected = set(chosen_cols)
        for c in chosen_cols:
            if c in columns:
                cur_inputs.add(c)
        if cur_target in cur_inputs:
            cur_inputs.discard(cur_target)

    elif action == "remove_inputs":
        for c in inputs_sel:
            cur_inputs.discard(c)

    elif action == "set_target":
        if target_choice and target_choice in columns:
            cur_target = target_choice
            # Default UX: once target is chosen, use all remaining columns as inputs.
            cur_inputs = {c for c in columns if c != cur_target}
            cur_selected = set(columns)
        elif chosen_cols:
            cur_selected = set(chosen_cols)

    elif action.startswith("set_target_direct:"):
        direct_target = action.split(":", 1)[1]
        if direct_target in columns:
            cur_target = direct_target
            cur_inputs = {c for c in columns if c != cur_target}
            cur_selected = set(columns)

    elif action.startswith("toggle_input:"):
        col_name = action.split(":", 1)[1]
        if col_name in columns and col_name != cur_target:
            if col_name in cur_inputs:
                cur_inputs.discard(col_name)
            else:
                cur_inputs.add(col_name)
            cur_selected = set(columns)

    elif action == "auto_inputs":
        # Explicit one-click action: use all columns except current target.
        if cur_target and cur_target in columns:
            cur_inputs = {c for c in columns if c != cur_target}
        else:
            cur_inputs = set(columns)
        cur_selected = set(columns)

    elif action == "clear_target":
        cur_target = None

    ordered_inputs = sorted(cur_inputs, key=lambda x: columns.index(x)) if columns else list(cur_inputs)
    ordered_selected = sorted(cur_selected, key=lambda x: columns.index(x)) if columns else list(cur_selected)
    mf["inputs"] = ordered_inputs
    mf["target"] = cur_target
    mf["selected"] = ordered_selected
    _save_manifest(ws_id, mf)

    ctx: Dict[str, Any] = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": mf.get("csv_path"),
        "all_columns": columns,
        "inputs": ordered_inputs,
        "target": cur_target,
        "selected": ordered_selected,
    }

    ctx = {
        "request": request,
        "ws_id": ws_id,
        "csv_path": mf.get("csv_path"),
        "all_columns": columns,
        "inputs": ordered_inputs,
        "target": cur_target,
        "selected": ordered_selected,
    }
    csvp = mf.get("csv_path")
    if csvp:
        try:
            df = pd.read_csv(csvp, nrows=10)
            ctx["preview_headers"] = list(df.columns)
            ctx["preview_rows"] = (
                df.astype(object).where(pd.notnull(df), None).values.tolist()
            )
        except Exception:
            pass

    return templates.TemplateResponse("correlation.html", ctx)


@app.get("/correlation/download_all")
async def correlation_download_all(ws_id: str):
    mf = _load_manifest(ws_id)
    if not mf or "corr" not in mf or not mf["corr"].get("artifacts"):
        return JSONResponse({"error": "No correlation artifacts available."}, status_code=404)

    corr = mf["corr"]
    artifacts = corr.get("artifacts") or {}
    if not artifacts:
        return JSONResponse({"error": "No metrics available."}, status_code=404)

    # Determine column order: use requested metrics order, filtered to existing artifacts
    desired = corr.get("metrics") or list(artifacts.keys())
    metrics_order = [m for m in desired if m in artifacts]
    if not metrics_order:
        metrics_order = list(artifacts.keys())

    # Merge per-metric CSVs on 'feature'
    df_all = None
    for m in metrics_order:
        csvp = artifacts[m].get("csv")
        if not csvp:
            continue
        try:
            dfm = pd.read_csv(csvp)
        except Exception as e:
            return JSONResponse({"error": f"Failed to read {m} CSV: {e}"}, status_code=500)
        # Expect columns ['feature','score']; rename score -> metric name
        if "feature" not in dfm.columns or "score" not in dfm.columns:
            return JSONResponse({"error": f"Unexpected columns in {m} CSV."}, status_code=500)
        dfm = dfm[["feature", "score"]].rename(columns={"score": m})
        if df_all is None:
            df_all = dfm
        else:
            df_all = df_all.merge(dfm, on="feature", how="outer")

    if df_all is None or df_all.empty:
        return JSONResponse({"error": "No data to combine."}, status_code=404)

    # Sort rows for readability, 'feature' first then metrics as selected
    df_all = df_all.sort_values(by="feature", kind="stable")

    # Save to workspace and return as a download
    data_dir, _ = _corr_dirs(ws_id)
    combined_path = data_dir / "combined.csv"
    df_all.to_csv(combined_path, index=False)

    # Persist reference for traceability
    artifacts["combined"] = {"csv": str(combined_path)}
    mf["corr"]["artifacts"] = artifacts
    _save_manifest(ws_id, mf)

    return FileResponse(str(combined_path), media_type="text/csv", filename="correlation_all.csv")

# Helper: pick a regressor by key
def _make_regressor(key: str, seed: Optional[int] = 42):
    k = (key or "rf").lower()
    if k == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    if k == "xgb" and xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            random_state=seed, tree_method="hist", n_jobs=0, verbosity=0
        )
    if k == "hgb":
        return HistGradientBoostingRegressor(random_state=seed)
    if k == "svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1))
    if k == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    if k == "linear":
        return LinearRegression()
    if k == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if k == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=10000))
    if k == "elastic":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))
    # Fallback
    return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)


def _make_classifier(key: str, seed: Optional[int] = 42):
    """Pick a classifier by key."""
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    k = (key or "rf").lower()
    if k == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if k == "xgb" and xgb is not None:
        return xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            n_jobs=0,
            verbosity=0,
        )
    if k == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    if k == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if k == "linear":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if k == "ridge":
        # Same practical behavior as "linear" for classification.
        return LogisticRegression(max_iter=2000, random_state=seed)
    # Fallback
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)

def _train_img_dir(ws_id: str) -> Path:
    return train_img_dir(STATIC_DIR, ws_id)

def _save_parity_plot(
    ws_id: str,
    y_train: np.ndarray, y_pred_train: np.ndarray,
    y_test: np.ndarray,  y_pred_test: np.ndarray,
    metrics_train: Dict[str, float], metrics_test: Dict[str, float],
) -> str:
    return save_parity_plot(
        STATIC_DIR,
        ws_id,
        y_train,
        y_pred_train,
        y_test,
        y_pred_test,
        metrics_train,
        metrics_test,
    )


def _save_confusion_plot(
    ws_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[Any],
) -> str:
    return save_confusion_plot(STATIC_DIR, ws_id, y_true, y_pred, labels)

# --------------------------
# Runs/save helpers & state
# --------------------------
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Cache the last trained estimator & context by workspace (for quick Save)
LAST_TRAIN: Dict[str, Any] = {}

def _slugify_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    # letters, numbers, dash, underscore only
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return slug.strip("_-")

def _unique_run_name(base: str) -> str:
    """Return a unique run name and atomically create its directory under runs/.

    Uses mkdir(exist_ok=False) to avoid TOCTOU race conditions under concurrent requests.
    """
    base = _slugify_name(base) or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base
    n = 2
    while True:
        try:
            (RUNS_DIR / candidate).mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = f"{base}_{n}"
            n += 1

def _list_saved_runs() -> List[Dict[str, Any]]:
    """Scan runs/* and load manifest + metrics for the ML Models pane."""
    out: list[dict] = []
    if not RUNS_DIR.exists():
        return out
    for p in sorted(RUNS_DIR.iterdir()):
        if not p.is_dir():
            continue
        man = p / "manifest.json"
        met = p / "metrics.csv"
        item: dict[str, Any] = {"name": p.name}
        try:
            if man.exists():
                item.update(json.loads(man.read_text(encoding="utf-8")))
        except Exception:
            pass
        # Load test metrics (prefer manifest if present)
        if met.exists():
            try:
                import pandas as _pd  # local to avoid global import shadowing
                dfm = _pd.read_csv(met)
                # Expect rows Train/Test; key columns differ by task.
                test_row = dfm[dfm["split"].str.lower() == "test"]
                if not test_row.empty:
                    tr = test_row.iloc[0]
                    item.setdefault("metrics", {})
                    for col in dfm.columns:
                        if col != "split":
                            try:
                                item["metrics"][col] = float(tr.get(col, float("nan")))
                            except Exception:
                                item["metrics"][col] = tr.get(col)
            except Exception:
                pass
        out.append(item)
    return out


app.include_router(
    create_predict_router(
        templates=templates,
        runs_dir=RUNS_DIR,
        list_saved_runs=_list_saved_runs,
        slugify_name=_slugify_name,
    )
)

# Replace the /train/run handler to accept seed & resample and use them
@app.post("/train/run", response_class=HTMLResponse)
async def train_run(
    request: Request,
    ws_id: str = Form(...),
    task: str = Form(...),          # "r" or "c"
    model: str = Form(...),         # rf/xgb/hgb/svr/knn/linear/ridge/lasso/elastic
    test_size: float = Form(...),   # e.g., 0.2
    tune: str = Form(...),          # off/quick/intense/optuna/bayes (ignored in Step B)
    seed: Optional[str] = Form(None), # numeric seed as text
    resample: Optional[str] = Form(None), # checkbox -> "on" when checked
):
    # Build base context
    ctx: Dict[str, Any] = {"request": request, "ws_id": ws_id}
    mf = _load_manifest(ws_id) or {}
    all_columns = mf.get("columns", [])
    inputs = mf.get("inputs", [])
    target = mf.get("target")

    # Determine seed value
    if resample:  # checkbox checked -> time-based seed
        seed_val = int(time.time()) & 0xFFFFFFFF
    elif seed not in (None, ""):
        try:
            seed_val = int(seed)
        except Exception:
            seed_val = 42
    else:
        seed_val = 42

    ctx.update({
        "csv_path": mf.get("csv_path"),
        "all_columns": all_columns,
        "selected": mf.get("selected", []),
        "inputs": inputs,
        "target": target,
        "train_params": {
            "task": task, "model": model, "test_size": test_size, "tune": tune,
            "seed": seed_val, "resample": bool(resample),
        },
    })

    # Guardrails
    task = (task or "r").lower()
    if task not in {"r", "c"}:
        ctx["train_error"] = "Task must be 'r' (regression) or 'c' (classification)."
        return templates.TemplateResponse("train.html", ctx)
    if not inputs or not target:
        ctx["train_error"] = "Please select at least one input and a target."
        return templates.TemplateResponse("train.html", ctx)

    # Load data
    csv_path = mf.get("csv_path")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        ctx["train_error"] = f"Failed to read CSV: {e}"
        return templates.TemplateResponse("train.html", ctx)

    # Keep only required columns & drop rows with NA in them
    needed = [c for c in inputs if c in df.columns] + ([target] if target in df.columns else [])
    if not needed or target not in needed:
        ctx["train_error"] = "Selected columns not found in CSV. (Case sensitivity or mismatch.)"
        return templates.TemplateResponse("train.html", ctx)

    df2 = df[needed].dropna(axis=0, how="any")
    X = df2[inputs]
    y = df2[target]

    # Split
    try:
        ts = float(test_size)
    except Exception:
        ts = 0.2
    # Split with chosen seed (stratify for classification if possible)
    stratify_vec = y if task == "c" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=seed_val, stratify=stratify_vec
    )

    # Model with chosen seed (where applicable)
    est = _make_regressor(model, seed=seed_val) if task == "r" else _make_classifier(model, seed=seed_val)

    # Fit & predict
    try:
        est.fit(X_train, y_train)
        y_pred_train = est.predict(X_train)
        y_pred_test = est.predict(X_test)
    except Exception as e:
        ctx["train_error"] = f"Model training failed: {e}"
        return templates.TemplateResponse("train.html", ctx)

    # Metrics + plot by task
    if task == "r":
        def _metrics_reg(y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            return {"R2": r2, "MAE": mae, "RMSE": rmse}

        ctx["metrics_train"] = _metrics_reg(y_train, y_pred_train)
        ctx["metrics_test"] = _metrics_reg(y_test, y_pred_test)
        try:
            ctx["parity_img_url"] = _save_parity_plot(
                ws_id,
                y_train, y_pred_train,
                y_test, y_pred_test,
                ctx["metrics_train"], ctx["metrics_test"],
            )
        except Exception as e:
            ctx["train_error"] = f"Failed to generate parity plot: {e}"
    else:
        def _metrics_clf(y_true, y_pred, estm, X_eval):
            out = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
            # ROC-AUC for binary class only (when probabilities are available)
            try:
                classes = pd.Series(y_true).dropna().unique()
                if len(classes) == 2 and hasattr(estm, "predict_proba"):
                    y_proba = estm.predict_proba(X_eval)
                    out["ROC_AUC"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                pass
            return out

        ctx["metrics_train"] = _metrics_clf(y_train, y_pred_train, est, X_train)
        ctx["metrics_test"] = _metrics_clf(y_test, y_pred_test, est, X_test)
        try:
            labels = sorted(pd.Series(y).dropna().unique().tolist())
            ctx["parity_img_url"] = _save_confusion_plot(
                ws_id,
                np.asarray(y_test),
                np.asarray(y_pred_test),
                labels,
            )
        except Exception as e:
            ctx["train_error"] = f"Failed to generate confusion matrix: {e}"

    # Cache last train to enable quick "Save Model" (cap at 20 entries to prevent unbounded growth)
    _LAST_TRAIN_MAX = 20
    if ws_id not in LAST_TRAIN and len(LAST_TRAIN) >= _LAST_TRAIN_MAX:
        del LAST_TRAIN[next(iter(LAST_TRAIN))]
    LAST_TRAIN[ws_id] = {
        "estimator": est,
        "params": ctx["train_params"],
        "inputs": inputs,
        "target": target,
        "csv_path": csv_path,
        "metrics_train": ctx["metrics_train"],
        "metrics_test": ctx["metrics_test"],
        "parity_img_url": ctx.get("parity_img_url"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Refresh saved runs list in the page
    ctx["saved_runs"] = _list_saved_runs()
        
    return templates.TemplateResponse("train.html", ctx)
