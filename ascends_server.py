from __future__ import annotations

import logging
from pathlib import Path
from joblib import dump
from datetime import datetime
import shutil
import json
from typing import Optional, Dict, Any, List
from fastapi import Request, Form
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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ascends.gui_plotting import save_confusion_plot, save_parity_plot, train_img_dir
from ascends.gui_correlation_routes import create_correlation_router
from ascends.gui_predict_routes import create_predict_router
from ascends.gui_run_registry import (
    RUNS_DIR,
    list_saved_runs as _list_saved_runs,
    slugify_name as _slugify_name,
    unique_run_name as _unique_run_name,
)
from ascends.gui_shap_routes import create_shap_router

logger = logging.getLogger("ascends.gui")

app = FastAPI(title="ASCENDS GUI", version="0.1.0")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
WORKSPACE_DIR = BASE_DIR / "workspace"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"

def _ws_dir(ws_id: str) -> Path:
    """Workspace directory for a given session id."""
    return WORKSPACE_DIR / ws_id

TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ASCENDSJinja2Templates(Jinja2Templates):
    """Keep existing template calls compatible with Starlette 1.0."""

    def TemplateResponse(self, *args, **kwargs):  # noqa: N802
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", None)
            request = (context or {}).get("request")
            return super().TemplateResponse(request, name, context, *args[2:], **kwargs)
        return super().TemplateResponse(*args, **kwargs)


templates = ASCENDSJinja2Templates(directory=str(TEMPLATES_DIR))


def _manifest_path(ws_id: str) -> Path:
    return _ws_dir(ws_id) / "manifest.json"


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

app.include_router(
    create_correlation_router(
        templates=templates,
        workspace_dir=WORKSPACE_DIR,
        static_dir=STATIC_DIR,
        uploads_dir=UPLOADS_DIR,
        load_manifest=_load_manifest,
        save_manifest=_save_manifest,
    )
)

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

# Cache the last trained estimator & context by workspace (for quick Save)
LAST_TRAIN: Dict[str, Any] = {}


app.include_router(
    create_shap_router(
        templates=templates,
        static_dir=STATIC_DIR,
        ws_dir=_ws_dir,
        load_manifest=_load_manifest,
        save_manifest=_save_manifest,
        list_saved_runs=_list_saved_runs,
        last_train=LAST_TRAIN,
    )
)


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
