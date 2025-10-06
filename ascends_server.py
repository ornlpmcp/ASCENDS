from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None  # xgb optional; we'll fallback if absent

import json
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")  # headless, no GUI windows
import matplotlib.pyplot as plt
import dcor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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


async def _save_csv(file: UploadFile) -> Path:
    name = _safe_csv_filename(file.filename or "data.csv")
    dest = UPLOADS_DIR / name
    content = await file.read()
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


def _plot_metric_bars(
    scores: pd.DataFrame,
    metric: str,
    target: str,
    n_used: int,
    out_png: Path,
    top_k: Optional[int] = None,
) -> None:
    # Sort & limit to Top-K
    dfp = scores.copy()
    if metric in {"pearson", "spearman"}:
        dfp = dfp.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
    else:
        dfp = dfp.sort_values(by="score", ascending=False)
    if top_k and top_k > 0:
        dfp = dfp.head(top_k)

    # Figure size (golden ratio), tuned a bit for readability
    fig_w = 8.0
    fig_h = fig_w / 1.618  # golden ratio
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    # Vertical bars: features on X axis
    x = np.arange(len(dfp))
    ax.bar(x, dfp["score"])
    ax.set_xticks(x)
    ax.set_xticklabels(list(dfp["feature"]), rotation=55, ha="right")

    ax.set_xlabel("Feature")
    ax.set_ylabel("Score")
    ax.set_title(f"{metric.title()} vs. {target}  (N={n_used})")

    # Grid and baseline for signed metrics
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if metric in {"pearson", "spearman"}:
        ax.axhline(0.0, linewidth=0.8, alpha=0.6, color="black")

    # Tight layout with extra bottom room for labels
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


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


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "port": 7777}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("home.html", {"request": request})


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
        ctx.update({
            "csv_path": mf.get("csv_path"),
            "all_columns": mf.get("columns", []),
            "selected": mf.get("selected", []),
            "inputs": mf.get("inputs", []),
            "target": mf.get("target"),
        })
    return templates.TemplateResponse("train.html", ctx)

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request, run: Optional[str] = None) -> HTMLResponse:
    return templates.TemplateResponse("predict.html", {"request": request})

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
                df = pd.read_csv(mf.get("csv_path"), nrows=10)
                ctx["preview_headers"] = list(df.columns)
                ctx["preview_rows"] = df.astype(object).where(pd.notnull(df), None).values.tolist()
            except Exception:
                pass
            return templates.TemplateResponse("correlation.html", ctx)
        mf["corr"] = {}
    mf["corr"].update({
        "metrics": chosen_metrics,
        "top_k": top_k_val,
        "task": task,
        "view": view,
    })
    # Prepare data and write cleaning log
    data_dir, img_dir = _corr_dirs(ws_id)
    df_clean, info = _prepare_corr_dataframe(mf["csv_path"], target, inputs)
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
            if cur_target in cur_inputs:
                cur_inputs.discard(cur_target)
        if chosen_cols:
            cur_selected = set(chosen_cols)

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
    import uvicorn  # local import to avoid E402
    uvicorn.run("ascends_server:app", host="127.0.0.1", port=7777, reload=True)
# Helper: unique while preserving order
def _unique_preserve(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# Helper: pick a regressor by key
def _make_regressor(key: str):
    k = (key or "rf").lower()
    if k == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    if k == "xgb" and xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            random_state=42, tree_method="hist", n_jobs=0, verbosity=0
        )
    if k == "hgb":
        return HistGradientBoostingRegressor(random_state=42)
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
    return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

# Replace the existing /train/run (from Step A) with this version:
@app.post("/train/run", response_class=HTMLResponse)
async def train_run(
    request: Request,
    ws_id: str = Form(...),
    task: str = Form(...),          # "r" or "c"
    model: str = Form(...),         # rf/xgb/hgb/svr/knn/linear/ridge/lasso/elastic
    test_size: float = Form(...),   # e.g., 0.2
    tune: str = Form(...),          # off/quick/intense/optuna/bayes (ignored in Step B)
):
    # Build base context
    ctx: Dict[str, Any] = {"request": request, "ws_id": ws_id}
    mf = _load_manifest(ws_id) or {}
    all_columns = mf.get("columns", [])
    inputs = mf.get("inputs", [])
    target = mf.get("target")

    ctx.update({
        "csv_path": mf.get("csv_path"),
        "all_columns": all_columns,
        "selected": mf.get("selected", []),
        "inputs": inputs,
        "target": target,
        "train_params": {"task": task, "model": model, "test_size": test_size, "tune": tune},
    })

    # Guardrails
    if task != "r":
        ctx["train_error"] = "Classification will be added later. Please use Regression for now."
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

    # Model
    est = _make_regressor(model)

    # Fit & predict
    try:
        est.fit(X_train, y_train)
        y_pred_train = est.predict(X_train)
        y_pred_test = est.predict(X_test)
    except Exception as e:
        ctx["train_error"] = f"Model training failed: {e}"
        return templates.TemplateResponse("train.html", ctx)

    # Metrics (train/test)
    def _metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        return {"R2": r2, "MAE": mae, "RMSE": rmse}

    ctx["metrics_train"] = _metrics(y_train, y_pred_train)
    ctx["metrics_test"]  = _metrics(y_test,  y_pred_test)

    # Save a small parity preview in context (used in Step C for plotting)
    ctx["parity_preview"] = {
        "train": {"actual": y_train.tolist(), "pred": y_pred_train.tolist()},
        "test":  {"actual": y_test.tolist(),  "pred": y_pred_test.tolist()},
    }
    return templates.TemplateResponse("train.html", ctx)
