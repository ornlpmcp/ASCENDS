from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any
from fastapi import Form, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
import pandas as pd
import time
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ascends.gui_correlation_routes import create_correlation_router
from ascends.gui_predict_routes import create_predict_router
from ascends.gui_run_registry import (
    RUNS_DIR,
    list_saved_runs as _list_saved_runs,
    slugify_name as _slugify_name,
    unique_run_name as _unique_run_name,
)
from ascends.gui_shap_routes import create_shap_router
from ascends.gui_saved_run_routes import create_saved_run_router
from ascends.gui_train_select_routes import create_train_select_router
from ascends.gui_train_run_routes import create_train_run_router
from ascends.gui_workflow import build_workflow_steps, create_sample_workspace

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
            enricher = getattr(self, "context_enricher", None)
            if enricher is not None and context is not None:
                enricher(context)
            request = (context or {}).get("request")
            return super().TemplateResponse(request, name, context, *args[2:], **kwargs)
        if len(args) >= 2 and isinstance(args[1], str):
            context = args[2] if len(args) > 2 else kwargs.get("context")
            enricher = getattr(self, "context_enricher", None)
            if enricher is not None and context is not None:
                enricher(context)
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


def _add_workflow_context(ctx: Dict[str, Any]) -> None:
    """Add workflow step state to every GUI template context."""
    request = ctx.get("request")
    path = request.url.path if request and request.url else ""
    ws = ctx.get("ws_id")
    if not ws and request:
        ws = request.query_params.get("ws_id")
    mf = _load_manifest(ws) if ws else {}
    if mf:
        ctx.setdefault("ws_id", ws)
        ctx.setdefault("inputs", mf.get("inputs", []))
        ctx.setdefault("target", mf.get("target"))
    saved_runs = ctx.get("saved_runs")
    if saved_runs is None:
        saved_runs = _list_saved_runs()
    ctx["workflow_steps"] = build_workflow_steps(
        path=path,
        ws_id=ws,
        manifest={**mf, "inputs": ctx.get("inputs", mf.get("inputs", [])), "target": ctx.get("target", mf.get("target"))},
        saved_runs=saved_runs,
        trained=bool(ws and ws in LAST_TRAIN) or bool(ctx.get("metrics_test")),
        prediction_done=bool(ctx.get("download_csv_url") or ctx.get("predict_preview_rows") or ctx.get("predict_preview_html")),
    )


templates.context_enricher = _add_workflow_context


def _available_samples() -> list[dict[str, str]]:
    samples = [
        {
            "kind": "regression",
            "label": "Start with Sample Regression Data",
            "path": BASE_DIR / "examples" / "BostonHousing.csv",
            "target": "medv",
            "task": "r",
        },
        {
            "kind": "classification",
            "label": "Start with Sample Classification Data",
            "path": BASE_DIR / "examples" / "iris.csv",
            "target": "Name",
            "task": "c",
        },
    ]
    return [
        {**sample, "path": str(sample["path"])}
        for sample in samples
        if Path(sample["path"]).exists()
    ]


@app.get("/favicon.ico")
@app.get("/favicon.svg")
async def _favicon_svg():
    # Serve the SVG to common favicon requests.
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
    return templates.TemplateResponse("home.html", {"request": request, "sample_data": _available_samples()})


@app.post("/sample/start")
async def start_sample(kind: str = Form(...)) -> RedirectResponse:
    sample_by_kind = {sample["kind"]: sample for sample in _available_samples()}
    sample = sample_by_kind.get(kind)
    if not sample:
        return RedirectResponse(url="/", status_code=303)
    ws_id = create_sample_workspace(
        sample_csv=Path(sample["path"]),
        workspace_dir=WORKSPACE_DIR,
        target=sample["target"],
        task=sample["task"],
    )
    return RedirectResponse(url=f"/correlation?ws_id={quote(ws_id)}", status_code=303)

@app.get("/ui-lab", response_class=HTMLResponse)
async def ui_lab(request: Request) -> HTMLResponse:
    """TS + utility-first UI proof page."""
    return templates.TemplateResponse("ui_lab.html", {"request": request})


# Replace /train GET with context that loads manifest using ws_id or cookie
# Replace the /train GET to load manifest by ws_id (from query or cookie)
# Replace the /train GET with a version that logs what it sees
@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request, ws_id: Optional[str] = None) -> HTMLResponse:
    ws = ws_id or request.query_params.get("ws_id")
    ctx: Dict[str, Any] = {"request": request, "ws_id": ws}
    if ws:
        mf = _load_manifest(ws) or {}
        shap_view = str(mf.get("shap_view", "default")).lower()
        if shap_view not in {"ascends", "default"}:
            shap_view = "default"
        ctx.update({
            "csv_path": mf.get("csv_path"),
            "all_columns": mf.get("columns", []),
            "selected": mf.get("selected", []),
            "inputs": mf.get("inputs", []),
            "target": mf.get("target"),
            "train_params": mf.get("train_params", {"task": mf.get("task", "r")}),
            "shap_view": shap_view,
        })

        # Prefer SHAP's beeswarm view, then fall back to ASCENDS' bar view for older runs.
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


app.include_router(
    create_train_select_router(
        templates=templates,
        load_manifest=_load_manifest,
        save_manifest=_save_manifest,
        list_saved_runs=_list_saved_runs,
    )
)


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

# Cache the last trained estimator & context by workspace (for quick Save)
LAST_TRAIN: Dict[str, Any] = {}


app.include_router(
    create_train_run_router(
        templates=templates,
        static_dir=STATIC_DIR,
        load_manifest=_load_manifest,
        list_saved_runs=_list_saved_runs,
        last_train=LAST_TRAIN,
    )
)

train_run = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/train/run")


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
    create_saved_run_router(
        templates=templates,
        runs_dir=RUNS_DIR,
        static_dir=STATIC_DIR,
        ws_dir=_ws_dir,
        load_manifest=_load_manifest,
        list_saved_runs=_list_saved_runs,
        unique_run_name=_unique_run_name,
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
