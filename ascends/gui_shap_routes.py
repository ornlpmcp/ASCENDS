"""SHAP and feature-importance routes for the ASCENDS FastAPI GUI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ascends.core.data import prepare_numeric_features
from ascends.core.explain import (
    explain_model as core_explain,
    save_default_shap_plot,
    save_importance_plot,
)
from ascends.gui_plotting import train_img_dir


def _prepare_shap_frame(
    source: pd.DataFrame,
    inputs: list[str],
    target: str,
    task: str,
) -> pd.DataFrame:
    """Apply the same numeric coercion and row filtering used for GUI training."""
    if target not in source.columns:
        raise ValueError("Target column missing in source CSV.")
    missing_inputs = [column for column in inputs if column not in source.columns]
    if missing_inputs:
        raise ValueError(
            "Input column(s) missing in source CSV: " + ", ".join(missing_inputs)
        )

    prepared = prepare_numeric_features(source, inputs, target, task=task)
    if prepared.used_inputs != inputs:
        skipped = [column for column in inputs if column not in prepared.used_inputs]
        raise ValueError(
            "Training input column(s) are no longer usable: " + ", ".join(skipped)
        )
    if prepared.frame.empty:
        raise ValueError("No valid rows remain after applying the training data rules.")
    return prepared.frame


def _train_context(
    request: Request,
    ws_id: str,
    manifest: dict[str, Any],
    saved_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "request": request,
        "ws_id": ws_id,
        "csv_path": manifest.get("csv_path"),
        "all_columns": manifest.get("columns", []),
        "selected": manifest.get("selected", []),
        "inputs": manifest.get("inputs", []),
        "target": manifest.get("target"),
        "saved_runs": saved_runs,
    }


def _add_training_outputs(ctx: dict[str, Any], rec: dict[str, Any] | None) -> None:
    if not rec:
        return
    ctx["metrics_train"] = rec.get("metrics_train")
    ctx["metrics_test"] = rec.get("metrics_test")
    ctx["parity_img_url"] = rec.get("parity_img_url")


def _add_shap_table(ctx: dict[str, Any], ws_dir: Path) -> None:
    shap_csv = ws_dir / "train" / "shap_importance.csv"
    if not shap_csv.exists():
        return
    try:
        df_shap = pd.read_csv(shap_csv).head(10)
        ctx["shap_rows"] = df_shap.values.tolist()
    except Exception:
        pass


def create_shap_router(
    *,
    templates: Jinja2Templates,
    static_dir: Path,
    ws_dir: Callable[[str], Path],
    load_manifest: Callable[[str], dict[str, Any]],
    save_manifest: Callable[[str, dict[str, Any]], None],
    list_saved_runs: Callable[[], list[dict[str, Any]]],
    last_train: dict[str, Any],
) -> APIRouter:
    """Create routes for SHAP computation and SHAP view switching."""
    router = APIRouter()

    @router.post("/train/shap", response_class=HTMLResponse)
    async def train_shap(
        request: Request,
        ws_id: str = Form(...),
        max_samples: int = Form(300),
        shap_view: str = Form("default"),
    ) -> HTMLResponse:
        """Compute SHAP/permutation importance for the latest trained model."""
        mf = load_manifest(ws_id) or {}
        ctx = _train_context(request, ws_id, mf, list_saved_runs())
        shap_view = str(shap_view or "default").lower()
        if shap_view not in {"ascends", "default"}:
            shap_view = "default"
        ctx["shap_view"] = shap_view

        rec = last_train.get(ws_id)
        if not rec:
            ctx["train_error"] = (
                "No trained model found in this workspace. Train first, then run SHAP."
            )
            return templates.TemplateResponse("train.html", ctx)

        _add_training_outputs(ctx, rec)

        csv_path = rec.get("csv_path")
        inputs = rec.get("inputs", [])
        target = rec.get("target")
        task = rec.get("params", {}).get("task", "r")
        est = rec.get("estimator")
        if not csv_path or not est or not inputs or not target:
            ctx["train_error"] = (
                "Insufficient training context for SHAP. Re-train and try again."
            )
            return templates.TemplateResponse("train.html", ctx)

        try:
            df = pd.read_csv(csv_path)
            df2 = _prepare_shap_frame(df, list(inputs), target, str(task))
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

        data_dir = ws_dir(ws_id) / "train"
        data_dir.mkdir(parents=True, exist_ok=True)
        img_dir = train_img_dir(static_dir, ws_id)
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
                extra = f"SHAP beeswarm view failed ({e}); using ASCENDS bar view."
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

        selected_png = (
            png_default if (shap_view == "default" and default_ready) else png_ascends
        )
        ctx["shap_img_url"] = (
            f"/static/workspace/{ws_id}/train/{selected_png.name}?ts={int(time.time())}"
        )
        ctx["shap_rows"] = imp_df.head(10).values.tolist()
        if expl.get("warning"):
            ctx["shap_warning"] = expl["warning"]

        mf["shap_view"] = shap_view
        save_manifest(ws_id, mf)
        return templates.TemplateResponse("train.html", ctx)

    @router.post("/train/shap/view", response_class=HTMLResponse)
    async def train_shap_view(
        request: Request,
        ws_id: str = Form(...),
        shap_view: str = Form("default"),
    ) -> HTMLResponse:
        """Switch displayed SHAP image without recomputing model explanation."""
        mf = load_manifest(ws_id) or {}
        shap_view = str(shap_view or "default").lower()
        if shap_view not in {"ascends", "default"}:
            shap_view = "default"
        mf["shap_view"] = shap_view
        save_manifest(ws_id, mf)

        ctx = _train_context(request, ws_id, mf, list_saved_runs())
        ctx["shap_view"] = shap_view
        _add_training_outputs(ctx, last_train.get(ws_id))

        img_dir = train_img_dir(static_dir, ws_id)
        selected_png = img_dir / f"shap_importance_{shap_view}.png"
        fallback_png = img_dir / "shap_importance_ascends.png"
        legacy_png = img_dir / "shap_importance.png"
        if selected_png.exists():
            ctx["shap_img_url"] = (
                f"/static/workspace/{ws_id}/train/{selected_png.name}?ts={int(time.time())}"
            )
        elif fallback_png.exists():
            ctx["shap_img_url"] = (
                f"/static/workspace/{ws_id}/train/{fallback_png.name}?ts={int(time.time())}"
            )
            if shap_view == "default":
                ctx["shap_warning"] = (
                    "SHAP beeswarm view is not available for this run. Showing ASCENDS bar view."
                )
        elif legacy_png.exists():
            ctx["shap_img_url"] = (
                f"/static/workspace/{ws_id}/train/{legacy_png.name}?ts={int(time.time())}"
            )

        _add_shap_table(ctx, ws_dir(ws_id))
        return templates.TemplateResponse("train.html", ctx)

    return router
