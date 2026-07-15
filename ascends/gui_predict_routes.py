"""Predict-tab routes for the ASCENDS FastAPI GUI."""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from joblib import load

from ascends.core.data import NON_ASCII_COLUMN_MESSAGE, warn_non_ascii_columns
from ascends.core.predict import prepare_prediction_frame
from ascends.gui_messages import (
    append_notice,
    attach_error_recovery,
    friendly_error,
    rows_removed_message,
)


def _safe_run_dir(runs_dir: Path, run_name: str) -> Path:
    """Resolve a prediction run directory while preventing traversal outside runs_dir."""
    if not run_name or "/" in run_name or "\\" in run_name or run_name in {".", ".."}:
        raise ValueError("Invalid run name.")
    root = runs_dir.resolve()
    target = (runs_dir / run_name).resolve()
    if target.parent != root:
        raise ValueError("Invalid run name.")
    return target


def _safe_predictions_dir(run_dir: Path) -> Path:
    """Resolve a run's predictions directory while keeping it inside that run."""
    predictions_dir = (run_dir / "predictions").resolve()
    if predictions_dir.parent != run_dir:
        raise ValueError("Invalid predictions directory.")
    return predictions_dir


def create_predict_router(
    *,
    templates: Jinja2Templates,
    runs_dir: Path,
    list_saved_runs: Callable[[], list[dict[str, Any]]],
    slugify_name: Callable[[str], str],
) -> APIRouter:
    """Create routes for prediction page, prediction run, and CSV download."""
    router = APIRouter()

    def _add_non_ascii_notice(ctx: dict[str, Any], columns) -> None:
        columns_with_non_ascii = warn_non_ascii_columns(columns)
        if columns_with_non_ascii:
            append_notice(
                ctx,
                f"{NON_ASCII_COLUMN_MESSAGE} Columns: {', '.join(columns_with_non_ascii)}",
                level="warning",
            )

    @router.get("/predict", response_class=HTMLResponse)
    async def predict_page(request: Request, run: Optional[str] = None) -> HTMLResponse:
        """Render Predict tab with saved runs and optional preselected run."""
        selected_run = run or request.query_params.get("run")
        ctx: dict[str, Any] = {
            "request": request,
            "saved_runs": list_saved_runs(),
            "selected_run": selected_run,
        }
        return templates.TemplateResponse("predict.html", ctx)

    @router.post("/predict/run", response_class=HTMLResponse)
    async def predict_run(
        request: Request,
        run_name: str = Form(...),
        csvfile: UploadFile = File(...),
    ) -> HTMLResponse:
        """Schema validation, prediction, CSV save, and preview for uploaded features."""
        errors: list[str] = []
        ctx: dict[str, Any] = {
            "request": request,
            "saved_runs": list_saved_runs(),
            "selected_run": run_name,
            "predict_summary": None,
            "predict_preview_headers": None,
            "predict_preview_rows": None,
            "download_csv_url": None,
            "download_xlsx_url": None,
        }

        if not run_name:
            errors.append("Please select a saved model (run).")
        if not csvfile or not csvfile.filename:
            errors.append("Please upload a CSV file.")
        if errors:
            ctx["predict_errors"] = errors
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        try:
            run_dir = _safe_run_dir(runs_dir, run_name)
        except ValueError as e:
            ctx["predict_errors"] = [str(e)]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx, status_code=400)

        man_path = run_dir / "manifest.json"
        if not man_path.exists():
            ctx["predict_errors"] = [f"Run '{run_name}' is missing manifest.json."]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)
        try:
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception as e:
            ctx["predict_errors"] = [friendly_error(e, "predict")]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        inputs: list[str] = manifest.get("inputs", []) or []
        features: list[str] = manifest.get("features", []) or []
        target: Optional[str] = manifest.get("target") or None
        if not inputs and not features:
            ctx["predict_errors"] = [
                f"Run '{run_name}' has no recorded input features in manifest.json."
            ]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        try:
            raw = await csvfile.read()
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            ctx["predict_errors"] = [friendly_error(e, "predict")]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)
        _add_non_ascii_notice(ctx, df.columns)

        if df.empty:
            ctx["predict_errors"] = ["Uploaded CSV is empty."]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        lower_candidates: dict[str, list[str]] = {}
        for column in list(df.columns):
            lower_candidates.setdefault(str(column).lower(), []).append(column)

        mapping: dict[str, str] = {}
        missing: list[str] = []
        for feature in inputs:
            if feature in df.columns:
                mapping[feature] = feature
                continue
            candidates = lower_candidates.get(str(feature).lower(), [])
            if len(candidates) == 1:
                mapping[feature] = candidates[0]
            else:
                missing.append(feature)

        if missing:
            ctx["predict_errors"] = [
                "Missing required feature(s) in CSV (case-insensitive match failed): "
                + ", ".join(missing)
            ]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        encoded_schema = bool(features) and (
            not inputs or set(features) != set(inputs)
        )
        if encoded_schema:
            renamed = df.rename(columns={actual: expected for expected, actual in mapping.items()})
            df_aligned = prepare_prediction_frame(renamed, manifest)
            output_rows = df.copy()
        else:
            aligned_cols = [mapping[feature] for feature in inputs]
            df_aligned = df[aligned_cols].copy()
            for column in df_aligned.columns:
                df_aligned[column] = pd.to_numeric(df_aligned[column], errors="coerce")
            output_rows = df_aligned

        rows_read = len(df_aligned)
        df_used = df_aligned.dropna(axis=0, how="any")
        rows_used = len(df_used)
        dropped = rows_read - rows_used
        if dropped > 0:
            append_notice(ctx, rows_removed_message(dropped), level="info")

        if rows_used == 0:
            ctx["predict_errors"] = [
                f"All {rows_read} rows contained NA/invalid values in required inputs; nothing to predict."
            ]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        model_path = run_dir / "model.joblib"
        if not model_path.exists():
            ctx["predict_errors"] = [f"Run '{run_name}' is missing model.joblib."]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)
        try:
            est = load(model_path)
        except Exception as e:
            ctx["predict_errors"] = [friendly_error(e, "predict")]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        try:
            preds = est.predict(df_used)
        except Exception as e:
            ctx["predict_errors"] = [friendly_error(e, "predict")]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        pred_col = f"{target}_pred" if target else "prediction"
        result_df = output_rows.loc[df_used.index].copy()
        result_df[pred_col] = preds

        try:
            pred_dir = _safe_predictions_dir(run_dir)
            pred_dir.mkdir(parents=True, exist_ok=True)
        except ValueError as e:
            ctx["predict_errors"] = [str(e)]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx, status_code=400)
        try:
            stem = Path(csvfile.filename).stem if csvfile.filename else "input"
        except Exception:
            stem = "input"
        safe_stem = slugify_name(stem) or "input"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{safe_stem}_{ts}_pred.csv"
        out_path = (pred_dir / out_name).resolve()
        if out_path.parent != pred_dir:
            ctx["predict_errors"] = ["Invalid prediction filename."]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx, status_code=400)
        try:
            result_df.to_csv(out_path, index=False)
        except Exception as e:
            ctx["predict_errors"] = [friendly_error(e, "predict")]
            attach_error_recovery(ctx, "predict")
            return templates.TemplateResponse("predict.html", ctx)

        preview_df = result_df.head(5)
        try:
            ctx["predict_preview_html"] = preview_df.to_html(
                classes="table",
                index=False,
                border=0,
                float_format=lambda x: f"{x:.3f}",
            )
        except Exception:
            preview = preview_df.astype(object).where(pd.notnull(preview_df), None)
            ctx["predict_preview_headers"] = list(preview.columns)
            ctx["predict_preview_rows"] = preview.values.tolist()

        ctx["rows_read"] = rows_read
        ctx["rows_used"] = rows_used
        ctx["rows_dropped"] = dropped
        ctx["saved_relpath"] = f"runs/{run_name}/predictions/{out_name}"
        ctx["predict_summary"] = None
        ctx["download_csv_url"] = (
            f"/predict/download?run={quote(run_name)}&file={quote(out_name)}"
        )
        ctx["predict_errors"] = None
        return templates.TemplateResponse("predict.html", ctx)

    @router.get("/predict/download")
    async def predict_download(
        run: str = Query(..., description="Saved run name"),
        file: str = Query(
            ..., description="Predictions filename in the run's predictions directory"
        ),
    ):
        """Serve a predictions CSV from runs/<run>/predictions/<file>."""
        try:
            pred_dir = _safe_predictions_dir(_safe_run_dir(runs_dir, run))
        except ValueError:
            return HTMLResponse(status_code=404, content="Not found")
        file_path = (pred_dir / file).resolve()
        try:
            file_path.relative_to(pred_dir)
        except ValueError:
            return HTMLResponse(status_code=404, content="Not found")
        if not file_path.is_file():
            return HTMLResponse(status_code=404, content="Not found")
        return FileResponse(file_path, media_type="text/csv", filename=file)

    return router
