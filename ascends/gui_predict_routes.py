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


def create_predict_router(
    *,
    templates: Jinja2Templates,
    runs_dir: Path,
    list_saved_runs: Callable[[], list[dict[str, Any]]],
    slugify_name: Callable[[str], str],
) -> APIRouter:
    """Create routes for prediction page, prediction run, and CSV download."""
    router = APIRouter()

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
            return templates.TemplateResponse("predict.html", ctx)

        man_path = runs_dir / run_name / "manifest.json"
        if not man_path.exists():
            ctx["predict_errors"] = [f"Run '{run_name}' is missing manifest.json."]
            return templates.TemplateResponse("predict.html", ctx)
        try:
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception as e:
            ctx["predict_errors"] = [f"Failed to read manifest.json for '{run_name}': {e}"]
            return templates.TemplateResponse("predict.html", ctx)

        inputs: list[str] = manifest.get("inputs", []) or []
        target: Optional[str] = manifest.get("target") or None
        if not inputs:
            ctx["predict_errors"] = [f"Run '{run_name}' has no recorded input features in manifest.json."]
            return templates.TemplateResponse("predict.html", ctx)

        try:
            raw = await csvfile.read()
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            ctx["predict_errors"] = [f"Failed to parse uploaded CSV: {e}"]
            return templates.TemplateResponse("predict.html", ctx)

        if df.empty:
            ctx["predict_errors"] = ["Uploaded CSV is empty."]
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
            return templates.TemplateResponse("predict.html", ctx)

        aligned_cols = [mapping[feature] for feature in inputs]
        df_aligned = df[aligned_cols].copy()
        for column in df_aligned.columns:
            df_aligned[column] = pd.to_numeric(df_aligned[column], errors="coerce")

        rows_read = len(df_aligned)
        df_used = df_aligned.dropna(axis=0, how="any")
        rows_used = len(df_used)
        dropped = rows_read - rows_used

        if rows_used == 0:
            ctx["predict_errors"] = [
                f"All {rows_read} rows contained NA/invalid values in required inputs; nothing to predict."
            ]
            return templates.TemplateResponse("predict.html", ctx)

        model_path = runs_dir / run_name / "model.joblib"
        if not model_path.exists():
            ctx["predict_errors"] = [f"Run '{run_name}' is missing model.joblib."]
            return templates.TemplateResponse("predict.html", ctx)
        try:
            est = load(model_path)
        except Exception as e:
            ctx["predict_errors"] = [f"Failed to load model.joblib: {e}"]
            return templates.TemplateResponse("predict.html", ctx)

        try:
            preds = est.predict(df_used)
        except Exception as e:
            ctx["predict_errors"] = [f"Prediction failed: {e}"]
            return templates.TemplateResponse("predict.html", ctx)

        pred_col = f"{target}_pred" if target else "prediction"
        result_df = df_used.copy()
        result_df[pred_col] = preds

        pred_dir = runs_dir / run_name / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        try:
            stem = Path(csvfile.filename).stem if csvfile.filename else "input"
        except Exception:
            stem = "input"
        safe_stem = slugify_name(stem) or "input"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{safe_stem}_{ts}_pred.csv"
        out_path = pred_dir / out_name
        try:
            result_df.to_csv(out_path, index=False)
        except Exception as e:
            ctx["predict_errors"] = [f"Failed to save predictions CSV: {e}"]
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
        ctx["download_csv_url"] = f"/predict/download?run={quote(run_name)}&file={quote(out_name)}"
        ctx["predict_errors"] = None
        return templates.TemplateResponse("predict.html", ctx)

    @router.get("/predict/download")
    async def predict_download(
        run: str = Query(..., description="Saved run name"),
        file: str = Query(..., description="Predictions filename in the run's predictions directory"),
    ):
        """Serve a predictions CSV from runs/<run>/predictions/<file>."""
        pred_dir = (runs_dir / run / "predictions").resolve()
        file_path = (pred_dir / file).resolve()
        try:
            file_path.relative_to(pred_dir)
        except ValueError:
            return HTMLResponse(status_code=404, content="Not found")
        if not file_path.is_file():
            return HTMLResponse(status_code=404, content="Not found")
        return FileResponse(file_path, media_type="text/csv", filename=file)

    return router
