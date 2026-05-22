"""Correlation-tab routes and helpers for the ASCENDS FastAPI GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import dcor
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ascends.core.data import NON_ASCII_COLUMN_MESSAGE, warn_non_ascii_columns
from ascends.gui_messages import (
    append_notice,
    attach_error_recovery,
    constant_columns_message,
    format_missing_columns_message,
    friendly_error,
    rows_removed_message,
)
from ascends.gui_plotting import plot_metric_bars

PREVIEW_NROWS = 5
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _safe_csv_filename(original: str) -> str:
    stem = Path(original).stem[:50] or "upload"
    return f"{stem}-{uuid4().hex[:8]}.csv"


async def _save_csv(file: UploadFile, uploads_dir: Path) -> Path:
    name = _safe_csv_filename(file.filename or "data.csv")
    dest = uploads_dir / name
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        mb = len(content) // (1024 * 1024)
        raise ValueError(f"File too large ({mb} MB). Maximum allowed is 50 MB.")
    dest.write_bytes(content)
    return dest


def _corr_dirs(workspace_dir: Path, static_dir: Path, ws_id: str) -> tuple[Path, Path]:
    """Return (data_dir, img_dir) for correlation artifacts."""
    data_dir = workspace_dir / ws_id / "corr"
    img_dir = static_dir / "workspace" / ws_id / "corr"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, img_dir


def _compute_correlations(
    df: pd.DataFrame,
    target: str,
    inputs: list[str],
    metrics: list[str],
    task: str,
) -> dict[str, pd.DataFrame]:
    if task == "c":
        y = pd.Series(df[target]).astype("category").cat.codes.values
    else:
        y = df[target].values
    x_frame = df[inputs]
    out: dict[str, pd.DataFrame] = {}

    if "pearson" in metrics:
        vals = []
        for column in inputs:
            x = df[column].values
            if np.std(x) == 0 or np.std(y) == 0:
                vals.append((column, 0.0))
            else:
                r = np.corrcoef(x, y)[0, 1]
                vals.append((column, float(r)))
        out["pearson"] = pd.DataFrame(vals, columns=["feature", "score"]).sort_values(
            by="score", key=lambda s: np.abs(s), ascending=False
        )

    if "spearman" in metrics:
        vals = []
        y_rank = pd.Series(y).rank(method="average").values
        for column in inputs:
            x_rank = df[column].rank(method="average").values
            if np.std(x_rank) == 0 or np.std(y_rank) == 0:
                vals.append((column, 0.0))
            else:
                r = np.corrcoef(x_rank, y_rank)[0, 1]
                vals.append((column, float(r)))
        out["spearman"] = pd.DataFrame(vals, columns=["feature", "score"]).sort_values(
            by="score", key=lambda s: np.abs(s), ascending=False
        )

    if "mi" in metrics:
        if task == "c":
            mi_vals = mutual_info_classif(x_frame.values, y, random_state=0)
        else:
            mi_vals = mutual_info_regression(x_frame.values, y, random_state=0)
        out["mi"] = pd.DataFrame({"feature": inputs, "score": mi_vals}).sort_values(
            by="score", ascending=False
        )

    if "dcor" in metrics:
        vals = []
        y_f = np.asarray(y, dtype=np.float64)
        n = len(y_f)
        if n > 5000:
            rng = np.random.RandomState(0)
            idx = rng.choice(n, 5000, replace=False)
            y_f = y_f[idx]
            x_sub = x_frame.iloc[idx, :]
        else:
            x_sub = x_frame
        for column in inputs:
            x = np.asarray(x_sub[column].values, dtype=np.float64)
            try:
                score = float(dcor.distance_correlation(x, y_f))
            except Exception:
                score = 0.0
            vals.append((column, score))
        out["dcor"] = pd.DataFrame(vals, columns=["feature", "score"]).sort_values(
            by="score", ascending=False
        )

    return out


def _prepare_corr_dataframe(
    csv_path: str,
    target: str,
    inputs: list[str],
    task: str = "r",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(csv_path)
    cols = list(inputs) + [target]
    missing = [column for column in cols if column not in raw.columns]
    existing = [column for column in cols if column in raw.columns]
    df = raw.loc[:, existing].copy()
    for column in existing:
        if task == "c" and column == target:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    rows_before = len(df)
    df = df.dropna(axis=0, how="any")
    rows_after = len(df)
    numeric_columns = [column for column in df.columns if not (task == "c" and column == target)]
    df.loc[:, numeric_columns] = df.loc[:, numeric_columns].astype("float64")
    dropped = rows_before - rows_after

    skipped: list[str] = []
    good_inputs: list[str] = []
    for column in inputs:
        if column in df.columns:
            if df[column].nunique(dropna=True) <= 1:
                skipped.append(column)
            else:
                good_inputs.append(column)

    keep_cols = [column for column in good_inputs if column in df.columns]
    if target in df.columns:
        keep_cols.append(target)
    df = df.loc[:, keep_cols]
    info: dict[str, Any] = {
        "rows_in": rows_before,
        "rows_used": rows_after,
        "rows_dropped": dropped,
        "skipped_inputs": skipped,
        "used_inputs": good_inputs,
        "missing_columns": missing,
    }
    return df, info


def _add_preview(ctx: dict[str, Any], csv_path: str | None, nrows: int = PREVIEW_NROWS) -> None:
    if not csv_path:
        return
    try:
        df = pd.read_csv(csv_path, nrows=nrows)
        ctx["preview_headers"] = list(df.columns)
        ctx["preview_rows"] = df.astype(object).where(pd.notnull(df), None).values.tolist()
    except Exception:
        pass


def _add_non_ascii_notice(ctx: dict[str, Any], columns) -> None:
    columns_with_non_ascii = warn_non_ascii_columns(columns)
    if columns_with_non_ascii:
        append_notice(
            ctx,
            f"{NON_ASCII_COLUMN_MESSAGE} Columns: {', '.join(columns_with_non_ascii)}",
            level="warning",
        )


def create_correlation_router(
    *,
    templates: Jinja2Templates,
    workspace_dir: Path,
    static_dir: Path,
    uploads_dir: Path,
    load_manifest: Callable[[str], dict[str, Any]],
    save_manifest: Callable[[str, dict[str, Any]], None],
) -> APIRouter:
    """Create routes for correlation upload, selection, execution, and downloads."""
    router = APIRouter()

    @router.get("/correlation", response_class=HTMLResponse)
    async def correlation_page(request: Request, ws_id: Optional[str] = None) -> HTMLResponse:
        ctx: dict[str, Any] = {"request": request}
        mf: dict[str, Any] = {}
        if ws_id:
            mf = load_manifest(ws_id)
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
                _add_preview(ctx, mf.get("csv_path"))

        corr = mf.get("corr", {}) if ws_id else {}
        artifacts = corr.get("artifacts") or {}
        if artifacts:
            available = list(artifacts.keys())
            active_metric = corr.get("active_metric") or (available[0] if available else None)
            chart_url = artifacts.get(active_metric, {}).get("png")
            metric_rows = []
            try:
                csv_path = artifacts.get(active_metric, {}).get("csv")
                if csv_path:
                    dfm = pd.read_csv(csv_path)
                    if active_metric in {"pearson", "spearman"}:
                        dfm = dfm.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
                    else:
                        dfm = dfm.sort_values(by="score", ascending=False)
                    top_k = corr.get("top_k")
                    if top_k and top_k > 0:
                        dfm = dfm.head(top_k)
                    metric_rows = dfm.values.tolist()
            except Exception:
                pass
            ctx.update(
                {
                    "available_metrics": available,
                    "active_metric": active_metric,
                    "chart_url": chart_url,
                    "metric_rows": metric_rows,
                }
            )
        return templates.TemplateResponse("correlation.html", ctx)

    @router.post("/correlation/run", response_class=HTMLResponse)
    async def correlation_run(
        request: Request,
        ws_id: str = Form(...),
        metrics: Optional[list[str]] = Form(None),
        top_k: Optional[str] = Form(None),
        task: str = Form("r"),
        view: str = Form("wide"),
    ) -> HTMLResponse:
        mf = load_manifest(ws_id)
        if not mf:
            ctx = {"request": request, "error": "Invalid session. Please re-upload CSV."}
            attach_error_recovery(ctx, "correlation")
            return templates.TemplateResponse(
                "correlation.html",
                ctx,
            )
        cols = mf.get("columns", [])
        inputs = mf.get("inputs", [])
        target = mf.get("target")
        base_ctx = {
            "request": request,
            "ws_id": ws_id,
            "csv_path": mf.get("csv_path"),
            "all_columns": cols,
            "inputs": inputs,
            "target": target,
            "selected": mf.get("selected", []),
        }
        if not target:
            ctx = {**base_ctx, "error": "Please set a target column before running correlation."}
            attach_error_recovery(ctx, "correlation", ws_id=ws_id)
            return templates.TemplateResponse(
                "correlation.html",
                ctx,
            )
        if not inputs:
            ctx = {**base_ctx, "error": "Please select at least one input feature."}
            attach_error_recovery(ctx, "correlation", ws_id=ws_id)
            return templates.TemplateResponse(
                "correlation.html",
                ctx,
            )

        corr_section = mf.setdefault("corr", {})
        chosen_metrics = metrics or ["pearson", "spearman", "mi", "dcor"]
        top_k_val = None
        if top_k is not None and str(top_k).strip() != "":
            try:
                top_k_val = int(str(top_k).strip())
                if top_k_val <= 0:
                    raise ValueError("top_k must be > 0")
            except Exception as e:
                ctx = {**base_ctx, "error": friendly_error(e, "correlation")}
                attach_error_recovery(ctx, "correlation", ws_id=ws_id)
                _add_preview(ctx, mf.get("csv_path"))
                return templates.TemplateResponse("correlation.html", ctx)

        corr_section["metrics"] = chosen_metrics
        corr_section["top_k"] = top_k_val
        corr_section["task"] = task
        corr_section["view"] = view

        data_dir, img_dir = _corr_dirs(workspace_dir, static_dir, ws_id)
        df_clean, info = _prepare_corr_dataframe(mf["csv_path"], target, inputs, task=task)
        mf["corr"]["used_inputs"] = info.get("used_inputs", [])
        (data_dir / "corr_log.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        ctx = {**base_ctx, "corr_info": info}
        if info.get("rows_dropped", 0) > 0:
            append_notice(ctx, rows_removed_message(info["rows_dropped"]), level="info")
        skipped_inputs = info.get("skipped_inputs", [])
        if skipped_inputs:
            append_notice(ctx, constant_columns_message(skipped_inputs), level="warning")
        _add_non_ascii_notice(ctx, df_clean.columns)
        missing_columns = info.get("missing_columns", [])
        if missing_columns:
            message = format_missing_columns_message(missing_columns)
            if target in missing_columns:
                ctx["error"] = message
                attach_error_recovery(ctx, "correlation", ws_id=ws_id)
                _add_preview(ctx, mf.get("csv_path"))
                return templates.TemplateResponse("correlation.html", ctx)
            append_notice(ctx, message, level="warning")
        _add_preview(ctx, mf.get("csv_path"))
        try:
            used_inputs = mf["corr"].get("used_inputs", inputs)
            if not used_inputs:
                ctx["error"] = "No usable input features after cleaning (all constant or dropped)."
                attach_error_recovery(ctx, "correlation", ws_id=ws_id)
                return templates.TemplateResponse("correlation.html", ctx)

            chosen_metrics = mf["corr"].get("metrics", ["pearson", "spearman", "mi", "dcor"])
            results = _compute_correlations(
                df_clean,
                target,
                used_inputs,
                chosen_metrics,
                mf["corr"].get("task", "r"),
            )

            artifacts: dict[str, dict[str, str]] = {}
            for metric_name, dfm in results.items():
                csv_path = data_dir / f"{metric_name}.csv"
                dfm.to_csv(csv_path, index=False)
                png_path = img_dir / f"{metric_name}.png"
                plot_metric_bars(
                    dfm,
                    metric_name,
                    target,
                    info["rows_used"],
                    png_path,
                    top_k=mf["corr"].get("top_k"),
                )
                artifacts[metric_name] = {
                    "csv": str(csv_path),
                    "png": f"/static/workspace/{ws_id}/corr/{metric_name}.png",
                }

            mf["corr"]["artifacts"] = artifacts
            save_manifest(ws_id, mf)

            active_metric = next(iter(artifacts.keys())) if artifacts else None
            chart_url = artifacts[active_metric]["png"] if active_metric else None
            top_df = results[active_metric].copy() if active_metric else pd.DataFrame(columns=["feature", "score"])
            top_k_saved = mf["corr"].get("top_k")
            if active_metric in {"pearson", "spearman"}:
                top_df = top_df.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
            else:
                top_df = top_df.sort_values(by="score", ascending=False)
            if top_k_saved and top_k_saved > 0:
                top_df = top_df.head(top_k_saved)
            ctx["available_metrics"] = list(artifacts.keys())
            ctx["active_metric"] = active_metric
            ctx["chart_url"] = chart_url
            ctx["metric_rows"] = top_df.values.tolist()
            return templates.TemplateResponse("correlation.html", ctx)
        except Exception as e:
            ctx["error"] = friendly_error(e, "correlation")
            attach_error_recovery(ctx, "correlation", ws_id=ws_id)
            return templates.TemplateResponse("correlation.html", ctx)

    @router.post("/correlation/view", response_class=HTMLResponse)
    async def correlation_view(
        request: Request,
        ws_id: str = Form(...),
        metric: str = Form(...),
    ) -> HTMLResponse:
        mf = load_manifest(ws_id)
        if not mf or "corr" not in mf or not mf["corr"].get("artifacts"):
            ctx = {"request": request, "error": "No correlation artifacts available. Please run correlation first."}
            attach_error_recovery(ctx, "correlation", ws_id=ws_id)
            return templates.TemplateResponse(
                "correlation.html",
                ctx,
            )

        cols = mf.get("columns", [])
        inputs = mf.get("inputs", [])
        target = mf.get("target")
        corr = mf["corr"]
        artifacts = corr.get("artifacts") or {}
        available = list(artifacts.keys())
        if not available:
            ctx = {"request": request, "error": "No metrics available."}
            attach_error_recovery(ctx, "correlation", ws_id=ws_id)
            return templates.TemplateResponse(
                "correlation.html",
                ctx,
            )
        if metric not in available:
            metric = available[0]

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
        _add_preview(ctx, mf.get("csv_path"))

        try:
            dfm = pd.read_csv(artifacts[metric]["csv"])
            if metric in {"pearson", "spearman"}:
                dfm = dfm.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
            else:
                dfm = dfm.sort_values(by="score", ascending=False)
            top_k = corr.get("top_k")
            if top_k and top_k > 0:
                dfm = dfm.head(top_k)
            ctx["metric_rows"] = dfm.values.tolist()
        except Exception:
            ctx["metric_rows"] = []

        corr["active_metric"] = metric
        save_manifest(ws_id, mf)
        return templates.TemplateResponse("correlation.html", ctx)

    @router.post("/correlation/upload", response_class=HTMLResponse)
    async def correlation_upload(request: Request, csvfile: UploadFile = File(...)) -> HTMLResponse:
        fname = (csvfile.filename or "").lower()
        if not fname.endswith(".csv"):
            ctx = {"request": request, "error": "Please upload a .csv file."}
            attach_error_recovery(ctx, "correlation")
            return templates.TemplateResponse(
                "correlation.html",
                ctx,
            )
        saved = await _save_csv(csvfile, uploads_dir)
        try:
            df = pd.read_csv(saved, nrows=PREVIEW_NROWS)
            headers = list(df.columns)
            rows = df.astype(object).where(pd.notnull(df), None).values.tolist()
        except Exception as e:
            ctx: dict[str, Any] = {"request": request, "error": friendly_error(e, "correlation")}
            attach_error_recovery(ctx, "correlation")
            return templates.TemplateResponse(
                "correlation.html",
                ctx,
            )
        ws_id = uuid4().hex
        manifest = {
            "csv_path": str(saved),
            "columns": headers,
            "inputs": [],
            "target": None,
            "selected": [],
        }
        save_manifest(ws_id, manifest)
        ctx: dict[str, Any] = {
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
        _add_non_ascii_notice(ctx, headers)
        return templates.TemplateResponse("correlation.html", ctx)

    @router.post("/correlation/select", response_class=HTMLResponse)
    async def correlation_select(
        request: Request,
        ws_id: str = Form(...),
        action: str = Form(...),
        cols: Optional[list[str]] = Form(None),
        inputs: Optional[list[str]] = Form(None),
        target_choice: Optional[str] = Form(None),
    ) -> HTMLResponse:
        mf = load_manifest(ws_id)
        if not mf:
            return templates.TemplateResponse(
                "correlation.html",
                {"request": request, "error": "Invalid session. Please re-upload CSV."},
            )

        columns: list[str] = list(mf.get("columns", []))
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
            for column in chosen_cols:
                if column in columns:
                    cur_inputs.add(column)
            if cur_target in cur_inputs:
                cur_inputs.discard(cur_target)
        elif action == "remove_inputs":
            for column in inputs_sel:
                cur_inputs.discard(column)
        elif action == "set_target":
            if target_choice and target_choice in columns:
                cur_target = target_choice
                cur_inputs = {column for column in columns if column != cur_target}
                cur_selected = set(columns)
            elif chosen_cols:
                cur_selected = set(chosen_cols)
        elif action.startswith("set_target_direct:"):
            direct_target = action.split(":", 1)[1]
            if direct_target in columns:
                cur_target = direct_target
                cur_inputs = {column for column in columns if column != cur_target}
                cur_selected = set(columns)
        elif action.startswith("toggle_input:"):
            column_name = action.split(":", 1)[1]
            if column_name in columns and column_name != cur_target:
                if column_name in cur_inputs:
                    cur_inputs.discard(column_name)
                else:
                    cur_inputs.add(column_name)
                cur_selected = set(columns)
        elif action == "auto_inputs":
            if cur_target and cur_target in columns:
                cur_inputs = {column for column in columns if column != cur_target}
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
        save_manifest(ws_id, mf)

        ctx: dict[str, Any] = {
            "request": request,
            "ws_id": ws_id,
            "csv_path": mf.get("csv_path"),
            "all_columns": columns,
            "inputs": ordered_inputs,
            "target": cur_target,
            "selected": ordered_selected,
        }
        _add_preview(ctx, mf.get("csv_path"), nrows=10)
        return templates.TemplateResponse("correlation.html", ctx)

    @router.get("/correlation/download_all")
    async def correlation_download_all(ws_id: str):
        mf = load_manifest(ws_id)
        if not mf or "corr" not in mf or not mf["corr"].get("artifacts"):
            return JSONResponse({"error": "No correlation artifacts available."}, status_code=404)

        corr = mf["corr"]
        artifacts = corr.get("artifacts") or {}
        if not artifacts:
            return JSONResponse({"error": "No metrics available."}, status_code=404)

        desired = corr.get("metrics") or list(artifacts.keys())
        metrics_order = [metric for metric in desired if metric in artifacts]
        if not metrics_order:
            metrics_order = list(artifacts.keys())

        df_all = None
        for metric_name in metrics_order:
            csv_path = artifacts[metric_name].get("csv")
            if not csv_path:
                continue
            try:
                dfm = pd.read_csv(csv_path)
            except Exception as e:
                return JSONResponse({"error": f"Failed to read {metric_name} CSV: {e}"}, status_code=500)
            if "feature" not in dfm.columns or "score" not in dfm.columns:
                return JSONResponse({"error": f"Unexpected columns in {metric_name} CSV."}, status_code=500)
            dfm = dfm[["feature", "score"]].rename(columns={"score": metric_name})
            if df_all is None:
                df_all = dfm
            else:
                df_all = df_all.merge(dfm, on="feature", how="outer")

        if df_all is None or df_all.empty:
            return JSONResponse({"error": "No data to combine."}, status_code=404)

        df_all = df_all.sort_values(by="feature", kind="stable")
        data_dir, _ = _corr_dirs(workspace_dir, static_dir, ws_id)
        combined_path = data_dir / "combined.csv"
        df_all.to_csv(combined_path, index=False)
        artifacts["combined"] = {"csv": str(combined_path)}
        mf["corr"]["artifacts"] = artifacts
        save_manifest(ws_id, mf)
        return FileResponse(str(combined_path), media_type="text/csv", filename="correlation_all.csv")

    return router
