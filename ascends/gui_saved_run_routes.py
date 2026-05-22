"""Saved-run and report routes for the ASCENDS FastAPI GUI."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from joblib import dump

from ascends.core.interpret import interpret_run

logger = logging.getLogger("ascends.gui")


def _train_context(
    request: Request,
    ws_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request": request,
        "ws_id": ws_id,
        "csv_path": manifest.get("csv_path"),
        "all_columns": manifest.get("columns", []),
        "selected": manifest.get("selected", []),
        "inputs": manifest.get("inputs", []),
        "target": manifest.get("target"),
    }


def _render_report_html(
    *,
    templates: Jinja2Templates,
    static_dir: Path,
    ws_dir: Callable[[str], Path],
    run_name: str,
    rec: dict[str, Any],
    ws_id: str,
    out_dir: Optional[Path] = None,
) -> str:
    """Render report.html from a LAST_TRAIN record and return HTML string."""
    task = rec["params"].get("task", "r")
    task_label = "Classification" if task == "c" else "Regression"
    train_metrics = dict(rec.get("metrics_train") or {})
    test_metrics = dict(rec.get("metrics_test") or {})
    inputs = rec.get("inputs", [])
    target = rec.get("target", "")
    n_train = rec.get("n_train") or 0
    n_test = rec.get("n_test") or 0
    metric_keys = list(dict.fromkeys(list(train_metrics.keys()) + list(test_metrics.keys())))

    importance_rows = []
    importance_df = None
    shap_csv = ws_dir(ws_id) / "train" / "shap_importance.csv"
    if shap_csv.exists():
        try:
            importance_df = pd.read_csv(shap_csv)
            importance_rows = importance_df.head(15).values.tolist()
        except Exception:
            pass

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
        if any(
            word in low
            for word in (
                "overfitting",
                "imbalance",
                "leakage",
                "worse",
                "low",
                "poor",
                "large error",
                "small training",
                "heavily relies",
            )
        ):
            return "warn"
        if any(word in low for word in ("very good", "excellent", "consistent", "low error", "good")):
            return "good"
        return ""

    insights = [{"text": text, "level": _level(text)} for text in raw_insights]

    plot_files = []
    if out_dir is not None:
        for fname, label in [
            ("parity.png", "Parity Plot"),
            ("confusion.png", "Confusion Matrix"),
            ("shap_importance.png", "Feature Importance"),
        ]:
            if (out_dir / fname).exists():
                plot_files.append({"src": fname, "label": label})
    else:
        ws_train_dir = static_dir / "workspace" / ws_id / "train"
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


def _generate_report(
    *,
    templates: Jinja2Templates,
    static_dir: Path,
    ws_dir: Callable[[str], Path],
    run_name: str,
    out_dir: Path,
    rec: dict[str, Any],
    ws_id: str,
) -> None:
    rec_saved = dict(rec)
    html = _render_report_html(
        templates=templates,
        static_dir=static_dir,
        ws_dir=ws_dir,
        run_name=run_name,
        rec=rec_saved,
        ws_id=ws_id,
        out_dir=out_dir,
    )
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def create_saved_run_router(
    *,
    templates: Jinja2Templates,
    runs_dir: Path,
    static_dir: Path,
    ws_dir: Callable[[str], Path],
    load_manifest: Callable[[str], dict[str, Any]],
    list_saved_runs: Callable[[], list[dict[str, Any]]],
    unique_run_name: Callable[[str], str],
    last_train: dict[str, Any],
) -> APIRouter:
    """Create routes for saving, deleting, and reporting model runs."""
    router = APIRouter()

    @router.post("/train/save", response_class=HTMLResponse)
    async def train_save(
        request: Request,
        ws_id: str = Form(...),
        save_name: Optional[str] = Form(None),
    ) -> HTMLResponse:
        ctx = _train_context(request, ws_id, load_manifest(ws_id) or {})
        rec = last_train.get(ws_id)
        if not rec:
            ctx["train_error"] = "No trained model available to save. Please Train first."
            ctx["saved_runs"] = list_saved_runs()
            return templates.TemplateResponse("train.html", ctx)

        base = save_name or (rec["params"].get("model", "model") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_name = unique_run_name(base)
        out_dir = runs_dir / run_name

        try:
            dump(rec["estimator"], out_dir / "model.joblib")
        except Exception as e:
            ctx["train_error"] = f"Failed to save model: {e}"
            shutil.rmtree(out_dir, ignore_errors=True)
            ctx["saved_runs"] = list_saved_runs()
            return templates.TemplateResponse("train.html", ctx)

        try:
            train_metrics = dict(rec.get("metrics_train", {}))
            test_metrics = dict(rec.get("metrics_test", {}))
            metrics_df = pd.DataFrame(
                [
                    {"split": "Train", **train_metrics},
                    {"split": "Test", **test_metrics},
                ]
            )
            metrics_df.to_csv(out_dir / "metrics.csv", index=False)
        except Exception as e:
            ctx["train_error"] = f"Failed to write metrics.csv: {e}"
            shutil.rmtree(out_dir, ignore_errors=True)
            ctx["saved_runs"] = list_saved_runs()
            return templates.TemplateResponse("train.html", ctx)

        manifest = {
            "name": run_name,
            "created_at": rec["timestamp"],
            "task": rec["params"].get("task", "r"),
            "model": rec["params"].get("model"),
            "seed": rec["params"].get("seed"),
            "test_size": rec["params"].get("test_size"),
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
            ctx["saved_runs"] = list_saved_runs()
            return templates.TemplateResponse("train.html", ctx)

        try:
            ws_train_dir = static_dir / "workspace" / ws_id / "train"
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

        try:
            _generate_report(
                templates=templates,
                static_dir=static_dir,
                ws_dir=ws_dir,
                run_name=run_name,
                out_dir=out_dir,
                rec=rec,
                ws_id=ws_id,
            )
        except Exception as e:
            logger.warning("Report generation failed for %s: %s", run_name, e)

        ctx["save_ok"] = f"Saved run: {run_name}"
        ctx["saved_runs"] = list_saved_runs()
        return templates.TemplateResponse("train.html", ctx)

    @router.get("/train/report", response_class=HTMLResponse)
    async def train_report_preview(request: Request, ws_id: str = Query(...)) -> HTMLResponse:
        """Render a live report from LAST_TRAIN without requiring Save."""
        rec = last_train.get(ws_id)
        if not rec:
            return HTMLResponse(content="No trained model found. Please train a model first.", status_code=404)
        return HTMLResponse(
            content=_render_report_html(
                templates=templates,
                static_dir=static_dir,
                ws_dir=ws_dir,
                run_name="(unsaved)",
                rec=rec,
                ws_id=ws_id,
            )
        )

    @router.get("/runs/{run_name}/report.html", response_class=HTMLResponse)
    async def serve_report(run_name: str) -> HTMLResponse:
        """Serve the saved report.html for a run."""
        report_path = runs_dir / run_name / "report.html"
        if not report_path.exists():
            return HTMLResponse(content="Report not found.", status_code=404)
        return HTMLResponse(content=report_path.read_text(encoding="utf-8"))

    @router.post("/train/delete", response_class=HTMLResponse)
    async def train_delete(
        request: Request,
        run_name: str = Form(...),
        ws_id: Optional[str] = Form(None),
    ) -> HTMLResponse:
        ctx: dict[str, Any] = {"request": request}
        target_dir = runs_dir / run_name
        if target_dir.exists() and target_dir.is_dir():
            try:
                shutil.rmtree(target_dir)
                ctx["save_ok"] = f"Deleted run: {run_name}"
            except Exception as e:
                ctx["train_error"] = f"Failed to delete run {run_name}: {e}"
        else:
            ctx["train_error"] = f"Run not found: {run_name}"

        if ws_id:
            return RedirectResponse(url=f"/train?ws_id={quote(ws_id)}", status_code=303)

        ctx["saved_runs"] = list_saved_runs()
        return templates.TemplateResponse("train.html", ctx)

    return router
