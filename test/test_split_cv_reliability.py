"""Regression coverage for infeasible training splits and CV folds."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from ascends.core.train import train_eval, train_model
from ascends.gui_train_run_routes import create_train_run_router


def test_train_eval_skips_cv_for_too_few_regression_samples() -> None:
    train_df = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]}
    )
    test_df = pd.DataFrame({"feature": [5.0], "target": [10.0]})

    result = train_eval(
        train_df=train_df,
        test_df=test_df,
        target="target",
        task="r",
        model_kind="ridge",
        random_state=42,
    )

    assert result["model"] is not None
    assert result["cv_scores"] == {}


def test_train_eval_skips_cv_for_singleton_class() -> None:
    train_df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": ["common", "common", "common", "rare"],
        }
    )
    test_df = pd.DataFrame({"feature": [5.0, 6.0], "target": ["common", "rare"]})

    result = train_eval(
        train_df=train_df,
        test_df=test_df,
        target="target",
        task="c",
        model_kind="rf",
        random_state=42,
    )

    assert result["model"] is not None
    assert result["cv_scores"] == {}


def test_train_eval_keeps_normal_classification_cv_output() -> None:
    train_df = pd.DataFrame(
        {
            "feature": np.arange(12, dtype=float),
            "target": ["a"] * 6 + ["b"] * 6,
        }
    )
    test_df = pd.DataFrame({"feature": [12.0, 13.0], "target": ["a", "b"]})

    result = train_eval(
        train_df=train_df,
        test_df=test_df,
        target="target",
        task="classification",
        model_kind="rf",
        random_state=42,
    )

    assert set(result["cv_scores"]) == {
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
    }


def test_train_model_records_when_rare_class_disables_stratification(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "rare_class.csv"
    frame = pd.DataFrame(
        {"feature": np.arange(20, dtype=float), "target": ["common"] * 19 + ["rare"]}
    )
    frame.to_csv(csv_path, index=False)
    run_dir = tmp_path / "run"

    train_model(
        csv_path=str(csv_path),
        target="target",
        task="classification",
        model="rf",
        test_size=0.2,
        out_dir=str(run_dir),
        random_state=42,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"]["stratify_col"] is None


def test_train_model_disables_stratification_when_test_set_cannot_hold_all_classes(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "many_classes.csv"
    pd.DataFrame(
        {"feature": np.arange(6, dtype=float), "target": ["a", "a", "b", "b", "c", "c"]}
    ).to_csv(csv_path, index=False)
    run_dir = tmp_path / "run"

    train_model(
        csv_path=str(csv_path),
        target="target",
        task="classification",
        model="rf",
        test_size=0.2,
        out_dir=str(run_dir),
        random_state=42,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"]["stratify_col"] is None


class CapturingTemplates:
    """Capture route context without relying on a Starlette template version."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    def TemplateResponse(self, _name: str, context: dict[str, object]) -> HTMLResponse:
        self.contexts.append(context)
        return HTMLResponse("")


def _train_route(csv_path: Path, static_dir: Path):
    app = FastAPI()
    templates = CapturingTemplates()
    router = create_train_run_router(
        templates=templates,
        static_dir=static_dir,
        load_manifest=lambda _ws_id: {
            "csv_path": str(csv_path),
            "columns": ["feature", "target"],
            "inputs": ["feature"],
            "target": "target",
        },
        list_saved_runs=lambda: [],
        last_train={},
    )
    app.include_router(router)
    route = next(route.endpoint for route in app.routes if route.path == "/train/run")
    return route, templates


def test_gui_train_rare_classes_disable_stratification_with_warning(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "rare_class.csv"
    pd.DataFrame(
        {"feature": np.arange(7, dtype=float), "target": ["common"] * 6 + ["rare"]}
    ).to_csv(csv_path, index=False)

    request = Request(
        {"type": "http", "method": "POST", "path": "/train/run", "headers": []}
    )
    route, templates = _train_route(csv_path, tmp_path / "static")
    response = asyncio.run(
        route(
            request=request,
            ws_id="workspace",
            task="c",
            model="rf",
            test_size=0.2,
            seed="42",
        )
    )

    assert response.status_code == 200
    context = templates.contexts[-1]
    assert context["notices"] == [
        {
            "level": "warning",
            "message": "This dataset has fewer than 100 rows. Treat model metrics as preliminary.",
        },
        {
            "level": "warning",
            "message": "Class counts are too small for a stratified split; using a random split instead.",
        },
        {
            "level": "warning",
            "message": "CV unavailable: dataset has fewer than 300 rows.",
        },
    ]
    assert context["metrics_train"]
    assert context["metrics_test"]
