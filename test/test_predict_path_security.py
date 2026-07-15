"""Security regression tests for prediction run and download paths."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import UploadFile
from fastapi.responses import HTMLResponse

from ascends.gui_predict_routes import create_predict_router


class ConstantModel:
    """Minimal predictor used to exercise the prediction route."""

    def predict(self, data) -> np.ndarray:
        return np.full(len(data), 3.5)


class RecordingTemplates:
    """Return simple responses while retaining contexts from route handlers."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    def TemplateResponse(
        self, _name: str, context: dict[str, object], status_code: int = 200
    ) -> HTMLResponse:
        self.contexts.append(context)
        return HTMLResponse("rendered", status_code=status_code)


def _routes_for(runs_dir: Path) -> tuple[RecordingTemplates, object, object]:
    templates = RecordingTemplates()
    router = create_predict_router(
        templates=templates,  # type: ignore[arg-type]
        runs_dir=runs_dir,
        list_saved_runs=lambda: [],
        slugify_name=lambda value: value,
    )
    handlers = {route.path: route.endpoint for route in router.routes}
    return templates, handlers["/predict/run"], handlers["/predict/download"]


def _upload() -> UploadFile:
    return UploadFile(file=io.BytesIO(b"feature\n1.0\n"), filename="features.csv")


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"inputs": ["feature"], "target": "target"}),
        encoding="utf-8",
    )
    joblib.dump(ConstantModel(), run_dir / "model.joblib")


def test_predict_run_rejects_run_name_that_resolves_outside_runs_dir(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outside_run = tmp_path / "outside_run"
    _write_run(outside_run)

    templates, predict_run, _ = _routes_for(runs_dir)
    response = asyncio.run(
        predict_run(
            request=None,
            run_name="../outside_run",
            csvfile=_upload(),
        )
    )

    assert response.status_code == 400
    assert templates.contexts[-1]["predict_errors"] == ["Invalid run name."]
    assert not (outside_run / "predictions").exists()


def test_predict_download_rejects_run_name_that_resolves_outside_runs_dir(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outside_predictions = tmp_path / "outside_run" / "predictions"
    outside_predictions.mkdir(parents=True)
    (outside_predictions / "export.csv").write_text(
        "feature,target_pred\n1,3.5\n", encoding="utf-8"
    )

    _, _, predict_download = _routes_for(runs_dir)
    response = asyncio.run(predict_download(run="../outside_run", file="export.csv"))

    assert response.status_code == 404


def test_predict_run_and_download_allow_saved_run_within_runs_dir(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "saved_run"
    _write_run(run_dir)

    templates, predict_run, predict_download = _routes_for(runs_dir)
    prediction_response = asyncio.run(
        predict_run(
            request=None,
            run_name="saved_run",
            csvfile=_upload(),
        )
    )

    assert prediction_response.status_code == 200
    assert templates.contexts[-1]["predict_errors"] is None
    predictions = list((run_dir / "predictions").glob("features_*_pred.csv"))
    assert len(predictions) == 1

    download_response = asyncio.run(
        predict_download(run="saved_run", file=predictions[0].name)
    )

    assert download_response.status_code == 200
    assert Path(download_response.path) == predictions[0]
