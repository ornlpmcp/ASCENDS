"""Compatibility tests for saved-run manifests and prediction preprocessing."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile
from fastapi.responses import HTMLResponse
from typer.testing import CliRunner

from ascends.cli import app
from ascends.core.train import train_eval, train_model
from ascends.gui_predict_routes import create_predict_router


class EncodedColumnModel:
    """Estimator stand-in that verifies the encoded prediction schema."""

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        assert list(data.columns) == ["color_blue", "color_red", "x"]
        return np.arange(len(data), dtype=float)


class RecordingTemplates:
    """Capture GUI prediction contexts without rendering Jinja templates."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    def TemplateResponse(
        self, _name: str, context: dict[str, object], status_code: int = 200
    ) -> HTMLResponse:
        self.contexts.append(context)
        return HTMLResponse("rendered", status_code=status_code)


def test_train_manifest_records_raw_inputs_and_encoded_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "categorical.csv"
    run_dir = tmp_path / "run"
    pd.DataFrame(
        {
            "color": ["red", "blue"] * 10,
            "x": np.arange(20, dtype=float),
            "target": np.arange(20, dtype=float) * 2,
        }
    ).to_csv(csv_path, index=False)

    train_model(
        str(csv_path),
        "target",
        task="r",
        model="ridge",
        test_size=0.2,
        out_dir=str(run_dir),
        random_state=42,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"] == ["color", "x"]
    assert set(manifest["features"]) == {"color_blue", "color_red", "x"}


def test_cli_predict_accepts_raw_categorical_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "predictions"
    csv_path = tmp_path / "input.csv"
    run_dir.mkdir()
    joblib.dump(EncodedColumnModel(), run_dir / "model.joblib")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "target": "target",
                "inputs": ["color", "x"],
                "features": ["color_blue", "color_red", "x"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"color": ["red", "blue"], "x": [1.0, 2.0]}).to_csv(
        csv_path, index=False
    )

    result = CliRunner().invoke(
        app,
        ["predict", str(run_dir), "--csv", str(csv_path), "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    output = pd.read_csv(out_dir / "predictions.csv")
    assert output["target_pred"].tolist() == [0.0, 1.0]


def test_gui_predict_accepts_cli_categorical_manifest(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "cli_run"
    run_dir.mkdir(parents=True)
    joblib.dump(EncodedColumnModel(), run_dir / "model.joblib")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "target": "target",
                "inputs": ["color", "x"],
                "features": ["color_blue", "color_red", "x"],
            }
        ),
        encoding="utf-8",
    )
    templates = RecordingTemplates()
    router = create_predict_router(
        templates=templates,  # type: ignore[arg-type]
        runs_dir=runs_dir,
        list_saved_runs=lambda: [],
        slugify_name=lambda value: value,
    )
    predict_run = next(
        route.endpoint for route in router.routes if route.path == "/predict/run"
    )
    upload = UploadFile(
        file=io.BytesIO(b"color,x\nred,1.0\nblue,2.0\n"),
        filename="input.csv",
    )

    response = asyncio.run(
        predict_run(request=None, run_name="cli_run", csvfile=upload)
    )

    assert response.status_code == 200
    assert templates.contexts[-1]["predict_errors"] is None
    output_files = list((run_dir / "predictions").glob("input_*_pred.csv"))
    assert len(output_files) == 1
    output = pd.read_csv(output_files[0])
    assert output["target_pred"].tolist() == [0.0, 1.0]


def test_training_feature_schema_does_not_learn_test_only_categories() -> None:
    train_df = pd.DataFrame(
        {
            "color": ["red", "blue"] * 5,
            "target": np.arange(10, dtype=float),
        }
    )
    test_df = pd.DataFrame({"color": ["green", "red"], "target": [10.0, 11.0]})

    result = train_eval(
        train_df=train_df,
        test_df=test_df,
        target="target",
        task="regression",
        model_kind="ridge",
        random_state=42,
    )

    assert set(result["features"]) == {"color_blue", "color_red"}
    assert "color_green" not in result["features"]
