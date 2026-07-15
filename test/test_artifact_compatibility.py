"""Compatibility tests for saved-run manifests and prediction preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ascends.cli import app
from ascends.core.train import train_model


class EncodedColumnModel:
    """Estimator stand-in that verifies the encoded prediction schema."""

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        assert list(data.columns) == ["color_blue", "color_red", "x"]
        return np.arange(len(data), dtype=float)


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
    pd.DataFrame({"color": ["red", "blue"], "x": [1.0, 2.0]}).to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        ["predict", str(run_dir), "--csv", str(csv_path), "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    output = pd.read_csv(out_dir / "predictions.csv")
    assert output["target_pred"].tolist() == [0.0, 1.0]
