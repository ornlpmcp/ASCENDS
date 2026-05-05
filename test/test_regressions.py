"""Regression tests for CLI/core training and prediction behavior."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ascends.core.predict import batch_predict
from ascends.core.train import train_eval


class ColumnCheckingModel:
    """Minimal fitted-estimator stand-in that validates prediction columns."""

    def __init__(self, expected_columns: list[str]) -> None:
        self.expected_columns = expected_columns

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        assert list(data.columns) == self.expected_columns
        return np.arange(len(data))


def test_batch_predict_uses_manifest_features_and_preserves_input_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "predict"
    run_dir.mkdir()

    model_path = run_dir / "model.joblib"
    joblib.dump(ColumnCheckingModel(["a", "b"]), model_path)
    (run_dir / "manifest.json").write_text(
        json.dumps({"target": "y", "features": ["a", "b"]}),
        encoding="utf-8",
    )

    data = pd.DataFrame(
        {
            "b": [2.0, 4.0],
            "unused": [100.0, 200.0],
            "a": [1.0, 3.0],
        }
    )

    result = batch_predict(str(model_path), data, out_dir=str(out_dir), run_dir=str(run_dir))

    assert result["pred_col"] == "y_pred"
    out_df = pd.read_csv(result["out_path"])
    assert list(out_df.columns) == ["b", "unused", "a", "y_pred"]
    assert out_df["y_pred"].tolist() == [0, 1]


def test_train_eval_regression_reports_positive_mae() -> None:
    train_df = pd.DataFrame({"x": np.arange(10, dtype=float), "target": np.arange(10, dtype=float) * 2})
    test_df = pd.DataFrame({"x": np.arange(10, 13, dtype=float), "target": np.arange(10, 13, dtype=float) * 2})

    result = train_eval(
        train_df=train_df,
        test_df=test_df,
        target="target",
        task="r",
        model_kind="ridge",
        random_state=42,
    )

    assert result["test_metrics"]["mae"] > 0
