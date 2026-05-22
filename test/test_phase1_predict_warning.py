"""Phase 1 prediction warning tests."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ascends.core.predict import batch_predict


class FeatureCheckingModel:
    """Minimal estimator that validates feature alignment during predict."""

    def __init__(self, expected_columns: list[str]) -> None:
        self.expected_columns = expected_columns

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        assert list(data.columns) == self.expected_columns
        return np.zeros(len(data))


def test_prediction_warns_when_unseen_categorical_dummy_is_ignored(
    tmp_path: Path,
    caplog,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "predict"
    run_dir.mkdir()
    features = ["color_blue", "color_red"]
    model_path = run_dir / "model.joblib"
    joblib.dump(FeatureCheckingModel(features), model_path)
    (run_dir / "manifest.json").write_text(
        json.dumps({"target": "y", "features": features}),
        encoding="utf-8",
    )
    data = pd.DataFrame({"color": ["green", "red"]})

    with caplog.at_level(logging.WARNING, logger="ascends.core.predict"):
        batch_predict(str(model_path), data, out_dir=str(out_dir), run_dir=str(run_dir))

    assert "Prediction data contains values not seen during training" in caplog.text
    assert "color_green" in caplog.text
