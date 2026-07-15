"""Tests that GUI SHAP reuses the training-time feature preparation rules."""

from __future__ import annotations

import pandas as pd

from ascends.gui_shap_routes import _prepare_shap_frame


def test_shap_frame_matches_numeric_training_preparation() -> None:
    source = pd.DataFrame(
        {
            "feature": ["1.5", "invalid", "3.5"],
            "target": [10.0, 20.0, 30.0],
        }
    )

    prepared = _prepare_shap_frame(source, ["feature"], "target", "r")

    assert prepared.index.tolist() == [0, 2]
    assert prepared["feature"].tolist() == [1.5, 3.5]
    assert prepared["feature"].dtype == "float64"


def test_shap_frame_rejects_a_missing_target() -> None:
    source = pd.DataFrame({"feature": [1.0, 2.0]})

    try:
        _prepare_shap_frame(source, ["feature"], "target", "r")
    except ValueError as exc:
        assert "Target column missing" in str(exc)
    else:
        raise AssertionError("Missing SHAP target should be rejected")
