"""Phase 2 friendly GUI error message tests."""

import pandas as pd

from ascends.gui_messages import friendly_error


def test_friendly_error_maps_empty_csv() -> None:
    message = friendly_error(pd.errors.EmptyDataError("No columns to parse"), "correlation")

    assert message == "Uploaded CSV has no data. Please check the file and try again."


def test_friendly_error_maps_target_value_error() -> None:
    message = friendly_error(ValueError("Target 'species' not in columns."), "train")

    assert message == "Target column 'species' is not in the CSV. Check the column name."


def test_friendly_error_maps_xgboost_import_error() -> None:
    message = friendly_error(ImportError("xgboost is required for --model xgb"), "train")

    assert message == "XGBoost is not installed. Run `uv sync` to install."


def test_friendly_error_uses_contextual_fallback() -> None:
    message = friendly_error(RuntimeError("boom"), "predict")

    assert message == "Prediction failed: boom"
