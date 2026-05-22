"""Phase 2 silent warning helper tests."""

import logging

import pandas as pd

from ascends.core.data import (
    NON_ASCII_COLUMN_MESSAGE,
    align_to_features,
    find_non_ascii_columns,
)
from ascends.gui_messages import (
    append_notice,
    constant_columns_message,
    rows_removed_message,
    stratify_disabled_message,
)


def test_append_notice_preserves_multiple_levels() -> None:
    ctx: dict[str, object] = {}

    append_notice(ctx, rows_removed_message(2), level="info")
    append_notice(ctx, constant_columns_message(["flat"]), level="warning")

    assert ctx["notices"] == [
        {"level": "info", "message": "Removed 2 rows containing missing values."},
        {"level": "warning", "message": "Excluded constant columns: flat."},
    ]
    assert ctx["notice"] == "Removed 2 rows containing missing values."


def test_silent_warning_message_text_is_user_facing() -> None:
    assert rows_removed_message(1) == "Removed 1 row containing missing values."
    assert rows_removed_message(3) == "Removed 3 rows containing missing values."
    assert constant_columns_message(["a", "b"]) == "Excluded constant columns: a, b."
    assert stratify_disabled_message() == (
        "Only one class found in the target column; stratified split was disabled."
    )


def test_find_non_ascii_columns_identifies_headers() -> None:
    df = pd.DataFrame({"alpha": [1], "온도": [2], "β": [3]})

    assert find_non_ascii_columns(df.columns) == ["온도", "β"]


def test_align_to_features_logs_non_ascii_columns(caplog) -> None:
    df = pd.DataFrame({"온도": [1, 2], "phase": ["a", "b"]})

    with caplog.at_level(logging.WARNING, logger="ascends.core.data"):
        aligned = align_to_features(df, ["온도", "phase_a", "phase_b"])

    assert list(aligned.columns) == ["온도", "phase_a", "phase_b"]
    assert NON_ASCII_COLUMN_MESSAGE in caplog.text
    assert "온도" in caplog.text
