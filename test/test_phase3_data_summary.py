"""Phase 3 dataset summary helper tests."""

import pandas as pd

from ascends.gui_interpretation import (
    SMALL_DATASET_WARNING,
    SMALL_DATASET_WARNING_ROWS,
    summarize_dataframe,
    small_dataset_warning,
)


def test_summarize_dataframe_counts_types_and_missing_values() -> None:
    df = pd.DataFrame(
        {
            "temperature": [1.0, 2.0, None, 4.0],
            "phase": ["alpha", "beta", "beta", None],
            "all_missing": [None, None, None, None],
        }
    )

    summary = summarize_dataframe(df)

    assert summary["total_rows"] == 4
    assert summary["total_columns"] == 3
    assert summary["numeric_columns"] == 2
    assert summary["categorical_columns"] == 1
    assert summary["missing_columns"] == 3
    assert summary["top_missing_columns"][0] == {
        "column": "all_missing",
        "missing": 4,
        "percent": 100.0,
    }


def test_small_dataset_warning_threshold() -> None:
    assert SMALL_DATASET_WARNING_ROWS == 100
    assert small_dataset_warning(99) == SMALL_DATASET_WARNING
    assert small_dataset_warning(100) is None
