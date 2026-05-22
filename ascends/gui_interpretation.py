"""Helpers for user-facing interpretation of GUI data and model results."""

from __future__ import annotations

from typing import Any

import pandas as pd

SMALL_DATASET_WARNING_ROWS = 100
SMALL_DATASET_WARNING = "This dataset has fewer than 100 rows. Treat model metrics as preliminary."


def summarize_dataframe(df: pd.DataFrame, *, top_n: int = 3) -> dict[str, Any]:
    """Return compact CSV summary data for GUI display."""
    total_rows = int(len(df))
    total_columns = int(len(df.columns))
    numeric_columns = int(len(df.select_dtypes(include="number").columns))
    categorical_columns = total_columns - numeric_columns
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    top_missing_columns = [
        {
            "column": str(column),
            "missing": int(count),
            "percent": round((int(count) / total_rows * 100) if total_rows else 0.0, 1),
        }
        for column, count in missing_counts.head(top_n).items()
    ]
    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_columns": int(len(missing_counts)),
        "top_missing_columns": top_missing_columns,
    }


def small_dataset_warning(row_count: int) -> str | None:
    if row_count < SMALL_DATASET_WARNING_ROWS:
        return SMALL_DATASET_WARNING
    return None
