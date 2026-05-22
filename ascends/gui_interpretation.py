"""Helpers for user-facing interpretation of GUI data and model results."""

from __future__ import annotations

from typing import Any

import pandas as pd

SMALL_DATASET_WARNING_ROWS = 100
SMALL_DATASET_WARNING = "This dataset has fewer than 100 rows. Treat model metrics as preliminary."
REGRESSION_R2_STRONG = 0.80
REGRESSION_R2_CAUTION = 0.50
CLASSIFICATION_STRONG = 0.85
CLASSIFICATION_CAUTION = 0.70
DOMAIN_VALIDATION_NOTE = "Domain validation required."
LOWER_IS_BETTER_NOTE = "Lower is better; compare against domain tolerance."


def _label_from_score(score: float, *, strong: float, caution: float) -> str:
    if score >= strong:
        return "Strong"
    if score >= caution:
        return "Caution"
    return "Weak"


def _metric_entry(label: str, note: str) -> dict[str, str]:
    return {"label": label, "note": note}


def interpret_regression_metrics(metrics: dict[str, float]) -> dict[str, Any]:
    """Return conservative user-facing labels for regression metrics."""
    r2 = metrics.get("R2")
    if r2 is None:
        overall = _metric_entry("N/A", "R2 is unavailable.")
    else:
        overall = _metric_entry(
            _label_from_score(float(r2), strong=REGRESSION_R2_STRONG, caution=REGRESSION_R2_CAUTION),
            DOMAIN_VALIDATION_NOTE,
        )
    metric_labels = {
        "R2": overall,
        "MAE": _metric_entry("N/A", LOWER_IS_BETTER_NOTE),
        "RMSE": _metric_entry("N/A", LOWER_IS_BETTER_NOTE),
    }
    return {"overall": overall, "metrics": metric_labels}


def interpret_classification_metrics(metrics: dict[str, float]) -> dict[str, Any]:
    """Return conservative user-facing labels for classification metrics."""
    accuracy = metrics.get("Accuracy")
    f1 = metrics.get("F1")
    if accuracy is None or f1 is None:
        overall = _metric_entry("N/A", "Accuracy and F1 are required for this label.")
    else:
        score = min(float(accuracy), float(f1))
        overall = _metric_entry(
            _label_from_score(score, strong=CLASSIFICATION_STRONG, caution=CLASSIFICATION_CAUTION),
            f"{DOMAIN_VALIDATION_NOTE} Accuracy alone can hide class imbalance.",
        )
    metric_labels = {
        "Accuracy": overall,
        "F1": overall,
        "Precision": _metric_entry("N/A", "Useful with recall; higher is usually better."),
        "Recall": _metric_entry("N/A", "Useful with precision; higher is usually better."),
        "ROC_AUC": _metric_entry("N/A", "Ranking metric for binary classification."),
    }
    return {"overall": overall, "metrics": metric_labels}


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
