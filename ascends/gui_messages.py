"""Shared GUI message formatting helpers."""

from __future__ import annotations

import re

import pandas as pd

MISSING_COLUMNS_PREFIX = "Selected columns not found in CSV (case-sensitive check)"


def format_missing_columns_message(missing_columns: list[str]) -> str:
    columns = ", ".join(missing_columns)
    return f"{MISSING_COLUMNS_PREFIX}: {columns}"


def friendly_error(exc: Exception, context: str) -> str:
    """Return a user-facing error message for common GUI failures."""
    if isinstance(exc, pd.errors.EmptyDataError):
        return "Uploaded CSV has no data. Please check the file and try again."

    text = str(exc)
    text_lower = text.lower()
    if isinstance(exc, ImportError) and "xgboost" in text_lower:
        return "XGBoost is not installed. Run `uv sync` to install."

    if isinstance(exc, ValueError) and "target" in text_lower:
        match = re.search(r"Target ['\"]?([^'\"]+)['\"]?", text)
        target = match.group(1) if match else "selected"
        return f"Target column '{target}' is not in the CSV. Check the column name."

    fallback_by_context = {
        "correlation": "Correlation failed",
        "train": "Training failed",
        "predict": "Prediction failed",
    }
    prefix = fallback_by_context.get(context, "Operation failed")
    return f"{prefix}: {text}"
