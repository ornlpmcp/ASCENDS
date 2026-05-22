"""Shared GUI message formatting helpers."""

from __future__ import annotations

import re

import pandas as pd

MISSING_COLUMNS_PREFIX = "Selected columns not found in CSV (case-sensitive check)"


def format_missing_columns_message(missing_columns: list[str]) -> str:
    columns = ", ".join(missing_columns)
    return f"{MISSING_COLUMNS_PREFIX}: {columns}"


def append_notice(ctx: dict[str, object], message: str, *, level: str = "info") -> None:
    """Append a renderable notice and keep the legacy single notice key populated."""
    notices = list(ctx.get("notices") or [])
    notices.append({"level": level, "message": message})
    ctx["notices"] = notices
    if "notice" not in ctx:
        ctx["notice"] = message


def rows_removed_message(count: int) -> str:
    noun = "row" if count == 1 else "rows"
    return f"Removed {count} {noun} containing missing values."


def constant_columns_message(columns: list[str]) -> str:
    return f"Excluded constant columns (only one unique value): {', '.join(columns)}."


def stratify_disabled_message() -> str:
    return "Only one class found in target; stratified split disabled. Results may be unreliable."


def attach_error_recovery(ctx: dict[str, object], context: str, *, ws_id: str | None = None) -> None:
    """Add a simple recovery link for GUI error notices."""
    if context == "train":
        url = f"/correlation?ws_id={ws_id}" if ws_id else "/correlation"
        label = "Back to Correlation"
    elif context == "predict":
        url = "/predict"
        label = "Back to Predict"
    else:
        url = f"/correlation?ws_id={ws_id}" if ws_id else "/correlation"
        label = "Back to Correlation"
    ctx["error_recovery_url"] = url
    ctx["error_recovery_label"] = label


def friendly_error(exc: Exception, context: str) -> str:
    """Return a user-facing error message for common GUI failures."""
    if isinstance(exc, pd.errors.EmptyDataError):
        return "Uploaded CSV has no data. Please check the file and try again."

    text = str(exc)
    text_lower = text.lower()
    if isinstance(exc, ImportError) and "xgboost" in text_lower:
        return "XGBoost is not installed. Run `uv sync` to install."

    if isinstance(exc, ValueError) and ("top-k" in text_lower or "top_k" in text_lower):
        return "Top-K must be a positive integer."

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
