"""Shared GUI message formatting helpers."""

MISSING_COLUMNS_PREFIX = "Selected columns not found in CSV (case-sensitive check)"


def format_missing_columns_message(missing_columns: list[str]) -> str:
    columns = ", ".join(missing_columns)
    return f"{MISSING_COLUMNS_PREFIX}: {columns}"
