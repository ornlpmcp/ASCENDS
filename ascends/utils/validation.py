"""Schema helpers, guardrails."""

from typing import Literal


def canonicalize_task(value: str) -> Literal["regression", "classification"]:
    """Convert task shorthand to canonical form.

    Args:
        value: The task shorthand or full name.

    Returns:
        The canonical task name.

    Raises:
        ValueError: If the task is not recognized.
    """
    value = value.lower()
    if value in {"r", "reg", "regression"}:
        return "regression"
    elif value in {"c", "cls", "class", "classification"}:
        return "classification"
    else:
        raise ValueError(f"Unrecognized task: {value}")
