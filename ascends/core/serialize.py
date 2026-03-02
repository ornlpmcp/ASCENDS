"""Joblib save/load, manifest.json."""

from typing import Any


def save_model(model: Any, filepath: str) -> None:
    """Save the model to a file.

    Args:
        model: The model to save.
        filepath: The path to save the model file.
    """
    import joblib

    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """Load a model from a file.

    Args:
        filepath: The path to the model file.

    Returns:
        The loaded model.
    """
    import joblib

    return joblib.load(filepath)
