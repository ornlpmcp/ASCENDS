"""Batch inference, feature alignment."""

from typing import Any, List


import joblib
import os

def batch_predict(model_path: str, data: Any, out_dir: str = ".") -> List[Any]:
    """Perform batch predictions with the model.

    Args:
        model: The model to use for predictions.
        data: The dataset for predictions.

    Returns:
        A list of predictions.
    """
    # --- Load model with backward compatibility ---
    obj = joblib.load(model_path)
    # Older ASCENDS runs saved a dict with "model"
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
    else:
        est = obj

    # Sanity check: must be a fitted estimator
    if not hasattr(est, "predict"):
        raise TypeError(
            f"Loaded object from {model_path} is a {type(obj).__name__} "
            "and does not have a .predict(...) method. "
            "Re-train or convert the artifact so it contains a fitted estimator."
        )

    predictions = []

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    for item in data:
        prediction = est.predict(item)
        predictions.append(prediction)
    return predictions
