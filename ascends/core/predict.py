"""Batch inference, feature alignment."""

from typing import Any, List


import joblib

def batch_predict(model_path: str, data: Any) -> List[Any]:
    """Perform batch predictions with the model.

    Args:
        model: The model to use for predictions.
        data: The dataset for predictions.

    Returns:
        A list of predictions.
    """
    obj = joblib.load(model_path)
    predictions = []

    # Backward-compat: older runs saved a result dict with a "model" key
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
    else:
        est = obj

    # Sanity check
    if not hasattr(est, "predict"):
        raise TypeError(
            f"Loaded object from {model_path} is a {type(obj).__name__} "
            "and does not have a .predict(...) method. "
            "If this run was trained with an older ASCENDS, retrain or "
            "re-run train to produce a model-only artifact."
        )
    for item in data:
        prediction = est.predict(item)
        predictions.append(prediction)
    return predictions
