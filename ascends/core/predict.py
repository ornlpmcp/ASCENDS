"""Batch inference, feature alignment."""

from typing import Any, List


def batch_predict(model: Any, data: Any) -> List[Any]:
    """Perform batch predictions with the model.

    Args:
        model: The model to use for predictions.
        data: The dataset for predictions.

    Returns:
        A list of predictions.
    """
    predictions = []
    for item in data:
        prediction = model.predict(item)
        predictions.append(prediction)
    return predictions
