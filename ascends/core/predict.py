"""Batch inference, feature alignment."""

from typing import Any, List
import pandas as pd
from pathlib import Path


import joblib
import os

def batch_predict(model_path: str, data: Any, out_dir: str = ".", run_dir: str = ".") -> List[Any]:
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

    # --- Generate predictions ---
    y_pred = est.predict(data)

    # --- Save predictions with descriptive column name ---
    pred_out = os.path.join(out_dir, "predictions.csv")
    # Try to get target name from manifest if available
    manifest_path = Path(run_dir) / "manifest.json"
    target = "prediction"
    if manifest_path.exists():
        try:
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            if "target" in manifest:
                target = manifest["target"]
        except Exception:
            pass

    pred_col = f"{target}_pred"
    pred_df = data.copy()
    pred_df[pred_col] = y_pred
    pred_df.to_csv(pred_out, index=False)
    print(f"Predictions saved to {pred_out} ({pred_col})")
