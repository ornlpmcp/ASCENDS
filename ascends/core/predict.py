"""Batch inference with manifest-based feature alignment."""

from typing import Any
import pandas as pd
from pathlib import Path


import joblib
from ascends.core.data import align_to_features


def _load_manifest(run_dir: str) -> dict[str, Any]:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        import json

        with open(manifest_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _casefold_columns_to_features(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    feature_by_lower: dict[str, list[str]] = {}
    for feature in features:
        feature_by_lower.setdefault(str(feature).lower(), []).append(feature)

    rename: dict[str, str] = {}
    existing = set(data.columns)
    for col in data.columns:
        matches = feature_by_lower.get(str(col).lower(), [])
        if len(matches) == 1 and col != matches[0] and matches[0] not in existing:
            rename[col] = matches[0]
    if not rename:
        return data
    return data.rename(columns=rename)


def _prepare_prediction_frame(data: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    features = (
        manifest.get("features")
        or manifest.get("inputs")
        or manifest.get("input_features")
        or manifest.get("X_features")
        or manifest.get("X_cols")
    )
    if not features:
        return data
    normalized = _casefold_columns_to_features(data, list(features))
    return align_to_features(normalized, list(features))


def batch_predict(model_path: str, data: Any, out_dir: str = ".", run_dir: str = ".") -> dict[str, str]:
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

    manifest = _load_manifest(run_dir)
    X_pred = _prepare_prediction_frame(data, manifest)

    # --- Generate predictions ---
    y_pred = est.predict(X_pred)

    # --- Load the manifest to get the target ---
    target = manifest.get("target", None)

    # --- Decide the output column name ---
    pred_col = f"{target}_pred" if target else "prediction"

    # --- Build the output DataFrame and save ---
    out_df = data.copy()
    out_df[pred_col] = y_pred
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "predictions.csv"
    out_df.to_csv(out_path, index=False)

    # --- Return the actual column name and output path ---
    return {"pred_col": pred_col, "out_path": str(out_path)}
