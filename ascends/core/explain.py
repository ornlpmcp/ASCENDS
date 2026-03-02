"""SHAP for trees + permutation fallback."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.inspection import permutation_importance

from ascends.core.models import is_tree_model
from ascends.utils.validation import canonicalize_task


def _mean_abs_shap_per_feature(shap_values: Any, n_features: int) -> np.ndarray:
    """Normalize different SHAP return shapes into per-feature importances."""
    if isinstance(shap_values, list):
        # Common in multiclass: one matrix per class
        mats = [np.asarray(v) for v in shap_values]
        vals = np.stack([np.mean(np.abs(m), axis=0) for m in mats], axis=0)
        return np.mean(vals, axis=0)

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        # (n_samples, n_features)
        return np.mean(np.abs(arr), axis=0)
    if arr.ndim == 3:
        # either (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if arr.shape[1] == n_features:
            return np.mean(np.abs(arr), axis=(0, 2))
        if arr.shape[2] == n_features:
            return np.mean(np.abs(arr), axis=(0, 1))

    raise ValueError(f"Unsupported SHAP value shape: {arr.shape}")


def explain_model(
    model: Any,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    task: str = "regression",
    max_samples: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Generate feature importances using SHAP (tree models) or permutation fallback."""
    task = canonicalize_task(task)
    if X is None or len(X) == 0:
        raise ValueError("X is empty; cannot explain model.")

    n = min(len(X), int(max_samples))
    Xs = X.sample(n=n, random_state=random_state) if len(X) > n else X.copy()
    ys = None
    if y is not None:
        ys = y.loc[Xs.index] if hasattr(y, "loc") else pd.Series(y).iloc[Xs.index]

    warning = None
    method = "permutation"

    if is_tree_model(model):
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xs)
            mean_abs = _mean_abs_shap_per_feature(shap_vals, Xs.shape[1])
            imp_df = pd.DataFrame(
                {"feature": list(Xs.columns), "importance": mean_abs.astype(float)}
            ).sort_values("importance", ascending=False, ignore_index=True)
            method = "shap"
            return {
                "method": method,
                "importance_df": imp_df,
                "n_samples": int(len(Xs)),
                "warning": warning,
            }
        except Exception as e:
            warning = f"SHAP failed; fell back to permutation importance ({e})"

    if ys is None:
        raise ValueError("Permutation fallback requires target values (y).")

    scoring = "r2" if task == "regression" else "f1_weighted"
    try:
        perm = permutation_importance(
            model,
            Xs,
            ys,
            n_repeats=10,
            random_state=random_state,
            scoring=scoring,
        )
    except Exception:
        # fallback to estimator default score if custom scoring fails
        perm = permutation_importance(
            model, Xs, ys, n_repeats=10, random_state=random_state
        )
    imp_df = pd.DataFrame(
        {"feature": list(Xs.columns), "importance": perm.importances_mean.astype(float)}
    ).sort_values("importance", ascending=False, ignore_index=True)

    return {
        "method": method,
        "importance_df": imp_df,
        "n_samples": int(len(Xs)),
        "warning": warning,
    }


def save_importance_plot(
    importance_df: pd.DataFrame,
    out_png: str | Path,
    method: str = "shap",
    top_n: int = 20,
) -> Path:
    """Save a styled horizontal importance plot."""
    if importance_df is None or importance_df.empty:
        raise ValueError("importance_df is empty; cannot render importance plot.")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_df = importance_df.head(max(1, int(top_n))).iloc[::-1].copy()
    vals = top_df["importance"].astype(float).to_numpy()
    max_abs = float(np.max(np.abs(vals))) if len(vals) else 1.0
    scale = np.abs(vals) / max(max_abs, 1e-12)
    colors = cm.get_cmap("viridis")(0.25 + 0.7 * scale)

    fig_h = max(5.2, 0.34 * len(top_df) + 2.0)
    fig, ax = plt.subplots(figsize=(14.8, fig_h), dpi=240)
    bars = ax.barh(top_df["feature"], vals, color=colors, edgecolor="#1f2937", linewidth=0.2)
    ax.set_title(f"Feature Importance ({method})", fontsize=16, pad=12)
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_ylabel("Feature", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.28, linewidth=0.7)
    ax.set_axisbelow(True)

    x_pad = 0.02 * max(max_abs, 1e-12)
    for bar, v in zip(bars, vals):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        x_txt = x + x_pad if x >= 0 else x - x_pad
        ha = "left" if x >= 0 else "right"
        ax.text(x_txt, y, f"{v:.4g}", va="center", ha=ha, fontsize=11, color="#1f2937")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
