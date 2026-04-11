"""Rule-based model interpretation for ASCENDS reports.

Produces human-readable insight strings from metrics and run metadata.
Designed to be replaced or augmented by an LLM layer later if needed.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _interpret_r2(r2: float) -> str:
    if r2 < 0:
        return "R² is negative — the model performs worse than simply predicting the mean. Review your features and data quality."
    if r2 < 0.5:
        return f"R²={r2:.3f}: Low explanatory power. Important features may be missing or the relationship may be non-linear."
    if r2 < 0.7:
        return f"R²={r2:.3f}: Moderate fit. Acceptable for noisy real-world data, but there is room to improve."
    if r2 < 0.9:
        return f"R²={r2:.3f}: Good fit."
    if r2 < 0.99:
        return f"R²={r2:.3f}: Very good fit."
    return f"R²={r2:.3f}: Near-perfect fit. Verify there is no data leakage (e.g. a feature that encodes the target)."


def _interpret_mae_relative(mae: float, target_range: float) -> Optional[str]:
    if target_range <= 0:
        return None
    ratio = mae / target_range
    if ratio > 0.2:
        return f"MAE is {ratio:.0%} of the target range — relatively large errors."
    if ratio > 0.1:
        return f"MAE is {ratio:.0%} of the target range — moderate errors."
    return f"MAE is {ratio:.0%} of the target range — low errors."


def _interpret_overfit_regression(r2_train: float, r2_test: float) -> str:
    gap = r2_train - r2_test
    if gap > 0.15:
        return f"Overfitting detected: train R²={r2_train:.3f} vs test R²={r2_test:.3f} (gap={gap:.3f}). Consider more training data, regularization, or a simpler model."
    if gap > 0.05:
        return f"Mild overfitting: train R²={r2_train:.3f} vs test R²={r2_test:.3f} (gap={gap:.3f}). Within acceptable range."
    return f"Train/test performance is consistent (gap={gap:.3f}). No significant overfitting."


def interpret_regression(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    n_train: int,
    n_test: int,
    target_values: Optional[List[float]] = None,
) -> List[str]:
    """Return a list of interpretation strings for a regression run."""
    insights: List[str] = []

    r2_test = test_metrics.get("r2")
    r2_train = train_metrics.get("r2")
    mae_test = test_metrics.get("mae")

    if r2_test is not None:
        insights.append(_interpret_r2(r2_test))

    if r2_train is not None and r2_test is not None:
        insights.append(_interpret_overfit_regression(r2_train, r2_test))

    if mae_test is not None and target_values is not None:
        t_range = max(target_values) - min(target_values)
        msg = _interpret_mae_relative(mae_test, t_range)
        if msg:
            insights.append(msg)

    insights.extend(_interpret_sample_size(n_train))

    return insights


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _interpret_accuracy(acc: float) -> str:
    if acc < 0.6:
        return f"Accuracy={acc:.1%}: Low. Check class balance and whether the right features are selected."
    if acc < 0.8:
        return f"Accuracy={acc:.1%}: Moderate."
    if acc < 0.9:
        return f"Accuracy={acc:.1%}: Good."
    return f"Accuracy={acc:.1%}: Very good."


def _interpret_f1(f1: float) -> str:
    if f1 < 0.6:
        return f"F1 (weighted)={f1:.3f}: Poor balance between precision and recall."
    if f1 < 0.8:
        return f"F1 (weighted)={f1:.3f}: Moderate."
    return f"F1 (weighted)={f1:.3f}: Good."


def _interpret_roc_auc(auc: float) -> str:
    if auc < 0.7:
        return f"ROC-AUC={auc:.3f}: Poor class discrimination."
    if auc < 0.9:
        return f"ROC-AUC={auc:.3f}: Good discrimination between classes."
    return f"ROC-AUC={auc:.3f}: Excellent discrimination."


def _interpret_overfit_classification(acc_train: float, acc_test: float) -> str:
    gap = acc_train - acc_test
    if gap > 0.15:
        return f"Overfitting detected: train accuracy={acc_train:.1%} vs test accuracy={acc_test:.1%} (gap={gap:.1%}). Consider more data or a simpler model."
    if gap > 0.05:
        return f"Mild overfitting: train accuracy={acc_train:.1%} vs test accuracy={acc_test:.1%} (gap={gap:.1%}). Within acceptable range."
    return f"Train/test accuracy is consistent (gap={gap:.1%}). No significant overfitting."


def interpret_classification(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    n_train: int,
    n_test: int,
    class_counts: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Return a list of interpretation strings for a classification run."""
    insights: List[str] = []

    acc_test = test_metrics.get("accuracy")
    acc_train = train_metrics.get("accuracy")
    f1_test = test_metrics.get("f1")
    auc = test_metrics.get("roc_auc")

    if acc_test is not None:
        insights.append(_interpret_accuracy(acc_test))

    if f1_test is not None:
        insights.append(_interpret_f1(f1_test))

    if auc is not None:
        insights.append(_interpret_roc_auc(auc))

    if acc_train is not None and acc_test is not None:
        insights.append(_interpret_overfit_classification(acc_train, acc_test))

    if class_counts:
        total = sum(class_counts.values())
        min_ratio = min(class_counts.values()) / total if total > 0 else 1.0
        if min_ratio < 0.2:
            insights.append(
                f"Class imbalance detected: the minority class represents only {min_ratio:.0%} of the data. "
                "F1 is a more reliable metric than accuracy in this case."
            )

    insights.extend(_interpret_sample_size(n_train))

    return insights


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

def _interpret_sample_size(n_train: int) -> List[str]:
    if n_train < 100:
        return [f"Very small training set ({n_train} samples). Results may not generalize well."]
    if n_train < 500:
        return [f"Small training set ({n_train} samples). Interpret results cautiously."]
    return []


def interpret_feature_importance(importance_df: Any) -> Optional[str]:
    """Warn if a single feature dominates importance."""
    try:
        total = importance_df["score"].abs().sum()
        if total <= 0:
            return None
        top_ratio = importance_df["score"].abs().iloc[0] / total
        top_name = importance_df["feature"].iloc[0]
        if top_ratio > 0.8:
            return (
                f'The model relies heavily on "{top_name}" ({top_ratio:.0%} of total importance). '
                "Verify this is expected and not a data leakage issue."
            )
    except Exception:
        pass
    return None


def interpret_run(
    task: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    n_train: int,
    n_test: int,
    target_values: Optional[List[float]] = None,
    class_counts: Optional[Dict[str, int]] = None,
    importance_df: Any = None,
) -> List[str]:
    """Top-level entry point — returns all insights for a run."""
    task = task.lower()
    if task in ("regression", "r"):
        insights = interpret_regression(
            train_metrics, test_metrics, n_train, n_test, target_values
        )
    elif task in ("classification", "c"):
        insights = interpret_classification(
            train_metrics, test_metrics, n_train, n_test, class_counts
        )
    else:
        insights = []

    if importance_df is not None:
        msg = interpret_feature_importance(importance_df)
        if msg:
            insights.append(msg)

    return insights
