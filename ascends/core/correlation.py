"""Correlation analysis (Pearson, Spearman, MI, dCor)."""

from ascends.utils.validation import canonicalize_task
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import dcor


def _safe_neighbors(n: int, default: int = 3) -> int:
    """Determine a safe number of neighbors for mutual information calculation."""
    return min(max(1, default), max(1, n - 1))


def _encode_target(target: pd.Series, task: str) -> pd.Series:
    """Return a numeric target series suitable for association metrics."""
    if task != "classification":
        return pd.to_numeric(target, errors="coerce")

    valid_target = target.notna()
    non_missing = target.loc[valid_target]
    encoded = pd.Series(np.nan, index=target.index, dtype=float)
    try:
        codes, _ = pd.factorize(non_missing, sort=True)
    except TypeError:
        codes, _ = pd.factorize(non_missing.astype(str), sort=True)
    encoded.loc[valid_target] = codes
    return encoded


def _complete_cases(
    feature: pd.Series, target: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite feature and target values."""
    pairs = pd.concat([feature, target], axis=1).apply(pd.to_numeric, errors="coerce")
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    return (
        pairs.iloc[:, 0].to_numpy(dtype=np.float64),
        pairs.iloc[:, 1].to_numpy(dtype=np.float64),
    )


def _require_samples(metric: str, feature: str, n_samples: int) -> None:
    """Validate the minimum aligned sample count required by a metric."""
    required = 3 if metric == "mi" else 2
    if n_samples < required:
        raise ValueError(
            f"Metric '{metric}' for feature '{feature}' requires at least "
            f"{required} aligned complete cases; found {n_samples}."
        )


def _finite_score(metric: str, feature: str, score: float) -> float:
    """Return a JSON-safe score or explain why the metric is undefined."""
    if not np.isfinite(score):
        raise ValueError(
            f"Metric '{metric}' for feature '{feature}' is undefined for the "
            "aligned complete cases."
        )
    return float(score)


def run_correlation(
    df: pd.DataFrame,
    target: str,
    task: str,
    metrics: List[str] = ["pearson", "spearman", "mi", "dcor"],
    topk: int = None,
    mi_neighbors: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Run correlation analysis on the dataset and return scores per metric.

    Args:
        df: The dataset to analyze.
        target: The target column name.
        task: The type of task ('regression' or 'classification').
        metrics: List of metrics to calculate.
        topk: Number of top features to return per metric.

    Returns:
        A dictionary mapping each metric to a dictionary of features and their scores.

    Raises:
        ValueError: If the task is not 'regression' or 'classification'.
    """
    task = canonicalize_task(task)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    supported_metrics = {"pearson", "spearman", "mi", "dcor"}
    unknown_metrics = set(metrics) - supported_metrics
    if unknown_metrics:
        raise ValueError(f"Unsupported correlation metrics: {sorted(unknown_metrics)}")

    y = _encode_target(df[target], task)

    # Initialize results dictionary for each metric
    results = {metric: {} for metric in metrics}

    # Select only numeric columns
    X = df.drop(columns=[target]).select_dtypes(include=[float, int])

    for metric in metrics:
        for feature in X.columns:
            x_values, y_values = _complete_cases(X[feature], y)
            _require_samples(metric, feature, len(x_values))

            if metric == "pearson":
                corr, _ = pearsonr(x_values, y_values)
            elif metric == "spearman":
                corr, _ = spearmanr(x_values, y_values)
            elif metric == "mi":
                k = _safe_neighbors(len(x_values), mi_neighbors)
                if task == "regression":
                    corr = mutual_info_regression(
                        x_values.reshape(-1, 1), y_values, n_neighbors=k
                    )[0]
                else:
                    corr = mutual_info_classif(
                        x_values.reshape(-1, 1), y_values, n_neighbors=k
                    )[0]
            elif metric == "dcor":
                corr = dcor.distance_correlation(x_values, y_values)
            results[metric][feature] = _finite_score(metric, feature, corr)

    # Convert np.float64 to float for JSON serialization
    results = {
        metric: {feature: float(score) for feature, score in scores.items()}
        for metric, scores in results.items()
    }
    for metric, scores in results.items():
        sorted_features = sorted(
            scores.items(), key=lambda item: abs(item[1]), reverse=True
        )
        if topk:
            sorted_features = sorted_features[:topk]
        results[metric] = dict(sorted_features)

    return results
