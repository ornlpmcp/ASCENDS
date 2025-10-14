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
    return min(default, max(1, n - 1))


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
    n = len(df)
    task = canonicalize_task(task)
    y = np.ravel(df[target].values)

    # Initialize results dictionary for each metric
    results = {metric: {} for metric in metrics}

    # Select only numeric columns
    X = df.drop(columns=[target]).select_dtypes(include=[float, int])

    for metric in metrics:
        for feature in X.columns:
            if metric == "pearson":
                corr, _ = pearsonr(X[feature], y)
            elif metric == "spearman":
                corr, _ = spearmanr(X[feature], y)
            elif metric == "mi":
                if n < 3:
                    corr = None  # or np.nan
                else:
                    k = _safe_neighbors(n, mi_neighbors)
                    if task == "regression":
                        corr = mutual_info_regression(X[[feature]], y, n_neighbors=k)[0]
                    else:
                        corr = mutual_info_classif(X[[feature]], y, n_neighbors=k)[0]
            elif metric == "dcor":
                if n < 2:
                    corr = None  # or np.nan
                else:
                    # distance correlation (dcor fast path needs float arrays)
                    x_dc = X[feature].dropna().to_numpy()
                    y_dc = y.dropna().to_numpy()
                    x_dc = np.asarray(x_dc, dtype=np.float64)
                    y_dc = np.asarray(y_dc, dtype=np.float64)
                    corr = dcor.distance_correlation(x_dc, y_dc)
            results[metric][feature] = corr

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
