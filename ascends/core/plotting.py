from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use("Agg")  # Use Agg backend for headless environments


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]
) -> dict:
    """
    Return dict with floats:
      {
        "r2": float,
        "mae": float,
        "rmse": float
      }
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def make_parity_plot(
    df: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "predicted",
    *,
    title: Optional[str] = None,
    metrics: Optional[dict] = None,
    color: str = "C0",
    marker: str = "o",
    alpha: float = 0.5,
    label: Optional[str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    dpi: int = 150,
    out_path: Optional[str] = None,
    equal_axes: bool = True,
    show_identity: bool = True,
    limit_quantile: Optional[float] = None,
) -> "matplotlib.figure.Figure":
    """
    Create an actual-vs-predicted parity scatter plot (PNG if out_path is given).
    - If metrics is None, compute metrics from df[actual_col], df[pred_col].
    - Compute axis limits from combined min/max of actual & predicted.
    - Optional quantile clipping if limit_quantile is provided.
    - Draw identity line y=x (dashed).
    - Scatter points with alpha & marker.
    - Add a small annotation box in the top-left with:
        R² = ...
        MAE = ...
        RMSE = ...
    - Default equal axes so diagonal means perfect parity.
    - Save to out_path (PNG) if provided; return the figure object.
    - Use headless backend (Agg) to ensure it works in servers/CI.
    """
    # TODO: Implement the parity plot creation logic.
    # This includes computing metrics, setting axis limits, plotting points, and saving the figure.
