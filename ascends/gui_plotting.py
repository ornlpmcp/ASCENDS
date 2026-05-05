"""Plotting helpers used by the ASCENDS FastAPI GUI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_metric_bars(
    scores: pd.DataFrame,
    metric: str,
    target: str,
    n_used: int,
    out_png: Path,
    top_k: Optional[int] = None,
) -> None:
    """Save a bar plot for a single correlation metric."""
    dfp = scores.copy()
    if metric in {"pearson", "spearman"}:
        dfp = dfp.sort_values(by="score", key=lambda s: np.abs(s), ascending=False)
    else:
        dfp = dfp.sort_values(by="score", ascending=False)
    if top_k and top_k > 0:
        dfp = dfp.head(top_k)

    fig_w = 8.0
    fig_h = fig_w / 1.618
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    x = np.arange(len(dfp))
    ax.bar(x, dfp["score"])
    ax.set_xticks(x)
    ax.set_xticklabels(list(dfp["feature"]), rotation=55, ha="right")

    ax.set_xlabel("Feature")
    ax.set_ylabel("Score")
    ax.set_title(f"{metric.title()} vs. {target}  (N={n_used})")

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if metric in {"pearson", "spearman"}:
        ax.axhline(0.0, linewidth=0.8, alpha=0.6, color="black")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def train_img_dir(static_dir: Path, ws_id: str) -> Path:
    """Return the static image directory for train-tab artifacts."""
    directory = static_dir / "workspace" / ws_id / "train"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_parity_plot(
    static_dir: Path,
    ws_id: str,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    metrics_train: dict[str, float],
    metrics_test: dict[str, float],
) -> str:
    """Save a parity plot PNG and return its static URL."""
    img_dir = train_img_dir(static_dir, ws_id)
    out_png = img_dir / "parity.png"

    phi = (1 + 5**0.5) / 2
    width = 8.0
    height = width / phi

    all_actual = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    vmin = float(np.nanmin([all_actual.min(), all_pred.min()]))
    vmax = float(np.nanmax([all_actual.max(), all_pred.max()]))
    pad = 0.02 * (vmax - vmin) if vmax > vmin else 1.0
    lo, hi = vmin - pad, vmax + pad

    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, alpha=0.8)
    ax.scatter(y_train, y_pred_train, s=14, alpha=0.7, label="Train")
    ax.scatter(y_test, y_pred_test, s=18, alpha=0.8, marker="x", label="Test")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(loc="upper left", frameon=True)

    box_text = (
        f"Train - R$^2$={metrics_train['R2']:.3f}, MAE={metrics_train['MAE']:.3f}, RMSE={metrics_train['RMSE']:.3f}\n"
        f"Test  - R$^2$={metrics_test['R2']:.3f}, MAE={metrics_test['MAE']:.3f}, RMSE={metrics_test['RMSE']:.3f}"
    )
    ax.text(
        0.98,
        0.02,
        box_text,
        transform=ax.transAxes,
        fontsize=16,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, linewidth=0.5),
        zorder=5,
    )

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return f"/static/workspace/{ws_id}/train/parity.png?ts={int(time.time())}"


def save_confusion_plot(
    static_dir: Path,
    ws_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
) -> str:
    """Save a confusion matrix PNG and return its static URL."""
    img_dir = train_img_dir(static_dir, ws_id)
    out_png = img_dir / "confusion.png"

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        colorbar=True,
        xticks_rotation=30,
        ax=ax,
    )
    disp.ax_.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return f"/static/workspace/{ws_id}/train/confusion.png?ts={int(time.time())}"
