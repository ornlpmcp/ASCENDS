"""Parity plotting command for the ASCENDS CLI."""

from typing import Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ascends.cli_app import app


def _parse_figsize(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    # Accept "W,H" or (W, H). Return (float(W), float(H)).
    if isinstance(s, tuple) and len(s) == 2:
        return (float(s[0]), float(s[1]))
    if isinstance(s, str):
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise ValueError("figsize must be 'W,H' (e.g., '6,3.7').")
        return (float(parts[0]), float(parts[1]))
    raise ValueError("figsize must be a 'W,H' string or a (W, H) tuple.")

def _compute_metrics(y, yhat) -> Dict[str, float]:
    return {
        "r2": r2_score(y, yhat),
        "mae": mean_absolute_error(y, yhat),
        "rmse": np.sqrt(mean_squared_error(y, yhat))
    }

def _build_metrics_box_text(metrics: Dict[str, float]) -> str:
    return (
        f"R² = {metrics['r2']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}"
    )

def _draw_metrics_box(ax, metrics: Dict[str, float], corner: str) -> None:
    """
    Draw a metrics box inside the axes in a specified corner.
    'corner' in {"lower left", "lower right", "upper left", "upper right"}.
    """
    pos_map = {
        "lower left":  ((0.02, 0.02), "left",  "bottom"),
        "lower right": ((0.98, 0.02), "right", "bottom"),
        "upper left":  ((0.02, 0.98), "left",  "top"),
        "upper right": ((0.98, 0.98), "right", "top"),
    }
    (x, y), ha, va = pos_map.get(corner, ((0.98, 0.02), "right", "bottom"))
    txt = f"R²={metrics['r2']:.3f}\nMAE={metrics['mae']:.3f}\nRMSE={float(metrics['rmse']):.3f}"
    ax.text(
        x, y, txt,
        transform=ax.transAxes,
        ha=ha, va=va, fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="round,pad=0.3"),
        zorder=6,
    )

def _plot_single(ax, df, subset_label, color, marker, alpha, draw_identity, equal_axes, limit, title, metrics_block_text, label: Optional[str] = None, draw_metrics: bool = True, metrics: Optional[Dict[str, float]] = None):
    ax.scatter(df['actual'], df['predicted'], alpha=alpha, c=color, marker=marker, label=label)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    if draw_identity:
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='gray', linestyle='--')
    if equal_axes:
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, limit if limit else df[['actual', 'predicted']].max().max())
        ax.set_ylim(0, limit if limit else df[['actual', 'predicted']].max().max())
    if draw_metrics and metrics:
        ax.text(0.05, 0.95, metrics_block_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

@app.command("parity-plot", help="Generate parity plot(s) for a saved run (train/test/both/combined).")
def parity_plot(
    run_dir: str = typer.Argument(..., help="Path to a trained run directory"),
    scope: str = typer.Option("both", help="Scope of the plot: test|train|both|combined"),
    out: str = typer.Option(None, help="Output path for the plot(s)"),
    dpi: int = typer.Option(300, help="DPI for the plot"),
    figsize: str = typer.Option("6,3.7", "--figsize", help="Figure size as 'W,H'"),
    alpha: float = typer.Option(0.8, help="Alpha for plot points"),
    train_marker: str = typer.Option("o", help="Marker for train points"),
    test_marker: str = typer.Option("s", help="Marker for test points"),
    train_color: Optional[str] = typer.Option(None, help="Color for train points"),
    test_color: Optional[str] = typer.Option(None, help="Color for test points"),
    identity: bool = typer.Option(True, "--identity/--no-identity", help="Draw the y=x line"),
    equal_axes: bool = typer.Option(False, "--equal-axes/--auto-axes", help="Use equal x/y axes"),
    limit: Optional[float] = typer.Option(None, help="Limit for axes"),
    save_parity_if_missing: bool = typer.Option(False, help="Regenerate parity data if missing")
):
    """Generate parity plot(s) for a saved run."""
    import os
    import pandas as pd
    from ascends.core.data import SplitConfig, split_train_test
    from ascends.core.serialize import load_model

    if scope not in {"train", "test", "both", "combined"}:
        raise typer.BadParameter("scope must be one of: train|test|both|combined")

    need_train = scope in {"train", "both", "combined"}
    need_test = scope in {"test", "both", "combined"}

    parity_train_path = os.path.join(run_dir, "parity_train.csv")
    parity_test_path = os.path.join(run_dir, "parity_test.csv")
    df_train, df_test = None, None
    if need_train and os.path.exists(parity_train_path):
        df_train = pd.read_csv(parity_train_path)
    if need_test and os.path.exists(parity_test_path):
        df_test = pd.read_csv(parity_test_path)

    import json

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    model_kind = manifest.get("model", "?")
    target_name = manifest.get("target", "?")
    if ((need_train and df_train is None) or (need_test and df_test is None)) and save_parity_if_missing:
        manifest_path = os.path.join(run_dir, "manifest.json")
        model_path = os.path.join(run_dir, "model.joblib")
        if not os.path.exists(manifest_path) or not os.path.exists(model_path):
            raise typer.BadParameter("Manifest or model file missing in run directory.")

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Check for required fields in manifest
        if 'csv_path' not in manifest or 'features' not in manifest:
            raise typer.BadParameter("Manifest missing 'csv_path' or 'features'. Re-train with ASCENDS.")

        # Load model
        model = load_model(model_path)

        df = pd.read_csv(manifest['csv_path'])
        split_cfg = manifest.get("split", {}) if isinstance(manifest.get("split"), dict) else {}
        split_method = split_cfg.get("method", "random")
        split_test_size = float(split_cfg.get("test_size", manifest.get("test_size", 0.2)))
        tr, te = split_train_test(
            df,
            manifest['target'],
            SplitConfig(
                method=split_method,
                test_size=split_test_size,
                random_state=manifest['random_state']
            )
        )
        # One-hot encode and reindex
        Xtrain = pd.get_dummies(tr.drop(columns=[manifest['target']]), drop_first=False).reindex(columns=manifest['features'], fill_value=0)
        Xtest = pd.get_dummies(te.drop(columns=[manifest['target']]), drop_first=False).reindex(columns=manifest['features'], fill_value=0)
        ytrain = tr[manifest['target']]
        ytest = te[manifest['target']]
        if need_train and df_train is None:
            preds_train = model.predict(Xtrain)
            df_train = pd.DataFrame({'actual': ytrain, 'predicted': preds_train})
            df_train.to_csv(parity_train_path, index=False)
        if need_test and df_test is None:
            preds_test = model.predict(Xtest)
            df_test = pd.DataFrame({'actual': ytest, 'predicted': preds_test})
            df_test.to_csv(parity_test_path, index=False)

    if (need_train and df_train is None) or (need_test and df_test is None):
        raise typer.BadParameter("Required parity data missing and --save-parity-if-missing not set.")

    if out is None:
        out = os.path.join(run_dir, "plots")
    # Prepare output directory or file
    if scope == "both" and not os.path.isdir(out):
        raise typer.BadParameter("Output must be a directory when scope is 'both'.")
    if scope == "combined" and not os.path.isdir(out):
        os.makedirs(out, exist_ok=True)
    else:
        os.makedirs(out, exist_ok=True)

    # Parse figsize once
    try:
        figsize_tuple = _parse_figsize(figsize)
    except ValueError:
        raise typer.BadParameter("figsize must be 'W,H' (e.g., '6,3.7')")
    # Set default colors if not provided
    if train_color is None:
        train_color = "C0"
    if test_color is None:
        test_color = "C1"

    if scope in {"train", "both", "combined"}:
        fig, ax = plt.subplots(figsize=figsize_tuple, dpi=dpi)
        metrics_train = _compute_metrics(df_train['actual'], df_train['predicted'])
        title_train = f"Parity Plot — Train (model={model_kind}, target={target_name}, n={len(df_train)})"
        _plot_single(ax, df_train, "Train", train_color, train_marker, alpha, identity, equal_axes, limit, title_train, _build_metrics_box_text(metrics_train), draw_metrics=True, metrics=metrics_train)
        train_out_path = os.path.join(out, "parity_train.png") if os.path.isdir(out) else out
        fig.savefig(train_out_path)
        plt.close(fig)
        typer.echo(f"Saved train parity plot to {train_out_path}")

    if scope in {"test", "both", "combined"}:
        fig, ax = plt.subplots(figsize=figsize_tuple, dpi=dpi)
        metrics_test = _compute_metrics(df_test['actual'], df_test['predicted'])
        title_test = f"Parity Plot — Test (model={model_kind}, target={target_name}, n={len(df_test)})"
        _plot_single(ax, df_test, "Test", test_color, test_marker, alpha, identity, equal_axes, limit, title_test, _build_metrics_box_text(metrics_test), draw_metrics=True, metrics=metrics_test)
        test_out_path = os.path.join(out, "parity_test.png") if os.path.isdir(out) else out
        fig.savefig(test_out_path)
        plt.close(fig)
        typer.echo(f"Saved test parity plot to {test_out_path}")

    if scope == "combined":
        fig, ax = plt.subplots(figsize=figsize_tuple, dpi=dpi)
        metrics_train = _compute_metrics(df_train['actual'], df_train['predicted'])
        metrics_test = _compute_metrics(df_test['actual'], df_test['predicted'])
        title_combined = f"Parity Plot — Combined (model={model_kind}, target={target_name}, n_train={len(df_train)}, n_test={len(df_test)})"
        _plot_single(ax, df_train, "Train", train_color, train_marker, alpha, identity, equal_axes, limit, title_combined, _build_metrics_box_text(metrics_train), label="Train", draw_metrics=False)
        _plot_single(ax, df_test, "Test", test_color, test_marker, alpha, identity, equal_axes, limit, title_combined, _build_metrics_box_text(metrics_test), label="Test", draw_metrics=False)
        _draw_metrics_box(ax, metrics_train, "lower left")
        _draw_metrics_box(ax, metrics_test, "lower right")
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            fancybox=True,
            title="Subset",
            fontsize=10,
            title_fontsize=11,
            borderpad=0.4,
            handlelength=1.2,
            handletextpad=0.6
        )
        combined_out_path = os.path.join(out, "parity_combined.png") if os.path.isdir(out) else out
        fig.savefig(combined_out_path)
        plt.close(fig)
        typer.echo(f"Saved combined parity plot to {combined_out_path}")
