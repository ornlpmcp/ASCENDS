"""CLI entrypoint (correlation, train, shap, predict)."""

import os
import json
from pathlib import Path
from typing import Optional
import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import List, Tuple, Union
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict
import uvicorn

from ascends.core.correlation import run_correlation
from ascends.core.models import list_supported_models
from ascends.core.predict import batch_predict as core_predict

import typer.main as _typer_main

class _CmdOrder(_typer_main.TyperGroup):
    DESIRED = ["gui", "correlation", "train", "parity-plot", "shap", "predict"]
    def list_commands(self, ctx):
        names = list(self.commands.keys())
        ordered = [n for n in self.DESIRED if n in names]
        remainder = [n for n in names if n not in self.DESIRED]
        return ordered + remainder

app = typer.Typer(cls=_CmdOrder, no_args_is_help=True)

models_txt = ", ".join(list_supported_models("regression"))


@app.command("correlation", help="Run correlation analysis.")
def correlation(
    csv: str = typer.Option(..., help="Path to CSV dataset"),
    target: str = typer.Option(..., help="Target column"),
    task: str = typer.Option(..., help="Task: r|regression or c|classification"),
    metrics: str = typer.Option(
        "pearson,spearman", help="Comma-separated: pearson,spearman,mi,dcor"
    ),
    view: str = typer.Option("long", help="Output layout: long|wide"),
    sort_by: str = typer.Option(
        "combined", help="Sort key for wide view: combined|pearson|spearman|mi|dcor"
    ),
    topk: int | None = typer.Option(
        None, help="Limit to top-k features (after sorting)"
    ),
    format: str = typer.Option("table", help="Output format: table|json"),
    out: str | None = typer.Option(
        None,
        help="If set, write results to CSV file (wide view writes a header, long view writes metric/feature/score rows)",
    ),
    random_state: int | None = typer.Option(
        None, help="Optional seed for metrics with randomness (e.g., MI)"
    ),
):
    """Run correlation analysis."""
    # Normalize task
    if task in ("r", "regression"):
        task = "regression"
    elif task in ("c", "classification"):
        task = "classification"
    else:
        raise typer.BadParameter("task must be 'r|regression' or 'c|classification'")

    df = pd.read_csv(csv)

    # If random_state provided and MI used, set numpy random seed for determinism
    if random_state is not None:

        np.random.seed(int(random_state))

    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    results = run_correlation(df, target, task, metrics_list, topk)

    console = Console()

    if view == "wide":
        # Expect a dict-like {metric: {feature: score}}
        # Convert to a DataFrame so we can sort and optionally write CSV

        # Build wide DF
        all_feats = set()
        for m in metrics_list:
            all_feats.update(results[m].keys())
        rows = []
        for feat in sorted(all_feats):
            row = {"feature": feat}
            for m in metrics_list:
                row[m] = results[m].get(feat, None)
            rows.append(row)
        wide_df = pd.DataFrame(rows)

        # Optional combined column for ranking if requested
        if sort_by == "combined":
            # Average absolute scores across available metrics

            metric_cols = [m for m in metrics_list if m in wide_df.columns]
            if metric_cols:
                wide_df["combined"] = np.nanmean(
                    [wide_df[m].abs() for m in metric_cols], axis=0
                )
                sort_key = "combined"
            else:
                # Fallback: if nothing to combine, just sort by the first available column
                sort_key = next((c for c in ["pearson", "spearman"] if c in wide_df.columns), "feature")
        else:
            sort_key = sort_by
        if sort_key not in wide_df.columns:
            raise typer.BadParameter(
                f"--sort-by '{sort_by}' not found in columns {list(wide_df.columns)}"
            )

        wide_df = wide_df.sort_values(by=sort_key, ascending=False, na_position="last")
        if topk:
            wide_df = wide_df.head(topk)

        if format == "json":
            console.print(wide_df.to_json(orient="records", indent=2))
        else:
            # pretty table
            tbl = Table(title="Correlation Analysis Results")
            for col in wide_df.columns:
                tbl.add_column(col)
            for _, r in wide_df.iterrows():
                tbl.add_row(
                    *[
                        f"{v:.6f}" if isinstance(v, float) else str(v)
                        for v in r.tolist()
                    ]
                )
            console.print(tbl)

        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            wide_df.to_csv(out, index=False)

    else:
        # long view: list of dicts [{metric: [feat order]}] OR score dicts — adapt to your current run_correlation long shape
        # If your current run_correlation returns scores per feature per metric, show as three columns
        tbl = Table(title="Correlation Analysis Results")
        tbl.add_column("Metric")
        tbl.add_column("Feature")
        tbl.add_column("Score", justify="right")
        # Construct a long list preserving order per metric
        for m in metrics_list:
            # If run_correlation already returns scores dict: {feature: score}
            scores = results.get(m, {})
            # Sort by abs(score) desc
            for feat, sc in sorted(
                scores.items(), key=lambda kv: abs(kv[1]), reverse=True
            )[: (topk or len(scores))]:
                tbl.add_row(m, feat, f"{float(sc):.6f}")
        if format == "json":
            # Emit JSON records

            long_rows = []
            for m in metrics_list:
                scores = results.get(m, {})
                for feat, sc in sorted(
                    scores.items(), key=lambda kv: abs(kv[1]), reverse=True
                )[: (topk or len(scores))]:
                    long_rows.append({"metric": m, "feature": feat, "score": float(sc)})
            typer.echo(json.dumps(long_rows, indent=2))
        else:
            console.print(tbl)


@app.command()
def gui(
    port: int = typer.Option(7777, help="Port number to run the ASCENDS GUI"),
    reload: bool = typer.Option(True, help="Auto-reload on code changes (dev mode)"),
):
    """Launch the ASCENDS GUI (FastAPI-based ascends_server).

    Example:
      uv run ascends gui
    """
    typer.echo(f"🚀 Launching ASCENDS GUI on http://127.0.0.1:{port} (reload={reload})")
    uvicorn.run("ascends_server:app", host="127.0.0.1", port=port, reload=reload)
try:
    from ascends.core.train import train_model as _asc_train_model
except Exception:
    _asc_train_model = None

@app.command(help="Train a model and save run artifacts.")
def train(
    csv: Path = typer.Option(..., "--csv", help="Input CSV file"),
    target: str = typer.Option(..., "--target", help="Target column"),
    task: str = typer.Option(
        "r",
        "--task",
        help="Task type. Accepts aliases: r|reg|regression, c|clf|class|classification",
    ),
    model: str = typer.Option("rf", "--model", help="rf|xgb|hgb|svr|knn|linear|ridge|lasso|elastic"),
    test_size: float = typer.Option(0.2, "--test-size", min=0.05, max=0.5, help="Test split fraction"),
    tune: str = typer.Option("off", "--tune", help="off|quick|intense|optuna|bayes (case-insensitive)"),
    tune_trials: Optional[int] = typer.Option(None, "--tune-trials", help="Override number of tuning trials (if supported)"),
    out: Path = typer.Option(..., "--out", help="Output run directory"),
    metrics_out: Optional[Path] = typer.Option(None, "--metrics-out", help="Write metrics CSV here"),
    parity_out: Optional[Path] = typer.Option(None, "--parity-out", help="Write parity plots here"),
    random_state: Optional[str] = typer.Option("auto", "--random-state", help="Random seed (int or 'auto' for time-based)"),
):
    """
    Example:
      uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf_v2 --tune quick
    """
    if _asc_train_model is None:
        typer.secho("ascends.core.train not available.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Normalize and map legacy tune presets
    _tune_in = (tune or "off").strip().lower()
    if _tune_in not in {"off", "quick", "intense", "optuna", "bayes"}:
        typer.secho(f"Invalid --tune value: {tune}. Use off|quick|intense|optuna|bayes.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Default trial counts (used when supported downstream)
    preset_trials = {
        "quick": 30,
        "optuna": 100,
        "intense": 200,
        "bayes": 100,
    }

    if _tune_in == "off":
        tune_method = "off"
        trials = None
    elif _tune_in in {"quick", "intense"}:
        tune_method = "optuna"   # legacy presets map to optuna under the hood
        trials = tune_trials if tune_trials is not None else preset_trials[_tune_in]
    elif _tune_in in {"optuna", "bayes"}:
        tune_method = _tune_in
        trials = tune_trials if tune_trials is not None else preset_trials[_tune_in]
    else:
        tune_method = "off"
        trials = None

    # Call core training. Prefer passing trials if supported; otherwise fall back.
    try:
        result = _asc_train_model(
            csv_path=str(csv),
            target=target,
            task=task,
            model=model,
            test_size=test_size,
            tune=tune_method,
            out_dir=str(out),
            metrics_out=str(metrics_out) if metrics_out else None,
            parity_out=str(parity_out) if parity_out else None,
            tune_trials=trials,  # OK if the function accepts it
            random_state=random_state,
        )
    except TypeError:
        # Older core without tune_trials support
        result = _asc_train_model(
            csv_path=str(csv),
            target=target,
            task=task,
            model=model,
            test_size=test_size,
            tune=tune_method,
            out_dir=str(out),
            metrics_out=str(metrics_out) if metrics_out else None,
            parity_out=str(parity_out) if parity_out else None,
        )

    if result is not None:
        msg = f"Training complete. tune={tune_method}"
        if trials is not None:
            msg += f", trials={trials}"
        typer.echo(msg)

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
    figsize_str: str = typer.Option("6,3.7", help="Figure size as 'W,H'"),
    alpha: float = typer.Option(0.8, help="Alpha for plot points"),
    train_marker: str = typer.Option("o", help="Marker for train points"),
    test_marker: str = typer.Option("s", help="Marker for test points"),
    train_color: Optional[str] = typer.Option(None, help="Color for train points"),
    test_color: Optional[str] = typer.Option(None, help="Color for test points"),
    no_identity: bool = typer.Option(False, help="Do not draw the y=x line"),
    equal_axes: bool = typer.Option(False, help="Set equal axes"),
    limit: Optional[float] = typer.Option(None, help="Limit for axes"),
    save_parity_if_missing: bool = typer.Option(False, help="Regenerate parity data if missing")
):
    """Generate parity plot(s) for a saved run."""
    import os
    import pandas as pd
    from ascends.core.data import SplitConfig, split_train_test
    from ascends.core.serialize import load_model

    parity_train_path = os.path.join(run_dir, "parity_train.csv")
    parity_test_path = os.path.join(run_dir, "parity_test.csv")
    df_train, df_test = None, None
    if scope in {"train", "both", "combined"} and os.path.exists(parity_train_path):
        df_train = pd.read_csv(parity_train_path)
    if scope in {"test", "both", "combined"} and os.path.exists(parity_test_path):
        df_test = pd.read_csv(parity_test_path)

    import json

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    model_kind = manifest.get("model", "?")
    target_name = manifest.get("target", "?")
    if (df_train is None or df_test is None) and save_parity_if_missing:
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
        tr, te = split_train_test(
            df,
            manifest['target'],
            SplitConfig(
                method=manifest['split'].get('method', 'random'),
                test_size=manifest['split']['test_size'],
                random_state=manifest['random_state']
            )
        )
        # One-hot encode and reindex
        Xtrain = pd.get_dummies(tr.drop(columns=[manifest['target']]), drop_first=False).reindex(columns=manifest['features'], fill_value=0)
        Xtest = pd.get_dummies(te.drop(columns=[manifest['target']]), drop_first=False).reindex(columns=manifest['features'], fill_value=0)
        ytrain = tr[manifest['target']]
        ytest = te[manifest['target']]
        if df_train is None:
            preds_train = model.predict(Xtrain)
            df_train = pd.DataFrame({'actual': ytrain, 'predicted': preds_train})
            df_train.to_csv(parity_train_path, index=False)
        if df_test is None:
            preds_test = model.predict(Xtest)
            df_test = pd.DataFrame({'actual': ytest, 'predicted': preds_test})
            df_test.to_csv(parity_test_path, index=False)

    if df_train is None or df_test is None:
        raise typer.BadParameter("Required parity data missing and --save-parity-if-missing not set.")

    # Load existing parity data if available
    parity_train_path = os.path.join(run_dir, "parity_train.csv")
    parity_test_path = os.path.join(run_dir, "parity_test.csv")
    df_train, df_test = None, None
    if scope in {"train", "both", "combined"} and os.path.exists(parity_train_path):
        df_train = pd.read_csv(parity_train_path)
    if scope in {"test", "both", "combined"} and os.path.exists(parity_test_path):
        df_test = pd.read_csv(parity_test_path)

    # Regenerate missing parity data if needed
    if (df_train is None or df_test is None) and save_parity_if_missing:
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
        tr, te = split_train_test(
            df,
            manifest['target'],
            SplitConfig(
                method=manifest['split']['method'],
                test_size=manifest['split']['test_size'],
                random_state=manifest['random_state']
            )
        )
        # One-hot encode and reindex
        Xtrain = pd.get_dummies(tr.drop(columns=[manifest['target']]), drop_first=False).reindex(columns=manifest['features'], fill_value=0)
        Xtest = pd.get_dummies(te.drop(columns=[manifest['target']]), drop_first=False).reindex(columns=manifest['features'], fill_value=0)
        ytrain = tr[manifest['target']]
        ytest = te[manifest['target']]
        if df_train is None:
            preds_train = model.predict(Xtrain)
            df_train = pd.DataFrame({'actual': ytrain, 'predicted': preds_train})
            df_train.to_csv(parity_train_path, index=False)
        if df_test is None:
            preds_test = model.predict(Xtest)
            df_test = pd.DataFrame({'actual': ytest, 'predicted': preds_test})
            df_test.to_csv(parity_test_path, index=False)

    if df_train is None or df_test is None:
        raise typer.BadParameter("Required parity data missing and --save-parity-if-missing not set.")

    if out is None:
        out = os.path.join(run_dir, "plots")
    # Prepare output directory or file
    if scope == "both" and not os.path.isdir(out):
        raise typer.BadParameter("Output must be a directory when scope is 'both'.")
    if scope == "combined" and not os.path.isdir(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    else:
        os.makedirs(out, exist_ok=True)

    # Parse figsize once
    try:
        figsize = _parse_figsize(figsize_str)
    except ValueError:
        raise typer.BadParameter("figsize must be 'W,H' (e.g., '6,3.7')")
    # Set default colors if not provided
    if train_color is None:
        train_color = "C0"
    if test_color is None:
        test_color = "C1"

    if scope in {"train", "both", "combined"}:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        metrics_train = _compute_metrics(df_train['actual'], df_train['predicted'])
        title_train = f"Parity Plot — Train (model={model_kind}, target={target_name}, n={len(df_train)})"
        _plot_single(ax, df_train, "Train", train_color, train_marker, alpha, not no_identity, equal_axes, limit, title_train, _build_metrics_box_text(metrics_train), draw_metrics=True, metrics=metrics_train)
        train_out_path = os.path.join(out, "parity_train.png") if os.path.isdir(out) else out
        fig.savefig(train_out_path)
        plt.close(fig)
        typer.echo(f"Saved train parity plot to {train_out_path}")

    if scope in {"test", "both", "combined"}:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        metrics_test = _compute_metrics(df_test['actual'], df_test['predicted'])
        title_test = f"Parity Plot — Test (model={model_kind}, target={target_name}, n={len(df_test)})"
        _plot_single(ax, df_test, "Test", test_color, test_marker, alpha, not no_identity, equal_axes, limit, title_test, _build_metrics_box_text(metrics_test), draw_metrics=True, metrics=metrics_test)
        test_out_path = os.path.join(out, "parity_test.png") if os.path.isdir(out) else out
        fig.savefig(test_out_path)
        plt.close(fig)
        typer.echo(f"Saved test parity plot to {test_out_path}")

    if scope == "combined":
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        metrics_train = _compute_metrics(df_train['actual'], df_train['predicted'])
        metrics_test = _compute_metrics(df_test['actual'], df_test['predicted'])
        title_combined = f"Parity Plot — Combined (model={model_kind}, target={target_name}, n_train={len(df_train)}, n_test={len(df_test)})"
        _plot_single(ax, df_train, "Train", train_color, train_marker, alpha, not no_identity, equal_axes, limit, title_combined, _build_metrics_box_text(metrics_train), label="Train", draw_metrics=False)
        _plot_single(ax, df_test, "Test", test_color, test_marker, alpha, not no_identity, equal_axes, limit, title_combined, _build_metrics_box_text(metrics_test), label="Test", draw_metrics=False)
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




@app.command(help="Compute SHAP values and plots for a saved run (placeholder).")
def shap(
    run_dir: Path = typer.Argument(..., help="Run directory containing model.joblib & manifest.json"),
    out: Optional[Path] = typer.Option(None, "--out", help="Directory to save SHAP plots (optional)"),
):
    """
    Placeholder command.
    To enable SHAP:
      uv add shap
    Then we will compute summary plots for tree models (RF/XGB/HGB).
    """
    typer.secho("SHAP command placeholder. We'll wire this after correlation/train.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=0)
@app.command(help="Run batch predictions using a saved model.")
def predict(
    run_dir: Path = typer.Argument(..., help="Run directory containing model.joblib & manifest.json"),
    csv: Path = typer.Option(..., "--csv", help="Feature CSV to score (headers case-insensitive)"),
    out: Path = typer.Option(..., "--out", help="Directory to write predictions.csv"),
):
    """
    Example:
      uv run ascends predict runs/boston_rf_v2 --csv examples/BostonHousing_test.csv --out runs/predict
    """
    model_path = run_dir / "model.joblib"
    manifest_path = run_dir / "manifest.json"
    if not model_path.exists():
        typer.secho(f"Missing model file: {model_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not manifest_path.exists():
        typer.secho(f"Missing manifest: {manifest_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Required features list from manifest (case-insensitive matching)
    feat_keys = (
        manifest.get("features")
        or manifest.get("input_features")
        or manifest.get("X_features")
        or manifest.get("X_cols")
    )
    if not feat_keys:
        typer.secho("Manifest does not include 'features' list. Retrain with a newer ASCENDS version.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = pd.read_csv(csv)
    # Build case-insensitive column mapping
    in_map = {c.lower(): c for c in df.columns}
    missing: List[str] = []
    ordered_cols: List[str] = []
    for f in feat_keys:
        key = str(f).lower()
        if key in in_map:
            ordered_cols.append(in_map[key])
        else:
            missing.append(f)
    if missing:
        typer.secho(
            "Input CSV is missing required features (case-insensitive): "
            + ", ".join(missing),
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    X = df[ordered_cols]
    Path(out).mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        typer.secho(f"Prediction failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    core_predict(model_path=model_path, data=df, out_dir=out, run_dir=run_dir)
    out_csv = out / "predictions.csv"
    out_df = df.copy()
    out_df["prediction"] = y_pred
    out_df.to_csv(out_csv, index=False)

    typer.echo(f"Predictions written to: {out_csv}")


if __name__ == "__main__":
    app()
