"""Training command for the ASCENDS CLI."""

from pathlib import Path
from typing import Optional

import typer

from ascends.cli_app import app


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

        # --- Minimal, file-free summary output (train & test) ---
        # Use values returned by core without touching metrics.csv
        try:
            tr = result.get("train_metrics", {})
            te = result.get("metrics", {}) or result.get("test_metrics", {})

            def _fmt(d):
                def fmt_val(v):
                    try:
                        f = float(v)
                        return f"{f:.4g}"
                    except Exception:
                        return v
                preferred = [
                    ("r2", "R2"),
                    ("rmse", "RMSE"),
                    ("mae", "MAE"),
                    ("accuracy", "Accuracy"),
                    ("precision", "Precision"),
                    ("recall", "Recall"),
                    ("f1", "F1"),
                    ("roc_auc", "ROC_AUC"),
                ]
                parts = []
                for key, label in preferred:
                    if key in d:
                        parts.append(f"{label}={fmt_val(d.get(key))}")
                if not parts:
                    for k, v in d.items():
                        parts.append(f"{k}={fmt_val(v)}")
                return " ".join(parts)

            if tr:
                print("Train:", _fmt(tr))
            if te:
                print("Test: ", _fmt(te))
        except Exception:
            # Keep CLI resilient; don't fail printing
            pass
