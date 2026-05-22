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
    out: Path = typer.Option(..., "--out", help="Output run directory"),
    metrics_out: Optional[Path] = typer.Option(None, "--metrics-out", help="Write metrics CSV here"),
    parity_out: Optional[Path] = typer.Option(None, "--parity-out", help="Write parity plots here"),
    random_state: Optional[str] = typer.Option("auto", "--random-state", help="Random seed (int or 'auto' for time-based)"),
):
    """
    Example:
      uv run ascends train --csv examples/BostonHousing.csv --target medv --task r --model rf --out runs/boston_rf_v2
    """
    if _asc_train_model is None:
        typer.secho("ascends.core.train not available.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    result = _asc_train_model(
        csv_path=str(csv),
        target=target,
        task=task,
        model=model,
        test_size=test_size,
        out_dir=str(out),
        metrics_out=str(metrics_out) if metrics_out else None,
        parity_out=str(parity_out) if parity_out else None,
        random_state=random_state,
    )

    if result is not None:
        typer.echo("Training complete.")

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
