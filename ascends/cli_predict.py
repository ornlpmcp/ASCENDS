"""Prediction command for the ASCENDS CLI."""

import json
from pathlib import Path
from typing import Dict, List

import click
import pandas as pd
import typer

from ascends.cli_app import app
from ascends.core.predict import batch_predict as core_predict


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
        or manifest.get("inputs")
        or manifest.get("input_features")
        or manifest.get("X_features")
        or manifest.get("X_cols")
    )
    if not feat_keys:
        typer.secho(
            "Manifest does not include input feature columns "
            "('features' or 'inputs'). Retrain and save the model again.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    df = pd.read_csv(csv)
    # Build robust mapping:
    # 1) exact-case match first
    # 2) otherwise fallback to case-insensitive match only when unique
    lower_candidates: Dict[str, List[str]] = {}
    for c in df.columns:
        lower_candidates.setdefault(str(c).lower(), []).append(c)
    missing: List[str] = []
    ordered_cols: List[str] = []
    for f in feat_keys:
        if f in df.columns:
            ordered_cols.append(f)
            continue
        key = str(f).lower()
        cands = lower_candidates.get(key, [])
        if len(cands) == 1:
            ordered_cols.append(cands[0])
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

    Path(out).mkdir(parents=True, exist_ok=True)
    try:
        result = core_predict(model_path=model_path, data=df, out_dir=out, run_dir=run_dir)
    except Exception as e:
        typer.secho(f"Prediction failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    pred_col = result.get("pred_col", "prediction")
    click.echo(f"Predictions saved to {out}/predictions.csv ({pred_col})")
