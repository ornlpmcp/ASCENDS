"""SHAP/feature-importance command for the ASCENDS CLI."""

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer

from ascends.cli_app import app
from ascends.core.data import align_to_features
from ascends.core.explain import explain_model as core_explain, save_importance_plot
from ascends.utils.validation import canonicalize_task


@app.command(help="Compute feature importance for a saved run (SHAP for tree models, permutation fallback).")
def shap(
    run_dir: Path = typer.Argument(..., help="Run directory containing model.joblib & manifest.json"),
    out: Optional[Path] = typer.Option(None, "--out", help="Directory to save SHAP plots (optional)"),
    csv: Optional[Path] = typer.Option(None, "--csv", help="Optional CSV override for explanation dataset"),
    max_samples: int = typer.Option(500, "--max-samples", min=50, help="Max number of rows sampled for explanation"),
):
    model_path = run_dir / "model.joblib"
    manifest_path = run_dir / "manifest.json"
    metadata_path = run_dir / "metadata.json"

    if not model_path.exists():
        typer.secho(f"Missing model file: {model_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not manifest_path.exists():
        typer.secho(f"Missing manifest: {manifest_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    task_raw = manifest.get("task") or metadata.get("task") or "regression"
    task = canonicalize_task(task_raw)
    target = manifest.get("target") or metadata.get("target")
    feature_keys = (
        manifest.get("features")
        or manifest.get("inputs")
        or manifest.get("input_features")
        or manifest.get("X_features")
        or manifest.get("X_cols")
    )

    csv_path = csv or metadata.get("csv_path") or manifest.get("csv_path")
    if not csv_path:
        typer.secho(
            "Could not determine dataset path. Pass --csv explicitly or include csv_path in metadata.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    csv_path = Path(csv_path)
    if not csv_path.exists():
        typer.secho(f"Dataset not found: {csv_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = pd.read_csv(csv_path)
    if target and target in df.columns:
        X_raw = df.drop(columns=[target])
        y = df[target]
    else:
        X_raw = df
        y = None

    if feature_keys:
        keys = list(feature_keys)
        if all(k in X_raw.columns for k in keys):
            X = X_raw[keys].copy()
        else:
            X = align_to_features(X_raw, keys)
    else:
        X = pd.get_dummies(X_raw, drop_first=False)

    obj = joblib.load(model_path)
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    try:
        expl = core_explain(model=model, X=X, y=y, task=task, max_samples=max_samples, random_state=42)
    except Exception as e:
        typer.secho(f"SHAP/explain failed: {e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)

    out_dir = out or (run_dir / "shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    imp_df = expl["importance_df"]
    csv_out = out_dir / "shap_importance.csv"
    imp_df.to_csv(csv_out, index=False)

    png_out = out_dir / "shap_importance.png"
    save_importance_plot(imp_df, png_out, method=str(expl.get("method", "shap")), top_n=20)

    report = {
        "method": expl.get("method"),
        "task": task,
        "target": target,
        "csv_path": str(csv_path),
        "n_samples": expl.get("n_samples"),
        "warning": expl.get("warning"),
        "importance_csv": str(csv_out),
        "importance_png": str(png_out),
    }
    report_out = out_dir / "shap_report.json"
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if expl.get("warning"):
        typer.secho(f"Warning: {expl['warning']}", fg=typer.colors.YELLOW)
    typer.echo(f"SHAP/explain complete ({expl['method']}).")
    typer.echo(f"- CSV:   {csv_out}")
    typer.echo(f"- Plot:  {png_out}")
    typer.echo(f"- Report:{report_out}")
