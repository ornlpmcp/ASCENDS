"""Correlation command for the ASCENDS CLI."""

import json
import os

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ascends.cli_app import app
from ascends.core.correlation import run_correlation


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
