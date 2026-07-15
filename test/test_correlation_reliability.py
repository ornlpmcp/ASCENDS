"""Reliability coverage for classification correlation analysis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ascends.cli import app
from ascends.core.correlation import run_correlation


def test_core_classification_correlation_encodes_string_target_and_drops_aligned_nans() -> (
    None
):
    """Numeric association metrics use deterministic class codes and complete cases."""
    df = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0],
            "class": [
                "zeta",
                "alpha",
                "alpha",
                "alpha",
                "zeta",
                "zeta",
                "zeta",
                "zeta",
                np.nan,
                "zeta",
            ],
        }
    )

    results = run_correlation(
        df,
        target="class",
        task="classification",
        metrics=["pearson", "spearman", "mi", "dcor"],
        mi_neighbors=2,
    )

    expected_x = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0])
    expected_y = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert results["pearson"]["feature"] == np.corrcoef(expected_x, expected_y)[0, 1]
    assert all(
        np.isfinite(results[metric]["feature"])
        for metric in ("pearson", "spearman", "mi", "dcor")
    )


def test_core_tiny_dataset_raises_clear_complete_case_error() -> None:
    """Insufficient complete cases fail before an undefined score is emitted."""
    with pytest.raises(ValueError, match="requires at least 2 aligned complete cases"):
        run_correlation(
            pd.DataFrame({"feature": [1.0], "class": ["alpha"]}),
            target="class",
            task="classification",
            metrics=["pearson", "spearman", "mi", "dcor"],
        )


def test_cli_classification_correlation_accepts_string_target_with_missing_values(
    tmp_path,
) -> None:
    """The public CLI emits JSON scores for a string classification target."""
    csv_path = tmp_path / "classification.csv"
    pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0],
            "class": [
                "zeta",
                "alpha",
                "alpha",
                "alpha",
                "zeta",
                "zeta",
                "zeta",
                "zeta",
                np.nan,
                "zeta",
            ],
        }
    ).to_csv(csv_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "correlation",
            "--csv",
            str(csv_path),
            "--target",
            "class",
            "--task",
            "classification",
            "--metrics",
            "pearson,spearman,dcor",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    scores = json.loads(result.output)
    assert {row["metric"] for row in scores} == {"pearson", "spearman", "dcor"}
    assert all(np.isfinite(row["score"]) for row in scores)
