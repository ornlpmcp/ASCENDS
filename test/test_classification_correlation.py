"""Regression tests for GUI classification correlation."""

from pathlib import Path

import pandas as pd

from ascends.gui_correlation_routes import _compute_correlations, _prepare_corr_dataframe


def test_prepare_corr_dataframe_keeps_classification_string_target(tmp_path: Path) -> None:
    csv_path = tmp_path / "iris_like.csv"
    pd.DataFrame(
        {
            "SepalLength": [5.1, 6.2, 7.1, 5.0],
            "SepalWidth": [3.5, 2.8, 3.0, 3.4],
            "Name": ["setosa", "versicolor", "virginica", "setosa"],
        }
    ).to_csv(csv_path, index=False)

    df, info = _prepare_corr_dataframe(
        str(csv_path),
        "Name",
        ["SepalLength", "SepalWidth"],
        task="c",
    )

    assert info["rows_used"] == 4
    assert info["rows_dropped"] == 0
    assert df["Name"].tolist() == ["setosa", "versicolor", "virginica", "setosa"]


def test_classification_correlation_accepts_string_target(tmp_path: Path) -> None:
    csv_path = tmp_path / "iris_like.csv"
    pd.DataFrame(
        {
            "SepalLength": [5.1, 6.2, 7.1, 5.0, 6.4, 7.3],
            "SepalWidth": [3.5, 2.8, 3.0, 3.4, 3.2, 2.9],
            "Name": ["setosa", "versicolor", "virginica", "setosa", "versicolor", "virginica"],
        }
    ).to_csv(csv_path, index=False)
    df, info = _prepare_corr_dataframe(
        str(csv_path),
        "Name",
        ["SepalLength", "SepalWidth"],
        task="c",
    )

    results = _compute_correlations(
        df,
        "Name",
        info["used_inputs"],
        ["pearson", "spearman", "mi", "dcor"],
        "c",
    )

    assert set(results) == {"pearson", "spearman", "mi", "dcor"}
    assert all(not frame.empty for frame in results.values())
