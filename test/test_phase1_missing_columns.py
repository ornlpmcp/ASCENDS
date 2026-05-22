"""Phase 1 missing-column message tests."""

from pathlib import Path

import pandas as pd

from ascends.gui_correlation_routes import _compute_correlations, _prepare_corr_dataframe
from ascends.gui_messages import format_missing_columns_message


def test_prepare_corr_dataframe_reports_missing_selected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "target": [2, 4, 6]}).to_csv(csv_path, index=False)

    _, info = _prepare_corr_dataframe(str(csv_path), "target", ["x", "missing_feature"])

    assert info["missing_columns"] == ["missing_feature"]


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


def test_missing_columns_message_lists_missing_columns() -> None:
    message = format_missing_columns_message(["a", "B"])

    assert message == "Selected columns not found in CSV (case-sensitive check): a, B"
