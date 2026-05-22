"""Phase 1 missing-column message tests."""

from pathlib import Path

import pandas as pd

from ascends.gui_correlation_routes import _prepare_corr_dataframe
from ascends.gui_messages import format_missing_columns_message


def test_prepare_corr_dataframe_reports_missing_selected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "target": [2, 4, 6]}).to_csv(csv_path, index=False)

    _, info = _prepare_corr_dataframe(str(csv_path), "target", ["x", "missing_feature"])

    assert info["missing_columns"] == ["missing_feature"]


def test_missing_columns_message_lists_missing_columns() -> None:
    message = format_missing_columns_message(["a", "B"])

    assert message == "Selected columns not found in CSV (case-sensitive check): a, B"
