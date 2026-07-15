"""Contract tests for GUI-saved run manifests."""

from __future__ import annotations

from ascends.gui_saved_run_routes import _build_run_manifest


def test_gui_saved_run_manifest_contains_cli_compatible_schema() -> None:
    record = {
        "timestamp": "2026-07-15T12:00:00",
        "params": {"task": "c", "model": "rf", "seed": 42, "test_size": 0.2},
        "inputs": ["feature_a", "feature_b"],
        "target": "class",
        "csv_path": "/tmp/training.csv",
        "stratify_col": "class",
    }

    manifest = _build_run_manifest(record, run_name="iris_rf", ws_id="workspace")

    assert manifest["schema_version"] == 2
    assert manifest["artifact_type"] == "estimator-only"
    assert manifest["inputs"] == ["feature_a", "feature_b"]
    assert manifest["features"] == ["feature_a", "feature_b"]
    assert manifest["random_state"] == 42
    assert manifest["split"] == {
        "method": "random",
        "test_size": 0.2,
        "stratify_col": "class",
    }
