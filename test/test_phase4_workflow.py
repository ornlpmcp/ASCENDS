"""Phase 4 workflow visibility and safe-delete tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ascends.gui_saved_run_routes import delete_saved_run, delete_saved_run_confirmation
from ascends.gui_workflow import build_workflow_steps, create_sample_workspace


def test_build_workflow_steps_marks_expected_completion() -> None:
    steps = build_workflow_steps(
        path="/train",
        ws_id="abc123",
        manifest={"target": "medv", "inputs": ["rm", "lstat"]},
        saved_runs=[{"name": "rf_run"}],
        prediction_done=False,
    )

    assert [step["label"] for step in steps] == [
        "Step 1: Correlation",
        "Step 2: Train",
        "Step 3: Predict",
    ]
    assert [step["complete"] for step in steps] == [True, True, False]
    assert steps[1]["active"] is True


def test_create_sample_workspace_writes_manifest_with_defaults(tmp_path: Path) -> None:
    sample_csv = tmp_path / "iris.csv"
    sample_csv.write_text(
        "SepalLength,SepalWidth,PetalLength,PetalWidth,Name\n"
        "5.1,3.5,1.4,0.2,Iris-setosa\n",
        encoding="utf-8",
    )
    workspace_dir = tmp_path / "workspace"

    ws_id = create_sample_workspace(
        sample_csv=sample_csv,
        workspace_dir=workspace_dir,
        target="Name",
        task="c",
    )

    manifest = json.loads((workspace_dir / ws_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["csv_path"] == str(sample_csv)
    assert manifest["columns"] == ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Name"]
    assert manifest["target"] == "Name"
    assert manifest["task"] == "c"
    assert manifest["train_params"]["task"] == "c"
    assert manifest["inputs"] == ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
    assert manifest["selected"] == manifest["columns"]


def test_train_delete_requires_confirmation(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "demo_run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"name":"demo_run"}', encoding="utf-8")

    confirmation = delete_saved_run_confirmation(runs_dir, "demo_run")

    assert confirmation["message"] == "Delete saved model 'demo_run'? This cannot be undone."
    assert run_dir.exists()

    result = delete_saved_run(runs_dir, "demo_run")

    assert result == "Deleted run: demo_run"
    assert not run_dir.exists()


def test_train_delete_rejects_path_traversal(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="Invalid run name"):
        delete_saved_run(runs_dir, "../outside")

    assert outside.exists()
