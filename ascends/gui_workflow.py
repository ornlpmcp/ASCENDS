"""Workflow-step and sample-data helpers for the ASCENDS GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def build_workflow_steps(
    *,
    path: str,
    ws_id: str | None,
    manifest: dict[str, Any] | None = None,
    saved_runs: list[dict[str, Any]] | None = None,
    trained: bool = False,
    prediction_done: bool = False,
) -> list[dict[str, Any]]:
    """Return display state for the three-step ASCENDS GUI workflow."""
    mf = manifest or {}
    inputs = mf.get("inputs") or []
    target = mf.get("target")
    has_workspace_selection = bool(ws_id and target and inputs)
    has_saved_run = bool(saved_runs)
    has_training_result = bool(trained)

    return [
        {
            "label": "Step 1: Correlation",
            "short_label": "Correlation",
            "href": f"/correlation?ws_id={ws_id}" if ws_id else "/correlation",
            "active": path.startswith("/correlation"),
            "complete": has_workspace_selection,
        },
        {
            "label": "Step 2: Train",
            "short_label": "Train",
            "href": f"/train?ws_id={ws_id}" if ws_id else "/train",
            "active": path.startswith("/train"),
            "complete": has_saved_run or has_training_result,
        },
        {
            "label": "Step 3: Predict",
            "short_label": "Predict",
            "href": "/predict",
            "active": path.startswith("/predict"),
            "complete": bool(prediction_done),
        },
    ]


def create_sample_workspace(*, sample_csv: Path, workspace_dir: Path, target: str) -> str:
    """Create a workspace manifest pointing at a bundled sample CSV."""
    df = pd.read_csv(sample_csv, nrows=1)
    columns = list(df.columns)
    if target not in columns:
        raise ValueError(f"Sample target '{target}' not found in {sample_csv.name}.")

    ws_id = uuid4().hex
    manifest = {
        "csv_path": str(sample_csv),
        "columns": columns,
        "inputs": [column for column in columns if column != target],
        "target": target,
        "selected": columns,
        "sample_data": sample_csv.name,
    }
    ws_dir = workspace_dir / ws_id
    ws_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ws_id
