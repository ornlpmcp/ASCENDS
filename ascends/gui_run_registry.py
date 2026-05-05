"""Run-directory naming and listing helpers for the ASCENDS GUI."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def slugify_name(name: str) -> str:
    """Return a filesystem-friendly run-name stem."""
    name = name.strip()
    if not name:
        return ""
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return slug.strip("_-")


def unique_run_name(base: str) -> str:
    """Return a unique run name and atomically create its directory under runs/."""
    base = slugify_name(base) or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base
    n = 2
    while True:
        try:
            (RUNS_DIR / candidate).mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = f"{base}_{n}"
            n += 1


def list_saved_runs() -> list[dict[str, Any]]:
    """Scan runs/* and load manifest plus metrics for the ML Models pane."""
    out: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return out
    for path in sorted(RUNS_DIR.iterdir()):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        metrics_path = path / "metrics.csv"
        item: dict[str, Any] = {"name": path.name}
        try:
            if manifest_path.exists():
                item.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception:
            pass

        if metrics_path.exists():
            try:
                metrics_df = pd.read_csv(metrics_path)
                test_row = metrics_df[metrics_df["split"].str.lower() == "test"]
                if not test_row.empty:
                    row = test_row.iloc[0]
                    item.setdefault("metrics", {})
                    for column in metrics_df.columns:
                        if column != "split":
                            try:
                                item["metrics"][column] = float(row.get(column, float("nan")))
                            except Exception:
                                item["metrics"][column] = row.get(column)
            except Exception:
                pass
        out.append(item)
    return out
