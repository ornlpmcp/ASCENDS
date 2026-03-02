#!/usr/bin/env python3
"""Smoke-test classification flow through GUI backend train handler."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

import pandas as pd
from starlette.requests import Request

import ascends_server as server


def make_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 0),
            "server": ("testserver", 80),
        }
    )


async def run_smoke(csv_path: Path, target: str, model: str, test_size: float) -> None:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")

    inputs = [c for c in df.columns if c != target]
    if not inputs:
        raise ValueError("No input columns found.")

    ws_id = uuid4().hex
    server._save_manifest(
        ws_id,
        {
            "csv_path": str(csv_path),
            "columns": list(df.columns),
            "inputs": inputs,
            "target": target,
            "selected": list(df.columns),
        },
    )

    response = await server.train_run(
        request=make_request("/train/run"),
        ws_id=ws_id,
        task="c",
        model=model,
        test_size=test_size,
        tune="off",
        seed="42",
        resample=None,
    )
    ctx = getattr(response, "context", {})
    err = ctx.get("train_error")
    if err:
        raise RuntimeError(f"Classification train failed: {err}")

    metrics = ctx.get("metrics_test", {})
    required = {"Accuracy", "Precision", "Recall", "F1"}
    missing = [k for k in required if k not in metrics]
    if missing:
        raise RuntimeError(f"Missing classification metrics: {missing}")

    img = Path(server.STATIC_DIR) / "workspace" / ws_id / "train" / "confusion.png"
    if not img.exists():
        raise RuntimeError(f"Confusion matrix image missing: {img}")

    print("Classification GUI smoke test passed")
    print(f"  ws_id: {ws_id}")
    print(f"  target: {target}")
    print(f"  inputs: {', '.join(inputs)}")
    print(f"  metrics_test: {metrics}")
    print(f"  confusion_png: {img}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="Classification CSV path")
    parser.add_argument("--target", default="Name", help="Target column name")
    parser.add_argument("--model", default="rf", help="Model key (rf/hgb/linear/knn/xgb)")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    asyncio.run(run_smoke(args.csv, args.target, args.model, args.test_size))


if __name__ == "__main__":
    main()
