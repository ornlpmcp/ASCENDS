"""Phase 1 train_model metrics contract tests."""

from pathlib import Path

import pandas as pd

from ascends.core.train import train_model


def test_train_model_writes_regression_train_and_test_metrics(tmp_path: Path) -> None:
    csv_path = tmp_path / "regression.csv"
    df = pd.DataFrame(
        {
            "x1": [float(value) for value in range(30)],
            "x2": [float(value % 5) for value in range(30)],
            "target": [float(value * 2 + 1) for value in range(30)],
        }
    )
    df.to_csv(csv_path, index=False)

    run_dir = tmp_path / "run_regression"
    train_model(
        csv_path=str(csv_path),
        target="target",
        task="regression",
        model="ridge",
        test_size=0.2,
        out_dir=str(run_dir),
        random_state=42,
    )

    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics["split"].tolist() == ["train", "test"]
    assert {"r2", "rmse", "mae"}.issubset(metrics.columns)


def test_train_model_writes_classification_train_and_test_metrics(tmp_path: Path) -> None:
    csv_path = tmp_path / "classification.csv"
    df = pd.DataFrame(
        {
            "x1": list(range(60)),
            "x2": [value % 4 for value in range(60)],
            "target": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        }
    )
    df.to_csv(csv_path, index=False)

    run_dir = tmp_path / "run_classification"
    train_model(
        csv_path=str(csv_path),
        target="target",
        task="classification",
        model="ridge",
        test_size=0.2,
        out_dir=str(run_dir),
        random_state=42,
    )

    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics["split"].tolist() == ["train", "test"]
    assert {"accuracy", "precision", "recall", "f1"}.issubset(metrics.columns)
