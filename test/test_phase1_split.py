"""Phase 1 split and manifest metadata tests."""

import json
from pathlib import Path

import pandas as pd

from ascends.cli_parity import _split_config_from_manifest
from ascends.core.data import SplitConfig, split_train_test
from ascends.core.train import train_model


def test_random_split_honors_stratify_col_when_available() -> None:
    df = pd.DataFrame(
        {
            "feature": range(100),
            "target": ["majority"] * 80 + ["minority"] * 20,
        }
    )

    train_df, test_df = split_train_test(
        df,
        "target",
        SplitConfig(method="random", test_size=0.25, random_state=42, stratify_col="target"),
    )

    assert test_df["target"].value_counts().to_dict() == {"majority": 20, "minority": 5}
    assert train_df["target"].value_counts().to_dict() == {"majority": 60, "minority": 15}


def test_train_model_records_split_metadata_for_classification(tmp_path: Path) -> None:
    csv_path = tmp_path / "iris_like.csv"
    df = pd.DataFrame(
        {
            "x1": list(range(30)),
            "x2": [value % 3 for value in range(30)],
            "species": ["setosa"] * 10 + ["versicolor"] * 10 + ["virginica"] * 10,
        }
    )
    df.to_csv(csv_path, index=False)

    run_dir = tmp_path / "run"
    train_model(
        csv_path=str(csv_path),
        target="species",
        task="classification",
        model="ridge",
        test_size=0.2,
        out_dir=str(run_dir),
        random_state=42,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["csv_path"] == str(csv_path)
    assert manifest["test_size"] == 0.2
    assert manifest["split"] == {
        "method": "random",
        "test_size": 0.2,
        "stratify_col": "species",
    }


def test_cli_parity_split_config_keeps_old_manifest_fallback() -> None:
    manifest = {"random_state": 42, "test_size": 0.3}

    cfg = _split_config_from_manifest(manifest)

    assert cfg.method == "random"
    assert cfg.test_size == 0.3
    assert cfg.random_state == 42
    assert cfg.stratify_col is None


def test_cli_parity_split_config_uses_new_manifest_stratify_col() -> None:
    manifest = {
        "random_state": 42,
        "split": {"method": "random", "test_size": 0.25, "stratify_col": "species"},
    }

    cfg = _split_config_from_manifest(manifest)

    assert cfg.method == "random"
    assert cfg.test_size == 0.25
    assert cfg.random_state == 42
    assert cfg.stratify_col == "species"
