"""CV loop, final test eval, orchestration."""

import logging
from typing import Any, Dict
import pandas as pd
from pathlib import Path
from ascends.core.models import make_model
from ascends.core.data import align_to_features
from ascends.utils.validation import canonicalize_task
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


logger = logging.getLogger(__name__)


def train_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    task: str,
    model_kind: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train/evaluate a model for regression or classification."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    import numpy as np

    task = canonicalize_task(task)

    # 1) Build X_train, y_train, X_test, y_test (with one-hot for categoricals)
    X_train_raw = train_df.drop(columns=[target])
    X_train_dum = pd.get_dummies(X_train_raw, drop_first=False)
    X_test_raw = test_df.drop(columns=[target])
    X_test_dum = pd.get_dummies(X_test_raw, drop_first=False)
    X_train = X_train_dum.reindex(
        columns=X_train_dum.columns.union(X_test_dum.columns), fill_value=0
    )
    y_train = train_df[target]

    # 2) Create model
    model = make_model(task, model_kind, random_state=random_state)
    if model is None:
        raise ValueError(f"make_model returned None for task={task!r}, kind={model_kind!r}")

    # 3) Build seeded CV splitter and task-specific metrics.
    cv_scores: Dict[str, float] = {}
    if task == "classification":
        class_counts = pd.Series(y_train).value_counts(dropna=True)
        if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
            n_splits = min(5, int(class_counts.min()))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            try:
                cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
                cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
                if np.isfinite(cv_acc).all() and np.isfinite(cv_f1).all():
                    cv_scores = {
                        "accuracy_mean": float(np.mean(cv_acc)),
                        "accuracy_std": float(np.std(cv_acc)),
                        "f1_mean": float(np.mean(cv_f1)),
                        "f1_std": float(np.std(cv_f1)),
                    }
            except Exception as exc:
                logger.warning("Classification cross-validation was skipped: %s", exc)
    else:
        # Five-fold R2 requires at least two observations in every test fold.
        if len(X_train) >= 10:
            cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
            try:
                cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
                cv_mae = cross_val_score(
                    model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
                )
                if np.isfinite(cv_r2).all() and np.isfinite(cv_mae).all():
                    cv_scores = {
                        "r2_mean": float(np.mean(cv_r2)),
                        "r2_std": float(np.std(cv_r2)),
                        "mae_mean": float(-np.mean(cv_mae)),
                        "mae_std": float(np.std(cv_mae)),
                    }
            except Exception as exc:
                logger.warning("Regression cross-validation was skipped: %s", exc)

    model.fit(X_train, y_train)

    # 6) Features list
    features = list(X_train.columns)

    # 7) Return the rich dictionary
    return {
        "model": model,
        "features": features,
        "cv_scores": cv_scores,
        "random_state": random_state,
    }
def train_model(csv_path, target, task="r", model="rf", test_size=0.2, out_dir="run", metrics_out=None, parity_out=None, random_state="auto"):
    """Train and evaluate a model."""
    import json
    import joblib
    import os
    import time

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Determine random seed
    if isinstance(random_state, str) and random_state.lower() == "auto":
        random_state = int(time.time() * 1000) % (2**32 - 1)

    np.random.seed(random_state)

    # --- Ensure output directory exists ---
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns.")

    task = canonicalize_task(task)

    stratify = None
    if task == "classification":
        class_counts = df[target].value_counts(dropna=True)
        class_count = len(class_counts)
        test_rows = int(np.ceil(len(df) * float(test_size)))
        train_rows = len(df) - test_rows
        if (
            class_count > 1
            and int(class_counts.min()) >= 2
            and test_rows >= class_count
            and train_rows >= class_count
        ):
            stratify = df[target]

    # simple split
    train_df, test_df = train_test_split(
        df, test_size=float(test_size), random_state=random_state, stratify=stratify
    )

    result = train_eval(
        train_df=train_df,
        test_df=test_df,
        target=target,
        task=task,
        model_kind=model,
        random_state=random_state,
    )

    est = result["model"]
    feats = result["features"]

    # === Parity data for TEST & TRAIN, plus a combined file ===
    # Compute directly from the fitted estimator; don't rely on train_eval for vectors.
    y_test = test_df[target].to_numpy()
    X_test = align_to_features(test_df.drop(columns=[target]), feats)
    y_pred_test = est.predict(X_test)
    parity_test = pd.DataFrame({"actual": y_test, "predicted": y_pred_test})

    # Compute parity data for TRAIN
    y_train = train_df[target].to_numpy()
    X_train = align_to_features(train_df.drop(columns=[target]), feats)
    y_pred_train = est.predict(X_train)
    parity_train = pd.DataFrame({"actual": y_train, "predicted": y_pred_train})

    if task == "classification":
        train_metrics = {
            "accuracy": float(accuracy_score(y_train, y_pred_train)),
            "precision": float(precision_score(y_train, y_pred_train, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_train, y_pred_train, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_train, y_pred_train, average="weighted", zero_division=0)),
        }
        test_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "precision": float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0)),
        }
        try:
            labels = pd.Series(y_test).dropna().unique()
            if len(labels) == 2 and hasattr(est, "predict_proba"):
                test_metrics["roc_auc"] = float(roc_auc_score(y_test, est.predict_proba(X_test)[:, 1]))
        except Exception:
            pass
        result["test_metrics"] = test_metrics
        result["train_metrics"] = train_metrics
    else:
        # === Metrics: write BOTH train and test, including RMSE ===
        r2_tr = float(r2_score(y_train, y_pred_train))
        mae_tr = float(mean_absolute_error(y_train, y_pred_train))
        rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        r2_te = float(r2_score(y_test, y_pred_test))
        mae_te = float(mean_absolute_error(y_test, y_pred_test))
        rmse_te = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        result["test_metrics"] = {"r2": r2_te, "rmse": rmse_te, "mae": mae_te}
        result["train_metrics"] = {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr}

    # Define model_path for saving the model
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(est, model_path)

    # Standard (always written) inside run dir:
    std_test_path  = Path(out_dir) / "parity_test.csv"
    std_train_path = Path(out_dir) / "parity_train.csv"
    std_all_path   = Path(out_dir) / "parity_all.csv"
    std_test_path.parent.mkdir(parents=True, exist_ok=True)
    std_train_path.parent.mkdir(parents=True, exist_ok=True)
    std_all_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional extra locations if --parity-out is provided:
    if parity_out:
        p = Path(str(parity_out))
    else:
        p = None

    extra_test_path = extra_train_path = extra_all_path = None
    if p is not None:
        if p.suffix.lower() == ".csv":
            extra_test_path = p
            extra_train_path = p.with_name(p.stem + "_train.csv")
            extra_all_path = p.with_name(p.stem + "_all.csv")
        else:
            # treat as directory
            p.mkdir(parents=True, exist_ok=True)
            extra_test_path = p / "parity_test.csv"
            extra_train_path = p / "parity_train.csv"
            extra_all_path = p / "parity_all.csv"

    # --- Write standard files into run dir (always) ---
    parity_test.to_csv(std_test_path, index=False)
    parity_train.to_csv(std_train_path, index=False)
    parity_all = pd.concat(
        [
            parity_train.assign(split="train"),
            parity_test.assign(split="test"),
        ],
        ignore_index=True,
    )
    parity_all.to_csv(std_all_path, index=False)

    # --- Also write to extra locations when requested ---
    if extra_test_path is not None:
        extra_test_path.parent.mkdir(parents=True, exist_ok=True)
        parity_test.to_csv(extra_test_path, index=False)
    if extra_train_path is not None:
        extra_train_path.parent.mkdir(parents=True, exist_ok=True)
        parity_train.to_csv(extra_train_path, index=False)
    if extra_all_path is not None:
        extra_all_path.parent.mkdir(parents=True, exist_ok=True)
        parity_all.to_csv(extra_all_path, index=False)

    metrics_rows = [
        {"split": "train", **result.get("train_metrics", {})},
        {"split": "test", **result.get("test_metrics", {})},
    ]
    metrics_csv = metrics_out or os.path.join(out_dir, "metrics.csv")
    # ensure parent directory exists for metrics output
    _met_dir = os.path.dirname(metrics_csv)
    if _met_dir:
        os.makedirs(_met_dir, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

    # also write a small metadata.json for convenience
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "csv_path": csv_path,
                "target": target,
                "task": task,
                "model": model,
                "test_size": test_size,
                "random_state": random_state,
                "model_path": model_path,
                "metrics_csv": metrics_csv,
            },
            f,
            indent=2,
        )

    # --- Write manifest.json for predict() ---
    from datetime import datetime
    manifest = {
        "schema_version": 2,
        "artifact_type": "estimator-only",
        "csv_path": csv_path,
        "model": model,
        "task": task,
        "target": target,
        "test_size": test_size,
        "inputs": [column for column in df.columns if column != target],
        "features": feats,
        "random_state": random_state,
        "split": {
            "method": "random",
            "test_size": test_size,
            "stratify_col": target if stratify is not None else None,
        },
        "timestamp": datetime.now().isoformat()
    }
    manifest_path = Path(out_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {manifest_path}")

    return {
        "model_path": model_path,
        "metrics": result["test_metrics"],
        "train_metrics": result["train_metrics"],
        "random_state": random_state,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "features": feats,
        "model": model,
    }
