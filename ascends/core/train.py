"""CV loop, final test eval, orchestration."""

from typing import Any, Dict
import pandas as pd
from pathlib import Path
from ascends.core.models import make_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    task: str,
    model_kind: str,
    tune_mode: str = "off",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Minimal training pipeline for regression:
    - Build X_train, y_train, X_test, y_test
      * One-hot encode categoricals with pd.get_dummies on train
      * Align test to train columns (missing -> 0, extra -> drop)
    - Build estimator via ascends.core.models.make_model(task, model_kind)
    - 5-fold CV on the training set:
        * R2 and MAE (use cross_val_score with scoring 'r2' and 'neg_mean_absolute_error')
        * Compute mean±std for both
    - Fit on full train, evaluate on hold-out test (R2, MAE)
    - Return:
        {
          "model": fitted_estimator,
          "features": ordered list of train feature columns,
          "cv_scores": {"r2_mean":..., "r2_std":..., "mae_mean":..., "mae_std":...},
          "test_metrics": {"r2":..., "mae":...}
        },
        "random_state": random_state
    For now ignore tune_mode!='off' (leave TODO).
    """
    from sklearn.model_selection import KFold, cross_val_score
    import numpy as np

    # 1) Build X_train, y_train, X_test, y_test (with one-hot for categoricals)
    X_train_raw = train_df.drop(columns=[target])
    X_train_dum = pd.get_dummies(X_train_raw, drop_first=False)
    X_test_raw = test_df.drop(columns=[target])
    X_test_dum = pd.get_dummies(X_test_raw, drop_first=False)
    X_train = X_train_dum.reindex(
        columns=X_train_dum.columns.union(X_test_dum.columns), fill_value=0
    )
    X_test = X_test_dum.reindex(columns=X_train.columns, fill_value=0)
    y_train = train_df[target]
    y_test = test_df[target]

    # 2) Create model
    model = make_model(task, model_kind, random_state=random_state)

    if model is None:
        raise ValueError(f"make_model returned None for task={task!r}, kind={model_kind!r}")
    model = make_model(task, model_kind, random_state=random_state)

    # 3) Build seeded CV splitter
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # 4) Cross-validation scores
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_mae = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
    )

    # 5) Fit model; predict on test; compute test R2/MAE
    model.fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    test_mae = -np.mean(np.abs(y_test - model.predict(X_test)))

    # 6) Features list
    features = list(X_train.columns)

    # 7) Return the rich dictionary
    return {
        "model": model,
        "features": features,
        "cv_scores": {
            "r2_mean": np.mean(cv_r2),
            "r2_std": np.std(cv_r2),
            "mae_mean": -np.mean(cv_mae),
            "mae_std": np.std(cv_mae),
        },
        "test_metrics": {
            "r2": test_r2,
            "mae": test_mae,
        },
        "random_state": random_state,
    }

def train_model(csv_path, target, task="r", model="rf", test_size=0.2, tune="off", tune_trials=None, out_dir="run", metrics_out=None, parity_out=None, random_state="auto"):
    """Train and evaluate a model."""
    import time
    import os
    from pathlib import Path
    import json, joblib
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Determine random seed
    import time
    import numpy as np
    if isinstance(random_state, str) and random_state.lower() == "auto":
        random_state = int(time.time() * 1000) % (2**32 - 1)

    np.random.seed(random_state)

    # --- Ensure output directory exists ---
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns.")

    # simple split
    train_df, test_df = train_test_split(
        df, test_size=float(test_size), random_state=random_state
    )

    result = train_eval(
        train_df=train_df,
        test_df=test_df,
        target=target,
        task=task,
        model_kind=model,
        tune_mode=tune or "off",
        random_state=random_state,
    )

    est = result["model"]
    feats = result["features"]

    # === Parity data for TEST & TRAIN, plus a combined file ===
    # Compute directly from the fitted estimator; don't rely on train_eval for vectors.
    y_test = test_df[target].to_numpy()
    X_test = test_df[feats]
    y_pred_test = est.predict(X_test)
    parity_test = pd.DataFrame({"actual": y_test, "predicted": y_pred_test})

    # Compute parity data for TRAIN
    y_train = train_df[target].to_numpy()
    X_train = train_df[feats]
    y_pred_train = est.predict(X_train)
    parity_train = pd.DataFrame({"actual": y_train, "predicted": y_pred_train})

    # === Metrics: write BOTH train and test, including RMSE ===
    # We compute fresh metrics here for consistency (MAE will be positive).
    r2_tr  = float(r2_score(y_train, y_pred_train))
    mae_tr = float(mean_absolute_error(y_train, y_pred_train))
    rmse_tr = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    r2_te  = float(r2_score(y_test, y_pred_test))
    mae_te = float(mean_absolute_error(y_test, y_pred_test))
    rmse_te = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    # Ensure consistent, positive metrics in the returned result
    # Some older code may not include 'train_metrics' or may carry a signed MAE.
    # Normalize here so the CLI can print without touching files.
    if "test_metrics" in result:
        tm = result["test_metrics"]
        if "mae" in tm:
            try:
                tm["mae"] = float(abs(float(tm["mae"])))
            except Exception:
                pass
    else:
        result["test_metrics"] = {"r2": r2_te, "rmse": rmse_te, "mae": mae_te}

    result.setdefault(
        "train_metrics",
        {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
    )
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
        {"split": "train", "r2": r2_tr, "mae": mae_tr, "rmse": rmse_tr},
        {"split": "test",  "r2": r2_te, "mae": mae_te, "rmse": rmse_te},
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
                "tune": tune,
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
        "schema_version": 1,
        "artifact_type": "estimator-only",
        "model": model,
        "task": task,
        "target": target,
        "features": feats,
        "random_state": random_state,
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

