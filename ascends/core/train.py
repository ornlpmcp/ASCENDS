"""CV loop, final test eval, orchestration."""

from typing import Any, Dict
import pandas as pd
from ascends.core.models import make_model


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

def train_model(
    csv_path: str,
    target: str,
    task: str,
    model: str,
    test_size: float,
    tune: str,
    out_dir: str,
    metrics_out: str | None = None,
    parity_out: str | None = None,
    tune_trials: int | None = None,   # accepted but unused for now
    random_state: int = 42,
):
    import os, json, joblib
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns.")

    # simple split
    train_df, test_df = train_test_split(
        df, test_size=float(test_size), random_state=random_state
    )

    # run the actual training/eval
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
    test_metrics = result["test_metrics"]

    # save model
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump({"estimator": est, "features": feats, "task": task, "target": target}, model_path)

    # save parity data
    parity = pd.DataFrame({
        "actual": result["y_test"].to_numpy(),
        "predicted": result["y_pred"].to_numpy(),
    })
    parity_csv = parity_out or os.path.join(out_dir, "parity_test.csv")
    parity.to_csv(parity_csv, index=False)

    # save metrics
    metrics_csv = metrics_out or os.path.join(out_dir, "metrics.csv")
    pd.DataFrame([test_metrics]).to_csv(metrics_csv, index=False)

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
                "parity_csv": parity_csv,
                "metrics_csv": metrics_csv,
            },
            f,
            indent=2,
        )

    return {"model_path": model_path, "metrics": test_metrics, "parity_csv": parity_csv}

