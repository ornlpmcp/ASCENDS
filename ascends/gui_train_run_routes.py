"""Train execution route for the ASCENDS FastAPI GUI."""

from __future__ import annotations

import time
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ascends.core.data import NON_ASCII_COLUMN_MESSAGE, warn_non_ascii_columns
from ascends.gui_interpretation import (
    interpret_classification_metrics,
    interpret_regression_metrics,
    small_dataset_warning,
)
from ascends.gui_messages import (
    append_notice,
    attach_error_recovery,
    format_missing_columns_message,
    friendly_error,
    rows_removed_message,
    stratify_disabled_message,
)
from ascends.gui_plotting import save_confusion_plot, save_parity_plot

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None


def _make_regressor(key: str, seed: Optional[int] = 42):
    k = (key or "rf").lower()
    if k == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    if k == "xgb" and xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            n_jobs=0,
            verbosity=0,
        )
    if k == "hgb":
        return HistGradientBoostingRegressor(random_state=seed)
    if k == "svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1))
    if k == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    if k == "linear":
        return LinearRegression()
    if k == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if k == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=10000))
    if k == "elastic":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))
    return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)


def _make_classifier(key: str, seed: Optional[int] = 42):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    k = (key or "rf").lower()
    if k == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if k == "xgb" and xgb is not None:
        return xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            n_jobs=0,
            verbosity=0,
        )
    if k == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    if k == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if k == "linear":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if k == "ridge":
        return RidgeClassifier(random_state=seed)
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def _save_parity_plot(
    static_dir: Path,
    ws_id: str,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    metrics_train: dict[str, float],
    metrics_test: dict[str, float],
) -> str:
    return save_parity_plot(
        static_dir,
        ws_id,
        y_train,
        y_pred_train,
        y_test,
        y_pred_test,
        metrics_train,
        metrics_test,
    )


def _save_confusion_plot(
    static_dir: Path,
    ws_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
) -> str:
    return save_confusion_plot(static_dir, ws_id, y_true, y_pred, labels)


def create_train_run_router(
    *,
    templates: Jinja2Templates,
    static_dir: Path,
    load_manifest: Callable[[str], dict[str, Any]],
    list_saved_runs: Callable[[], list[dict[str, Any]]],
    last_train: dict[str, Any],
) -> APIRouter:
    """Create the Train tab model-fitting route."""
    router = APIRouter()

    def _add_non_ascii_notice(ctx: dict[str, Any], columns) -> None:
        columns_with_non_ascii = warn_non_ascii_columns(columns)
        if columns_with_non_ascii:
            append_notice(
                ctx,
                f"{NON_ASCII_COLUMN_MESSAGE} Columns: {', '.join(columns_with_non_ascii)}",
                level="warning",
            )

    @router.post("/train/run", response_class=HTMLResponse)
    async def train_run(
        request: Request,
        ws_id: str = Form(...),
        task: str = Form(...),
        model: str = Form(...),
        test_size: float = Form(...),
        seed: Optional[str] = Form(None),
        resample: Optional[str] = Form(None),
    ) -> HTMLResponse:
        ctx: dict[str, Any] = {"request": request, "ws_id": ws_id}
        mf = load_manifest(ws_id) or {}
        all_columns = mf.get("columns", [])
        inputs = mf.get("inputs", [])
        target = mf.get("target")

        if resample:
            seed_val = int(time.time()) & 0xFFFFFFFF
        elif seed not in (None, ""):
            try:
                seed_val = int(seed)
            except Exception:
                seed_val = 42
        else:
            seed_val = 42

        ctx.update(
            {
                "csv_path": mf.get("csv_path"),
                "all_columns": all_columns,
                "selected": mf.get("selected", []),
                "inputs": inputs,
                "target": target,
                "train_params": {
                    "task": task,
                    "model": model,
                    "test_size": test_size,
                    "seed": seed_val,
                    "resample": bool(resample),
                },
            }
        )

        task = (task or "r").lower()
        if task not in {"r", "c"}:
            ctx["train_error"] = "Task must be 'r' (regression) or 'c' (classification)."
            attach_error_recovery(ctx, "train", ws_id=ws_id)
            return templates.TemplateResponse("train.html", ctx)
        if not inputs or not target:
            ctx["train_error"] = "Please select at least one input and a target."
            attach_error_recovery(ctx, "train", ws_id=ws_id)
            return templates.TemplateResponse("train.html", ctx)

        csv_path = mf.get("csv_path")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            ctx["train_error"] = friendly_error(e, "train")
            attach_error_recovery(ctx, "train", ws_id=ws_id)
            return templates.TemplateResponse("train.html", ctx)
        _add_non_ascii_notice(ctx, df.columns)

        selected_columns = list(inputs) + [target]
        missing_columns = [column for column in selected_columns if column not in df.columns]
        if missing_columns:
            ctx["train_error"] = format_missing_columns_message(missing_columns)
            attach_error_recovery(ctx, "train", ws_id=ws_id)
            return templates.TemplateResponse("train.html", ctx)
        needed = list(inputs) + [target]

        df2 = df[needed].dropna(axis=0, how="any")
        rows_dropped = len(df[needed]) - len(df2)
        if rows_dropped > 0:
            append_notice(ctx, rows_removed_message(rows_dropped), level="info")
        small_warning = small_dataset_warning(len(df2))
        if small_warning:
            append_notice(ctx, small_warning, level="warning")
        X = df2[inputs]
        y = df2[target]

        try:
            ts = float(test_size)
        except Exception:
            ts = 0.2
        stratify_vec = y if task == "c" and y.nunique(dropna=True) > 1 else None
        if task == "c" and stratify_vec is None:
            append_notice(ctx, stratify_disabled_message(), level="warning")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=ts,
            random_state=seed_val,
            stratify=stratify_vec,
        )

        est = _make_regressor(model, seed=seed_val) if task == "r" else _make_classifier(model, seed=seed_val)
        try:
            est.fit(X_train, y_train)
            y_pred_train = est.predict(X_train)
            y_pred_test = est.predict(X_test)
        except Exception as e:
            ctx["train_error"] = friendly_error(e, "train")
            attach_error_recovery(ctx, "train", ws_id=ws_id)
            return templates.TemplateResponse("train.html", ctx)

        if task == "r":

            def _metrics_reg(y_true, y_pred):
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = sqrt(mean_squared_error(y_true, y_pred))
                return {"R2": r2, "MAE": mae, "RMSE": rmse}

            ctx["metrics_train"] = _metrics_reg(y_train, y_pred_train)
            ctx["metrics_test"] = _metrics_reg(y_test, y_pred_test)
            ctx["metric_interpretation"] = interpret_regression_metrics(ctx["metrics_test"])
            try:
                ctx["parity_img_url"] = _save_parity_plot(
                    static_dir,
                    ws_id,
                    y_train,
                    y_pred_train,
                    y_test,
                    y_pred_test,
                    ctx["metrics_train"],
                    ctx["metrics_test"],
                )
            except Exception as e:
                ctx["train_error"] = friendly_error(e, "train")
                attach_error_recovery(ctx, "train", ws_id=ws_id)
        else:

            def _metrics_clf(y_true, y_pred, estm, X_eval):
                out = {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                    "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                }
                try:
                    classes = pd.Series(y_true).dropna().unique()
                    if len(classes) == 2 and hasattr(estm, "predict_proba"):
                        y_proba = estm.predict_proba(X_eval)
                        out["ROC_AUC"] = roc_auc_score(y_true, y_proba[:, 1])
                except Exception:
                    pass
                return out

            ctx["metrics_train"] = _metrics_clf(y_train, y_pred_train, est, X_train)
            ctx["metrics_test"] = _metrics_clf(y_test, y_pred_test, est, X_test)
            ctx["metric_interpretation"] = interpret_classification_metrics(ctx["metrics_test"])
            try:
                labels = sorted(pd.Series(y).dropna().unique().tolist())
                ctx["parity_img_url"] = _save_confusion_plot(
                    static_dir,
                    ws_id,
                    np.asarray(y_test),
                    np.asarray(y_pred_test),
                    labels,
                )
            except Exception as e:
                ctx["train_error"] = friendly_error(e, "train")
                attach_error_recovery(ctx, "train", ws_id=ws_id)

        last_train_max = 20
        if ws_id not in last_train and len(last_train) >= last_train_max:
            del last_train[next(iter(last_train))]
        last_train[ws_id] = {
            "estimator": est,
            "params": ctx["train_params"],
            "inputs": inputs,
            "target": target,
            "csv_path": csv_path,
            "metrics_train": ctx["metrics_train"],
            "metrics_test": ctx["metrics_test"],
            "parity_img_url": ctx.get("parity_img_url"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        ctx["saved_runs"] = list_saved_runs()
        return templates.TemplateResponse("train.html", ctx)

    return router
