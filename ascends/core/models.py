"""Model registry (RF, XGB, Linear/LogReg, optional SVM/KNN)."""

from ascends.utils.validation import canonicalize_task
from typing import Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore

    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None  # type: ignore[assignment]
    XGBClassifier = None  # type: ignore[assignment]
    HAS_XGBOOST = False


SUPPORTED_TASKS = {"regression", "classification"}
TASK_ALIASES = {
    "r": "regression",
    "reg": "regression",
    "regression": "regression",
    "c": "classification",
    "clf": "classification",
    "class": "classification",
    "classification": "classification",
}

KIND_ALIASES = {
    "rf": "rf",
    "random_forest": "rf",
    "xgb": "xgb",
    "xgboost": "xgb",
    "linear": "linear",
    "ridge": "ridge",
    "lasso": "lasso",
    "elastic": "elasticnet",
    "elasticnet": "elasticnet",
    "hgb": "hgb",
    "hist_gradient_boosting": "hgb",
    "svr": "svr",
    "knn": "knn",
}


def _normalize(task: str, kind: str):
    t = TASK_ALIASES.get((task or "").lower().strip())
    k = KIND_ALIASES.get((kind or "").lower().strip())
    return t, k


def make_model(task: str, kind: str, random_state: Optional[int] = None):
    task, kind = _normalize(task, kind)
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task: {task!r}. Supported: {sorted(SUPPORTED_TASKS)} "
            f"(aliases: {sorted(set(TASK_ALIASES) - SUPPORTED_TASKS)})"
        )
    if kind is None:
        raise ValueError(
            f"Unsupported model kind for task={task!r}. "
            f"Known aliases: {sorted(KIND_ALIASES)}"
        )

    if task == "regression":
        if kind == "linear":
            return LinearRegression()
        if kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        if kind == "lasso":
            return make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=10000))
        if kind == "elasticnet":
            return make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
            )
        if kind == "rf":
            return RandomForestRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1
            )
        if kind == "hgb":
            return HistGradientBoostingRegressor(random_state=random_state)
        if kind == "svr":
            return make_pipeline(
                StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1)
            )
        if kind == "knn":
            return make_pipeline(StandardScaler(), KNeighborsRegressor())
        if kind == "xgb":
            try:
                from xgboost import XGBRegressor
            except Exception as e:
                raise ImportError(
                    "xgboost is required for --model xgb. Install with: "
                    "`uv sync` (or `uv pip install xgboost`)."
                ) from e
            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=random_state,
                tree_method="hist",
                n_jobs=0,
                verbosity=0,
            )
        raise ValueError(
            f"Unsupported regression model kind: {kind!r}. Try 'rf' (or 'random_forest')."
        )

    if task == "classification":
        if kind == "linear":
            return LogisticRegression(max_iter=2000, random_state=random_state)
        if kind == "ridge":
            return RidgeClassifier(random_state=random_state)
        if kind == "hgb":
            return HistGradientBoostingClassifier(random_state=random_state)
        if kind == "knn":
            return make_pipeline(StandardScaler(), KNeighborsClassifier())
        if kind == "rf":
            return RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1
            )
        if kind == "xgb":
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                raise ImportError(
                    "xgboost is required for --model xgb. Install with: "
                    "`uv sync` (or `uv pip install xgboost`)."
                ) from e
            return XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=random_state,
                tree_method="hist",
                n_jobs=0,
                verbosity=0,
            )
        raise ValueError(
            f"Unsupported classification model kind: {kind!r}. Try 'rf' (or 'random_forest')."
        )


def is_tree_model(est) -> bool:
    """
    Return True if estimator is a tree ensemble suitable for SHAP TreeExplainer
    (RF/XGB/HGB, reg+clf), else False.
    """
    tree_types = [
        RandomForestRegressor,
        RandomForestClassifier,
        HistGradientBoostingRegressor,
        HistGradientBoostingClassifier,
    ]
    if HAS_XGBOOST:
        tree_types.extend([XGBRegressor, XGBClassifier])  # type: ignore[arg-type]
    tree_types_t = tuple(tree_types)

    if isinstance(est, tree_types_t):
        return True
    if isinstance(est, Pipeline):
        return any(isinstance(step, tree_types_t) for step in est.named_steps.values())
    return False


def list_supported_models(task: str) -> list[str]:
    """
    Returns the allowed 'kind' strings for the given task.
    """
    task = canonicalize_task(task)
    if task == "regression":
        models = [
            "linear",
            "ridge",
            "lasso",
            "elasticnet",
            "rf",
            "hgb",
            "svr",
            "knn",
        ]
        if HAS_XGBOOST:
            models.insert(5, "xgb")
        return models
    else:
        models = [
            "linear",
            "ridge",
            "rf",
            "hgb",
            "knn",
        ]
        if HAS_XGBOOST:
            models.insert(3, "xgb")
        return models
