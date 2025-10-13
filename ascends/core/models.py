"""Model registry (RF, XGB, Linear/LogReg, optional SVM/KNN)."""

from ascends.utils.validation import canonicalize_task
from typing import Union, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    raise ImportError(
        "XGBoost is required for kind='xgb'. Install with: uv add xgboost"
    )


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
        if kind == "rf":
            return RandomForestRegressor(random_state=random_state)
        if kind == "xgb":
            try:
                from xgboost import XGBRegressor
            except Exception as e:
                raise ImportError(
                    "xgboost is required for --model xgb. Install with: "
                    "`uv pip install xgboost`"
                ) from e
            return XGBRegressor(random_state=random_state)
        raise ValueError(f"Unsupported regression model kind: {kind!r}. Try 'rf' (or 'random_forest').")

    if task == "classification":
        if kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=random_state)
        if kind == "xgb":
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                raise ImportError(
                    "xgboost is required for --model xgb. Install with: "
                    "`uv pip install xgboost`"
                ) from e
            return XGBClassifier(random_state=random_state)
        raise ValueError(f"Unsupported classification model kind: {kind!r}. Try 'rf' (or 'random_forest').")


def is_tree_model(est) -> bool:
    """
    Return True if estimator is a tree ensemble suitable for SHAP TreeExplainer
    (RandomForestRegressor, XGBRegressor, HistGradientBoostingRegressor), else False.
    """
    if isinstance(
        est, (RandomForestRegressor, XGBRegressor, HistGradientBoostingRegressor)
    ):
        return True
    if isinstance(est, Pipeline):
        return any(
            isinstance(
                step,
                (RandomForestRegressor, XGBRegressor, HistGradientBoostingRegressor),
            )
            for step in est.named_steps.values()
        )
    return False


def list_supported_models(task: str) -> list[str]:
    """
    Returns the allowed 'kind' strings for the given task.
    """
    task = canonicalize_task(task)
    if task == "regression":
        return [
            "linear",
            "ridge",
            "lasso",
            "elasticnet",
            "rf",
            "xgb",
            "hgb",
            "svr",
            "knn",
        ]
    else:
        raise ValueError(f"Unsupported task: {task}")
