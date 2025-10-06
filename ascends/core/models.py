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


def make_model(
    task: str, kind: str, random_state: Optional[int] = None
) -> Union[RandomForestRegressor, XGBRegressor, Pipeline]:
    """
    Returns a sklearn-compatible estimator for the given task and kind, with an optional random_state for reproducibility.
    Supported kinds for task='regression':
      - 'linear': Pipeline([StandardScaler(), LinearRegression()])
      - 'ridge':  Pipeline([StandardScaler(), Ridge()])
      - 'lasso':  Pipeline([StandardScaler(), Lasso()])
      - 'elasticnet': Pipeline([StandardScaler(), ElasticNet()])
      - 'rf': RandomForestRegressor(random_state=random_state)
      - 'xgb': XGBRegressor(random_state=random_state, n_estimators=300, learning_rate=0.05)
      - 'hgb': HistGradientBoostingRegressor(random_state=random_state)
      - 'svr': Pipeline([StandardScaler(), SVR(kernel='rbf')])
      - 'knn': Pipeline([StandardScaler(), KNeighborsRegressor()])
    Raise ValueError on unsupported task or kind.
    """
    if task == "regression":
        if kind == "rf":
            return RandomForestRegressor(random_state=random_state)
        elif kind == "xgb":
            return XGBRegressor(random_state=random_state, n_estimators=300, learning_rate=0.05)
        elif kind == "hgb":
            return HistGradientBoostingRegressor(random_state=random_state)
        elif kind == "linear":
            return LinearRegression()
        elif kind == "ridge":
            return Ridge(random_state=random_state) if hasattr(Ridge(), "random_state") else Ridge()
        elif kind == "lasso":
            return Lasso(random_state=random_state) if hasattr(Lasso(), "random_state") else Lasso()
        elif kind == "elasticnet":
            return ElasticNet(random_state=random_state) if hasattr(ElasticNet(), "random_state") else ElasticNet()
        elif kind == "svr":
            return SVR()  # no random_state
        elif kind == "knn":
            return KNeighborsRegressor()
        else:
            raise ValueError(f"Unsupported regression kind: {kind}")
    elif task == "classification":
        # (Return proper classifiers per kind, mirroring the pattern)
        ...
    else:
        raise ValueError(f"Unsupported task: {task}")


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
