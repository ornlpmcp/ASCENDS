"""Tests that GUI and CLI model names resolve to the same estimator semantics."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from ascends.core.models import make_model
from ascends.gui_train_run_routes import _make_classifier, _make_regressor


def test_random_forest_settings_match_between_gui_and_core() -> None:
    gui_reg = _make_regressor("rf", seed=7)
    core_reg = make_model("regression", "rf", random_state=7)
    gui_cls = _make_classifier("rf", seed=7)
    core_cls = make_model("classification", "rf", random_state=7)

    assert gui_reg.get_params()["n_estimators"] == core_reg.get_params()["n_estimators"]
    assert gui_reg.get_params()["n_jobs"] == core_reg.get_params()["n_jobs"]
    assert gui_cls.get_params()["n_estimators"] == core_cls.get_params()["n_estimators"]
    assert gui_cls.get_params()["n_jobs"] == core_cls.get_params()["n_jobs"]


def test_scaled_regression_models_match_between_gui_and_core() -> None:
    for model_name in ("ridge", "lasso", "elastic"):
        gui_model = _make_regressor(model_name, seed=7)
        core_model = make_model("regression", model_name, random_state=7)

        assert isinstance(gui_model, Pipeline)
        assert isinstance(core_model, Pipeline)
        assert list(gui_model.named_steps) == list(core_model.named_steps)
