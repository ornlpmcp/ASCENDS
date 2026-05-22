"""Phase 1 model registry and GUI factory tests."""

from sklearn.linear_model import RidgeClassifier

from ascends.core.models import make_model
from ascends.gui_train_run_routes import _make_classifier


def test_core_ridge_classification_uses_ridge_classifier() -> None:
    model = make_model("classification", "ridge", random_state=42)

    assert isinstance(model, RidgeClassifier)


def test_gui_ridge_classification_uses_ridge_classifier() -> None:
    model = _make_classifier("ridge", seed=42)

    assert isinstance(model, RidgeClassifier)
