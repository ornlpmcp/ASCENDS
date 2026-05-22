"""Phase 3 model interpretation helper tests."""

from ascends.gui_interpretation import (
    CLASSIFICATION_CAUTION,
    CLASSIFICATION_STRONG,
    CV_UNAVAILABLE_CLASS_IMBALANCE,
    CV_UNAVAILABLE_SMALL_DATA,
    REGRESSION_R2_CAUTION,
    REGRESSION_R2_STRONG,
    cv_unavailable_reason,
    format_cv_summary,
    interpret_classification_metrics,
    interpret_regression_metrics,
)


def test_interpret_regression_metrics_labels_r2_conservatively() -> None:
    assert REGRESSION_R2_STRONG == 0.80
    assert REGRESSION_R2_CAUTION == 0.50

    strong = interpret_regression_metrics({"R2": 0.84, "MAE": 1.2, "RMSE": 2.4})
    caution = interpret_regression_metrics({"R2": 0.63})
    weak = interpret_regression_metrics({"R2": 0.31})

    assert strong["overall"]["label"] == "Strong"
    assert caution["overall"]["label"] == "Caution"
    assert weak["overall"]["label"] == "Weak"
    assert strong["metrics"]["MAE"]["label"] == "N/A"


def test_interpret_classification_metrics_uses_accuracy_and_f1() -> None:
    assert CLASSIFICATION_STRONG == 0.85
    assert CLASSIFICATION_CAUTION == 0.70

    strong = interpret_classification_metrics({"Accuracy": 0.91, "F1": 0.88})
    caution = interpret_classification_metrics({"Accuracy": 0.78, "F1": 0.73})
    weak = interpret_classification_metrics({"Accuracy": 0.94, "F1": 0.62})

    assert strong["overall"]["label"] == "Strong"
    assert caution["overall"]["label"] == "Caution"
    assert weak["overall"]["label"] == "Weak"
    assert strong["metrics"]["Accuracy"]["label"] == "Strong"


def test_cv_unavailable_reason_catches_small_data_and_class_imbalance() -> None:
    assert cv_unavailable_reason(row_count=299, task="r") == CV_UNAVAILABLE_SMALL_DATA
    assert (
        cv_unavailable_reason(row_count=300, task="c", class_counts={"a": 20, "b": 2})
        == CV_UNAVAILABLE_CLASS_IMBALANCE
    )
    assert cv_unavailable_reason(row_count=300, task="c", class_counts={"a": 150, "b": 150}) is None


def test_format_cv_summary_uses_mean_plus_minus_std() -> None:
    assert format_cv_summary({"metric": "R2", "mean": 0.81234, "std": 0.04567}) == "CV R2: 0.8123 ± 0.0457"
