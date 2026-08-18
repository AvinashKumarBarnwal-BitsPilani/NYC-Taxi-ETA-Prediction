"""Tests for Phase 4.6 final model."""

import numpy as np

from src.training.final_model import (
    load_best_tuning_params,
    train_final_model,
)
from src.training.final_model_validation import evaluate_final_model

# from src.training.final_model import (
#     SELECTED_XGBOOST_PARAMS,
#     train_final_model,
# )
from src.training.final_model_validation import evaluate_final_model


def test_final_model_uses_selected_xgboost_configuration():
    """Verify the final model uses the selected Optuna parameters."""

    selected_params = load_best_tuning_params()

    model = train_final_model(
    X_train=np.array([[1.0], [2.0], [3.0], [4.0]]),
    y_train=np.array([10.0, 20.0, 30.0, 40.0]),
    selected_params=selected_params,
    )

    for parameter, expected_value in selected_params.items():
        assert model.get_params()[parameter] == expected_value


def test_final_model_evaluation_returns_required_metrics():
    """Verify final model evaluation returns RMSE, MAE and R²."""

    selected_params = load_best_tuning_params()

    model = train_final_model(
        X_train=np.array([[1.0], [2.0], [3.0], [4.0]]),
        y_train=np.array([10.0, 20.0, 30.0, 40.0]),
        selected_params=selected_params,
    )

    metrics = evaluate_final_model(
        model,
        np.array([[1.5], [2.5]]),
        np.array([15.0, 25.0]),
    )

    assert set(metrics.keys()) == {"rmse", "mae", "r2"}

    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["r2"], float)

    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0