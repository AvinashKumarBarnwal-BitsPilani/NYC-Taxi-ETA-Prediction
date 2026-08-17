"""Tests for the Phase 4.2 baseline model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.training.baseline import calculate_regression_metrics


def test_dummy_regressor_mean_prediction():
    """Verify that the mean baseline predicts the training mean."""

    X_train = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0],
        }
    )

    y_train = pd.Series(
        [100.0, 200.0, 300.0],
        name="trip_duration",
    )

    X_val = pd.DataFrame(
        {
            "feature_1": [4.0, 5.0],
            "feature_2": [40.0, 50.0],
        }
    )

    model = DummyRegressor(
        strategy="mean"
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_val
    )

    expected_prediction = y_train.mean()

    assert np.all(
        predictions == expected_prediction
    )


def test_regression_metrics_are_correct():
    """Verify RMSE, MAE and R² calculations."""

    y_true = np.array(
        [100.0, 200.0, 300.0]
    )

    y_pred = np.array(
        [110.0, 190.0, 310.0]
    )

    metrics = calculate_regression_metrics(
        y_true,
        y_pred,
    )

    expected_rmse = np.sqrt(
        (
            10**2
            + 10**2
            + 10**2
        )
        / 3
    )

    expected_mae = 10.0

    assert metrics["rmse"] == pytest.approx(
        expected_rmse
    )

    assert metrics["mae"] == pytest.approx(
        expected_mae
    )

    assert -1.0 <= metrics["r2"] <= 1.0


def test_regression_metrics_return_expected_keys():
    """Verify the standard Phase 4 metric contract."""

    y_true = np.array(
        [100.0, 200.0, 300.0]
    )

    y_pred = np.array(
        [100.0, 200.0, 300.0]
    )

    metrics = calculate_regression_metrics(
        y_true,
        y_pred,
    )

    assert set(metrics.keys()) == {
        "rmse",
        "mae",
        "r2",
    }


def test_perfect_predictions_have_zero_error():
    """Verify that perfect predictions produce zero RMSE and MAE."""

    y_true = np.array(
        [100.0, 200.0, 300.0]
    )

    y_pred = np.array(
        [100.0, 200.0, 300.0]
    )

    metrics = calculate_regression_metrics(
        y_true,
        y_pred,
    )

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)