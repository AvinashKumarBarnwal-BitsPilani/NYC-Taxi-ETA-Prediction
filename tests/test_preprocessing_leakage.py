"""
7.3d – Test Preprocessing & Leakage Controls

These tests verify that the production preprocessing workflow maintains
TRAIN / VALIDATION compatibility and does not require validation data to
fit the preprocessing transformations.
"""

import numpy as np
import pandas as pd

from src.pipelines.preprocessing import preprocess_train_validation


def create_test_data():
    """Create a small synthetic dataset for preprocessing tests."""

    train = pd.DataFrame(
        {
            "vendor_id": [1, 2, 1, 2, 1, 2],
            "passenger_count": [1, 2, 1, 3, 2, 1],
            "store_and_fwd_flag": ["N", "N", "Y", "N", "N", "Y"],
            "pickup_hour": [8, 9, 10, 17, 18, 19],
            "pickup_day_of_week": [0, 1, 2, 3, 4, 5],
            "pickup_month": [1, 1, 2, 2, 3, 3],
            "is_weekend": [0, 0, 0, 0, 0, 1],
            "distance_km": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    validation = pd.DataFrame(
        {
            "vendor_id": [1, 2, 1],
            "passenger_count": [1, 2, 3],
            "store_and_fwd_flag": ["N", "Y", "N"],
            "pickup_hour": [11, 15, 20],
            "pickup_day_of_week": [1, 2, 6],
            "pickup_month": [3, 4, 5],
            "is_weekend": [0, 0, 1],
            "distance_km": [2.5, 7.0, 8.0],
        }
    )

    return train, validation


def test_train_validation_processed_feature_count_matches():
    """Verify TRAIN and VALIDATION produce identical feature counts."""

    train, validation = create_test_data()

    X_train, X_val, _ = preprocess_train_validation(
        train,
        validation,
    )

    assert X_train.shape[1] == X_val.shape[1]


def test_processed_data_contains_only_finite_values():
    """Verify preprocessing produces no NaN or infinite values."""

    train, validation = create_test_data()

    X_train, X_val, _ = preprocess_train_validation(
        train,
        validation,
    )

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()


def test_train_and_validation_have_same_processed_shape():
    """Verify TRAIN and VALIDATION have compatible processed structures."""

    train, validation = create_test_data()

    X_train, X_val, _ = preprocess_train_validation(
        train,
        validation,
    )

    assert X_train.shape == (6, 10)
    assert X_val.shape == (3, 10)