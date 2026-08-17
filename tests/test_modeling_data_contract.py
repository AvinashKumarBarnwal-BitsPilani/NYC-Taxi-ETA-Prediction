"""Tests for the Phase 4 modeling strategy and data contract."""

import pandas as pd
import pytest

from src.training.data_contract import (
    load_modeling_config,
    validate_data_contract,
    validate_modeling_config,
)


EXPECTED_FEATURES = [
    "numerical__distance_km",
    "categorical__vendor_id_1.0",
    "categorical__vendor_id_2.0",
    "categorical__store_and_fwd_flag_N",
    "categorical__store_and_fwd_flag_Y",
]


def test_modeling_config_is_valid():
    """Verify the Phase 4 modeling configuration."""

    config = load_modeling_config()

    validate_modeling_config(config)


def test_valid_data_contract():
    """Verify that valid Phase 4 datasets pass validation."""

    X_train = pd.DataFrame(
        {
            feature: [0.1, 0.2]
            for feature in EXPECTED_FEATURES
        }
    )

    X_val = X_train.copy()

    y_train = pd.DataFrame(
        {"trip_duration": [100, 200]}
    )

    y_val = pd.DataFrame(
        {"trip_duration": [150, 250]}
    )

    config = {
        "problem": {
            "type": "regression",
            "target": "trip_duration",
        },
        "features": EXPECTED_FEATURES,
        "metrics": {
            "primary": "rmse",
            "secondary": "mae",
            "context": "r2",
        },
        "data": {
            "train_features": "data/processed/X_train_processed.csv",
            "validation_features": "data/processed/X_val_processed.csv",
            "train_target": "data/split/y_train.csv",
            "validation_target": "data/split/y_val.csv",
        },
    }

    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )


def test_wrong_feature_order_fails():
    """Verify that feature-order changes are detected."""

    X_train = pd.DataFrame(
        {
            feature: [0.1, 0.2]
            for feature in EXPECTED_FEATURES[::-1]
        }
    )

    X_val = X_train.copy()

    y_train = pd.DataFrame(
        {"trip_duration": [100, 200]}
    )

    y_val = pd.DataFrame(
        {"trip_duration": [150, 250]}
    )

    config = {
        "problem": {
            "type": "regression",
            "target": "trip_duration",
        },
        "features": EXPECTED_FEATURES,
        "metrics": {
            "primary": "rmse",
            "secondary": "mae",
            "context": "r2",
        },
        "data": {
            "train_features": "data/processed/X_train_processed.csv",
            "validation_features": "data/processed/X_val_processed.csv",
            "train_target": "data/split/y_train.csv",
            "validation_target": "data/split/y_val.csv",
        },
    }

    with pytest.raises(ValueError, match="feature columns"):
        validate_data_contract(
            X_train,
            X_val,
            y_train,
            y_val,
            config,
        )


def test_row_count_mismatch_fails():
    """Verify that X/y row-count mismatches are detected."""

    X_train = pd.DataFrame(
        {
            feature: [0.1, 0.2]
            for feature in EXPECTED_FEATURES
        }
    )

    X_val = X_train.copy()

    y_train = pd.DataFrame(
        {"trip_duration": [100]}
    )

    y_val = pd.DataFrame(
        {"trip_duration": [150, 250]}
    )

    config = {
        "problem": {
            "type": "regression",
            "target": "trip_duration",
        },
        "features": EXPECTED_FEATURES,
        "metrics": {
            "primary": "rmse",
            "secondary": "mae",
            "context": "r2",
        },
        "data": {
            "train_features": "data/processed/X_train_processed.csv",
            "validation_features": "data/processed/X_val_processed.csv",
            "train_target": "data/split/y_train.csv",
            "validation_target": "data/split/y_val.csv",
        },
    }

    with pytest.raises(ValueError, match="row counts"):
        validate_data_contract(
            X_train,
            X_val,
            y_train,
            y_val,
            config,
        )