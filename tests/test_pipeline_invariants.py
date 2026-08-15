"""
7.3c – Test Data & Pipeline Invariants

These tests verify important properties that must remain true in the
Phase 3 data-engineering pipeline.

The tests use the generated train/validation and processed datasets to
verify:

- Record conservation
- Chronological ordering
- TRAIN / VALIDATION feature compatibility
- Absence of missing values
- Absence of infinite values

These are pipeline invariants rather than implementation-specific tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "split"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_VAL_PATH = SPLIT_DIR / "X_val.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
Y_VAL_PATH = SPLIT_DIR / "y_val.csv"

X_TRAIN_PROCESSED_PATH = (
    PROCESSED_DIR / "X_train_processed.csv"
)

X_VAL_PROCESSED_PATH = (
    PROCESSED_DIR / "X_val_processed.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split_datasets():
    """Load generated train/validation split datasets."""

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_val = pd.read_csv(X_VAL_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_val = pd.read_csv(Y_VAL_PATH)

    return X_train, X_val, y_train, y_val


def load_processed_datasets():
    """Load generated preprocessed train/validation datasets."""

    X_train_processed = pd.read_csv(
        X_TRAIN_PROCESSED_PATH
    )

    X_val_processed = pd.read_csv(
        X_VAL_PROCESSED_PATH
    )

    return X_train_processed, X_val_processed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_record_conservation():
    """
    Verify that TRAIN + VALIDATION records equal the original
    cleaned TRAIN dataset size.
    """

    X_train, X_val, y_train, y_val = load_split_datasets()

    total_records = len(X_train) + len(X_val)

    assert total_records == 1_458_542

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


def test_chronological_ordering():
    """
    Verify that the train/validation split preserves chronological ordering.
    """

    from src.pipelines.train_validation_split import (
        split_train_validation,
    )

    data = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                [
                    "2016-01-01 08:00:00",
                    "2016-01-02 09:00:00",
                    "2016-01-03 10:00:00",
                    "2016-01-04 11:00:00",
                    "2016-01-05 12:00:00",
                    "2016-01-06 13:00:00",
                    "2016-01-07 14:00:00",
                    "2016-01-08 15:00:00",
                    "2016-01-09 16:00:00",
                    "2016-01-10 17:00:00",
                ]
            ),
            "vendor_id": [1, 1, 2, 2, 1, 1, 2, 2, 1, 2],
            "passenger_count": [1] * 10,
            "store_and_fwd_flag": ["N"] * 10,
            "pickup_hour": list(range(10)),
            "pickup_day_of_week": list(range(7)) + [0, 1, 2],
            "pickup_month": [1] * 10,
            "is_weekend": [0] * 10,
            "distance_km": [1.0] * 10,
            "trip_duration": [300] * 10,
        }
    )

    X_train, X_val, y_train, y_val = split_train_validation(data)

    train_max = data.loc[
        X_train.index,
        "pickup_datetime",
    ].max()

    validation_min = data.loc[
        X_val.index,
        "pickup_datetime",
    ].min()

    assert train_max < validation_min


def test_train_validation_feature_compatibility():
    """
    Verify that TRAIN and VALIDATION contain identical feature columns.
    """

    X_train, X_val, _, _ = load_split_datasets()

    assert list(X_train.columns) == list(X_val.columns)
    assert X_train.shape[1] == X_val.shape[1]


def test_no_missing_values_in_processed_data():
    """
    Verify that processed TRAIN and VALIDATION datasets contain
    no missing values.
    """

    X_train_processed, X_val_processed = (
        load_processed_datasets()
    )

    assert not X_train_processed.isnull().values.any()
    assert not X_val_processed.isnull().values.any()


def test_no_infinite_values_in_processed_data():
    """
    Verify that processed TRAIN and VALIDATION datasets contain
    no infinite values.
    """

    X_train_processed, X_val_processed = (
        load_processed_datasets()
    )

    assert np.isfinite(
        X_train_processed.to_numpy()
    ).all()

    assert np.isfinite(
        X_val_processed.to_numpy()
    ).all()