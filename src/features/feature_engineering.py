"""Reusable feature engineering module for the NYC Taxi project.

This module contains finalized feature creation logic from the Phase 3C
datetime and geographic feature-engineering scripts.

Created features:
- pickup_hour
- pickup_day_of_week
- pickup_month
- is_weekend
- distance_km

All features are derived from prediction-time inputs only.
"""
# Created using the finalized logic from:
# 1. scripts/5. Feature-Engineering/5.3a-Create-Datetime-Features.py
# 2. scripts/5. Feature-Engineering/5.4b-Create-Haversine-Distance.py

from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

TRAIN_INPUT = INTERIM_DIR / "train_clean.csv"
TEST_INPUT = INTERIM_DIR / "test_clean.csv"


# ---------------------------------------------------------------------------
# Datetime feature engineering
# ---------------------------------------------------------------------------

def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from pickup_datetime."""

    logger.info("Creating datetime features")

    df = df.copy()

    if "pickup_datetime" not in df.columns:
        logger.error("Required column 'pickup_datetime' not found")
        raise KeyError("Required column 'pickup_datetime' not found.")

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce",
    )

    invalid_count = int(df["pickup_datetime"].isna().sum())

    if invalid_count > 0:
        logger.error(
            "Datetime conversion produced %d invalid values",
            invalid_count,
        )
        raise ValueError(
            "Invalid pickup_datetime values found during feature engineering."
        )

    df["pickup_hour"] = df["pickup_datetime"].dt.hour

    df["pickup_day_of_week"] = (
        df["pickup_datetime"].dt.dayofweek
    )

    df["pickup_month"] = (
        df["pickup_datetime"].dt.month
    )

    df["is_weekend"] = (
        df["pickup_day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    logger.info(
        "Datetime features created successfully for %d records",
        len(df),
    )

    return df


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """Calculate great-circle distance between two geographic points.

    Returns distance in kilometers using an Earth radius of 6371 km.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0

    return earth_radius_km * c


def create_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create the distance_km feature from pickup/dropoff coordinates."""

    logger.info("Creating distance_km feature")

    df = df.copy()

    required_columns = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            "Required geographic columns missing: %s",
            missing_columns,
        )
        raise KeyError(
            f"Required geographic columns missing: {missing_columns}"
        )

    df["distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    logger.info(
        "distance_km created successfully for %d records",
        len(df),
    )

    return df


# ---------------------------------------------------------------------------
# Complete feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all finalized Phase 3C features for one dataset."""

    logger.info(
        "Starting feature engineering for %d records",
        len(df),
    )

    df = create_datetime_features(df)
    df = create_distance_feature(df)

    logger.info(
        "Feature engineering completed successfully for %d records",
        len(df),
    )

    return df


def engineer_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply identical feature engineering to TRAIN and TEST."""

    logger.info("Starting feature engineering for TRAIN and TEST")

    train_engineered = engineer_features(train)

    logger.info("TRAIN feature engineering completed")

    test_engineered = engineer_features(test)

    logger.info("TEST feature engineering completed")

    return train_engineered, test_engineered


# ---------------------------------------------------------------------------
# File-based convenience function
# ---------------------------------------------------------------------------

def load_and_engineer(
    train_path: Path = TRAIN_INPUT,
    test_path: Path = TEST_INPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned TRAIN/TEST datasets and return engineered datasets."""

    logger.info(
        "Loading cleaned TRAIN dataset: %s",
        train_path,
    )

    train = pd.read_csv(train_path)

    logger.info(
        "Loading cleaned TEST dataset: %s",
        test_path,
    )

    test = pd.read_csv(test_path)

    logger.info(
        "Datasets loaded successfully - TRAIN: %d, TEST: %d",
        len(train),
        len(test),
    )

    return engineer_train_test(train, test)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run feature engineering using the project's default cleaned datasets."""

    logger.info("PHASE 3C - FEATURE ENGINEERING STARTED")

    try:
        train, test = load_and_engineer()

        logger.info(
            "Feature engineering successful - "
            "TRAIN shape: %s, TEST shape: %s",
            train.shape,
            test.shape,
        )

        logger.info(
            "Created features: "
            "pickup_hour, pickup_day_of_week, "
            "pickup_month, is_weekend, distance_km"
        )

        logger.info("PHASE 3C - FEATURE ENGINEERING COMPLETED")

    except Exception:
        logger.exception("Feature engineering failed")
        raise


if __name__ == "__main__":
    main()