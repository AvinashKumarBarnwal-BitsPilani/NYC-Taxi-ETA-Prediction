# Create a realistic rush-hour drift scenario by modifying a copy of normal
# production-like data: concentrate pickup hours around peak periods and
# shift distance_km toward longer trips, while preserving the original data.

"""Simulate realistic production feature drift."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# prediction_logs.csv (In drift_detection.py) → actual production-like input
# normal_production_data.csv → simulation baseline/normal scenario
# drifted_production_data.csv → simulation of changed production distribution

SOURCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "normal_production_data.csv"
)

DRIFTED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "drifted_production_data.csv"
)

REQUIRED_FEATURES = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "distance_km",
]

RUSH_HOURS = [
    7, 8, 9,
    17, 18, 19,
]

RUSH_HOUR_RATIO = 0.60

DISTANCE_SHIFT_FACTOR = 1.35


def validate_source_data(df: pd.DataFrame) -> None:
    """Validate required features exist."""

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )


def simulate_rush_hour(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a realistic rush-hour traffic surge."""

    drifted = df.copy()

    rng = np.random.default_rng(42)

    rush_hour_count = int(
        len(drifted) * RUSH_HOUR_RATIO
    )

    rush_hour_indices = rng.choice(
        drifted.index,
        size=rush_hour_count,
        replace=False,
    )

    drifted.loc[
        rush_hour_indices,
        "pickup_hour",
    ] = rng.choice(
        RUSH_HOURS,
        size=rush_hour_count,
    )

    drifted.loc[
        rush_hour_indices,
        "distance_km",
    ] = (
        drifted.loc[
            rush_hour_indices,
            "distance_km",
        ]
        * DISTANCE_SHIFT_FACTOR
    )

    return drifted


def generate_drifted_dataset() -> pd.DataFrame:
    """Generate and persist the simulated drift dataset."""

    logger.info(
        "Loading normal production-like data: %s",
        SOURCE_DATA_PATH,
    )

    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Source dataset not found: "
            f"{SOURCE_DATA_PATH}"
        )

    df = pd.read_csv(SOURCE_DATA_PATH)

    logger.info(
        "Source dataset shape: %s",
        df.shape,
    )

    if df.empty:
        raise ValueError(
            "Source dataset cannot be empty."
        )

    validate_source_data(df)

    drifted_data = simulate_rush_hour(df)

    DRIFTED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drifted_data.to_csv(
        DRIFTED_DATA_PATH,
        index=False,
    )

    logger.info(
        "Drifted dataset saved: %s",
        DRIFTED_DATA_PATH,
    )

    logger.info(
        "Drifted dataset shape: %s",
        drifted_data.shape,
    )

    return drifted_data


def main() -> None:
    """Run drift simulation."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    logger.info(
        "DRIFT SIMULATION STARTED"
    )

    generate_drifted_dataset()

    logger.info(
        "DRIFT SIMULATION COMPLETED"
    )


if __name__ == "__main__":
    main()