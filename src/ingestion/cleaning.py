"""Reusable data cleaning module for the NYC Taxi Trip Duration project."""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

TRAIN_INPUT = RAW_DIR / "train.csv"
TEST_INPUT = RAW_DIR / "test.csv"

TRAIN_OUTPUT = INTERIM_DIR / "train_clean.csv"
TEST_OUTPUT = INTERIM_DIR / "test_clean.csv"


COMMON_DTYPES = {
    "id": "string",
    "vendor_id": "Int64",
    "passenger_count": "Int64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "store_and_fwd_flag": "string",
}

TRAIN_DTYPES = {
    **COMMON_DTYPES,
    "trip_duration": "Int64",
}


def load_data(
    train_path: Path = TRAIN_INPUT,
    test_path: Path = TEST_INPUT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw TRAIN and TEST datasets with explicit data types."""

    train = pd.read_csv(
        train_path,
        dtype=TRAIN_DTYPES,
        parse_dates=["pickup_datetime", "dropoff_datetime"],
    )

    test = pd.read_csv(
        test_path,
        dtype=COMMON_DTYPES,
        parse_dates=["pickup_datetime"],
    )

    return train, test


def clean_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    has_target: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the finalized Phase 3B cleaning rules and return audit details."""

    original_count = len(df)

    audit = {
        "original_records": original_count,
        "missing_value_removals": 0,
        "duplicate_removals": 0,
        "invalid_passenger_removals": 0,
        "invalid_coordinate_removals": 0,
        "invalid_timestamp_removals": 0,
        "invalid_target_removals": 0,
        "extreme_outlier_removals": 0,
    }

    missing_mask = df.isnull().any(axis=1)
    audit["missing_value_removals"] = int(missing_mask.sum())
    df = df.loc[~missing_mask].copy()

    duplicate_mask = df.duplicated()
    audit["duplicate_removals"] = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    duplicate_id_mask = df["id"].duplicated(keep="first")
    duplicate_id_count = int(duplicate_id_mask.sum())
    audit["duplicate_removals"] += duplicate_id_count
    df = df.loc[~duplicate_id_mask].copy()

    invalid_passenger_mask = df["passenger_count"] <= 0
    audit["invalid_passenger_removals"] = int(invalid_passenger_mask.sum())
    df = df.loc[~invalid_passenger_mask].copy()

    invalid_coordinate_mask = (
        ~df["pickup_longitude"].between(-75, -72)
        | ~df["pickup_latitude"].between(40, 42)
        | ~df["dropoff_longitude"].between(-75, -72)
        | ~df["dropoff_latitude"].between(40, 42)
    )
    audit["invalid_coordinate_removals"] = int(
        invalid_coordinate_mask.sum()
    )
    df = df.loc[~invalid_coordinate_mask].copy()

    if has_target:
        invalid_timestamp_mask = (
            df["pickup_datetime"] >= df["dropoff_datetime"]
        )
        audit["invalid_timestamp_removals"] = int(
            invalid_timestamp_mask.sum()
        )
        df = df.loc[~invalid_timestamp_mask].copy()

        calculated_duration = (
            df["dropoff_datetime"] - df["pickup_datetime"]
        ).dt.total_seconds()

        invalid_target_mask = (
            (df["trip_duration"] <= 0)
            | (df["trip_duration"] != calculated_duration)
        )
        audit["invalid_target_removals"] = int(invalid_target_mask.sum())
        df = df.loc[~invalid_target_mask].copy()

        extreme_outlier_mask = df["trip_duration"] > 86400
        audit["extreme_outlier_removals"] = int(
            extreme_outlier_mask.sum()
        )
        df = df.loc[~extreme_outlier_mask].copy()

    audit["final_records"] = len(df)
    audit["total_removed"] = original_count - len(df)

    return df, audit


def log_summary(dataset_name: str, audit: dict[str, int]) -> None:
    """Log a cleaning audit summary."""

    logger.info("%s CLEANING SUMMARY", dataset_name)

    for key, value in audit.items():
        logger.info(
            "%s: %s",
            key.replace("_", " ").title(),
            value,
        )


def clean_raw_data(
    train_path: Path = TRAIN_INPUT,
    test_path: Path = TEST_INPUT,
    train_output: Path = TRAIN_OUTPUT,
    test_output: Path = TEST_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    """Load, clean, save, and return cleaned TRAIN and TEST datasets."""

    train, test = load_data(train_path, test_path)

    train_clean, train_audit = clean_dataset(
        train,
        dataset_name="TRAIN",
        has_target=True,
    )

    test_clean, test_audit = clean_dataset(
        test,
        dataset_name="TEST",
        has_target=False,
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)

    train_clean.to_csv(train_output, index=False)
    test_clean.to_csv(test_output, index=False)

    return train_clean, test_clean, train_audit, test_audit


def main() -> None:
    """Run cleaning using the project's default paths."""

    logger.info("PHASE 3B - DATA CLEANING PIPELINE STARTED")

    try:
        train_clean, test_clean, train_audit, test_audit = clean_raw_data()

        log_summary("TRAIN", train_audit)
        log_summary("TEST", test_audit)

        logger.info("CLEANING COMPLETED")
        logger.info("Clean TRAIN dataset: %s", TRAIN_OUTPUT)
        logger.info("Clean TEST dataset: %s", TEST_OUTPUT)
        logger.info("TRAIN records: %d", len(train_clean))
        logger.info("TEST records: %d", len(test_clean))

        logger.info("PHASE 3B - DATA CLEANING PIPELINE COMPLETED")

    except Exception:
        logger.exception("Data cleaning failed")
        raise


if __name__ == "__main__":
    main()
