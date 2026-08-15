"""Reusable chronological train/validation split module.

This module contains the finalized train/validation split logic from Phase 3C.

The split is chronological so that earlier observations are used for training
and the latest observations are reserved for validation, simulating
future-trip prediction.

The module is designed to be reusable by the pipeline layer and can also be
executed directly using the project's default cleaned TRAIN dataset.
"""

# Created using the finalized logic from:
# scripts/6. Train-Validation-Split/6.2-Implement-Train-Validation-Split.py

from pathlib import Path

import pandas as pd

from src.features.feature_engineering import engineer_features
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
SPLIT_DIR = PROJECT_ROOT / "data" / "split"

TRAIN_INPUT = INTERIM_DIR / "train_clean.csv"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_VAL_PATH = SPLIT_DIR / "X_val.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
Y_VAL_PATH = SPLIT_DIR / "y_val.csv"


# ---------------------------------------------------------------------------
# Final candidate features from Phase 3C
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "distance_km",
]

TARGET_COLUMN = "trip_duration"
DATETIME_COLUMN = "pickup_datetime"
VALIDATION_SIZE = 0.20


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_train_validation(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    datetime_column: str = DATETIME_COLUMN,
    validation_size: float = VALIDATION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a chronological train/validation split.

    Earlier observations are assigned to training and the latest observations
    are assigned to validation.

    Returns:
        X_train, X_val, y_train, y_val
    """

    logger.info(
        "Starting chronological train/validation split for %d records",
        len(df),
    )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be greater than 0 and less than 1."
        )

    required_columns = set(feature_columns) | {
        target_column,
        datetime_column,
    }

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        logger.error(
            "Required columns missing for train/validation split: %s",
            missing_columns,
        )
        raise KeyError(
            f"Required columns missing: {missing_columns}"
        )

    split_df = df.copy()

    split_df[datetime_column] = pd.to_datetime(
        split_df[datetime_column],
        errors="coerce",
    )

    invalid_datetime_count = int(
        split_df[datetime_column].isna().sum()
    )

    if invalid_datetime_count > 0:
        logger.error(
            "Found %d invalid %s values",
            invalid_datetime_count,
            datetime_column,
        )
        raise ValueError(
            f"Invalid {datetime_column} values found."
        )

    split_df = split_df.sort_values(
        datetime_column
    ).reset_index(drop=True)

    split_index = int(
        len(split_df) * (1 - validation_size)
    )

    if split_index <= 0 or split_index >= len(split_df):
        raise ValueError(
            "Calculated split index does not produce both "
            "training and validation datasets."
        )

    train_part = split_df.iloc[:split_index].copy()
    validation_part = split_df.iloc[split_index:].copy()

    X_train = train_part[feature_columns].copy()
    y_train = train_part[target_column].copy()

    X_val = validation_part[feature_columns].copy()
    y_val = validation_part[target_column].copy()

    training_max = train_part[datetime_column].max()
    validation_min = validation_part[datetime_column].min()

    if not training_max < validation_min:
        logger.error(
            "Chronological ordering check failed: "
            "training_max=%s, validation_min=%s",
            training_max,
            validation_min,
        )
        raise ValueError(
            "Training data must end before validation data begins."
        )

    logger.info(
        "Chronological split completed - total=%d, train=%d, validation=%d",
        len(split_df),
        len(X_train),
        len(X_val),
    )

    logger.info(
        "Split proportions - train=%.4f, validation=%.4f",
        len(X_train) / len(split_df),
        len(X_val) / len(split_df),
    )

    logger.info(
        "Training date range: %s to %s",
        train_part[datetime_column].min(),
        training_max,
    )

    logger.info(
        "Validation date range: %s to %s",
        validation_min,
        validation_part[datetime_column].max(),
    )

    logger.info(
        "Feature shapes - X_train=%s, X_val=%s, y_train=%s, y_val=%s",
        X_train.shape,
        X_val.shape,
        y_train.shape,
        y_val.shape,
    )

    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Save split datasets
# ---------------------------------------------------------------------------

def save_split_datasets(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    output_dir: Path = SPLIT_DIR,
) -> None:
    """Save train/validation datasets to the configured split directory."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_train_path = output_dir / "X_train.csv"
    x_val_path = output_dir / "X_val.csv"
    y_train_path = output_dir / "y_train.csv"
    y_val_path = output_dir / "y_val.csv"

    logger.info("Saving train/validation datasets")

    X_train.to_csv(x_train_path, index=False)
    X_val.to_csv(x_val_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_val.to_csv(y_val_path, index=False)

    logger.info("X_train saved: %s", x_train_path)
    logger.info("X_val saved: %s", x_val_path)
    logger.info("y_train saved: %s", y_train_path)
    logger.info("y_val saved: %s", y_val_path)


# ---------------------------------------------------------------------------
# File-based pipeline convenience function
# ---------------------------------------------------------------------------

def load_and_split(
    train_path: Path = TRAIN_INPUT,
    output_dir: Path = SPLIT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load cleaned TRAIN data, engineer features, split, and save datasets."""

    logger.info("Loading cleaned TRAIN dataset: %s", train_path)

    train_df = pd.read_csv(train_path)

    logger.info(
        "Cleaned TRAIN dataset loaded: %d records",
        len(train_df),
    )

    logger.info(
        "Applying finalized feature engineering before the split"
    )

    train_engineered = engineer_features(train_df)

    X_train, X_val, y_train, y_val = split_train_validation(
        train_engineered
    )

    save_split_datasets(
        X_train,
        X_val,
        y_train,
        y_val,
        output_dir,
    )

    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the train/validation split using project default paths."""

    logger.info(
        "TRAIN / VALIDATION SPLIT PIPELINE STARTED"
    )

    try:
        X_train, X_val, y_train, y_val = load_and_split()

        logger.info(
            "TRAIN / VALIDATION SPLIT PIPELINE COMPLETED"
        )

        logger.info(
            "Final shapes - X_train=%s, X_val=%s, y_train=%s, y_val=%s",
            X_train.shape,
            X_val.shape,
            y_train.shape,
            y_val.shape,
        )

    except Exception:
        logger.exception(
            "TRAIN / VALIDATION SPLIT PIPELINE FAILED"
        )
        raise


if __name__ == "__main__":
    main()
