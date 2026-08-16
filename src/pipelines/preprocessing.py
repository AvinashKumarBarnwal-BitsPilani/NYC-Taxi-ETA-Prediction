"""Reusable preprocessing module for the NYC Taxi project.

This module contains the finalized preprocessing decisions from Phase 3C:
- Scale distance_km using StandardScaler.
- One-hot encode vendor_id and store_and_fwd_flag.
- Fit the preprocessor on TRAIN data only.
- Transform VALIDATION using the fitted TRAIN preprocessor.

The module is designed to be reusable by the pipeline layer.

Created using the finalized logic from:
1. scripts/6. Train-Validation-Split/6.3-Analyze-Feature-Scaling.py
2. scripts/6. Train-Validation-Split/6.4-Analyze-Categorical-Features.py
3. scripts/6. Train-Validation-Split/6.5-Verify-Preprocessed-Datasets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

import joblib

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "split"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_VAL_PATH = SPLIT_DIR / "X_val.csv"


# ---------------------------------------------------------------------------
# Final preprocessing feature groups
# ---------------------------------------------------------------------------

NUMERICAL_FEATURES = [
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "vendor_id",
    "store_and_fwd_flag",
]

# ---------------------------------------------------------------------------
# Save Preprocessor
# ---------------------------------------------------------------------------
def save_preprocessor(
    preprocessor: ColumnTransformer,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Save the fitted TRAIN preprocessor for reuse during inference."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessor_path = output_dir / "preprocessor.joblib"

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    logger.info(
        "Fitted preprocessor saved: %s",
        preprocessor_path,
    )

# ---------------------------------------------------------------------------
# Preprocessor construction
# ---------------------------------------------------------------------------

def create_preprocessor() -> ColumnTransformer:
    """Create the finalized preprocessing transformer.

    The transformer:
    - standardizes distance_km
    - one-hot encodes vendor_id and store_and_fwd_flag
    - ignores unseen categorical values during transformation
    """

    logger.info("Creating preprocessing transformer")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    logger.info(
        "Preprocessor configured - numerical=%s, categorical=%s",
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_preprocessing_input(
    X: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Validate that required preprocessing columns are available."""

    required_columns = set(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    missing_columns = sorted(
        required_columns - set(X.columns)
    )

    if missing_columns:
        logger.error(
            "%s is missing required preprocessing columns: %s",
            dataset_name,
            missing_columns,
        )
        raise KeyError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )

    if X.empty:
        logger.error("%s is empty", dataset_name)
        raise ValueError(f"{dataset_name} cannot be empty.")


# ---------------------------------------------------------------------------
# Fit on TRAIN and transform TRAIN
# ---------------------------------------------------------------------------

def fit_transform_train(
    X_train: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Fit the preprocessor using TRAIN only and transform TRAIN."""

    logger.info(
        "Fitting preprocessor on TRAIN data: %d records",
        len(X_train),
    )

    validate_preprocessing_input(X_train, "X_train")

    if preprocessor is None:
        preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    logger.info(
        "TRAIN preprocessing completed - shape=%s",
        X_train_processed_df.shape,
    )

    return X_train_processed_df, preprocessor


# ---------------------------------------------------------------------------
# Transform VALIDATION
# ---------------------------------------------------------------------------

def transform_validation(
    X_val: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    """Transform VALIDATION using a preprocessor already fitted on TRAIN."""

    logger.info(
        "Transforming VALIDATION data using TRAIN-fitted preprocessor: %d records",
        len(X_val),
    )

    validate_preprocessing_input(X_val, "X_val")

    if not hasattr(preprocessor, "transformers_"):
        logger.error(
            "Preprocessor has not been fitted. "
            "Fit it on TRAIN before transforming VALIDATION."
        )
        raise ValueError(
            "Preprocessor must be fitted on TRAIN before transforming "
            "VALIDATION."
        )

    X_val_processed = preprocessor.transform(X_val)

    feature_names = preprocessor.get_feature_names_out()

    X_val_processed_df = pd.DataFrame(
        X_val_processed,
        columns=feature_names,
        index=X_val.index,
    )

    logger.info(
        "VALIDATION preprocessing completed - shape=%s",
        X_val_processed_df.shape,
    )

    return X_val_processed_df


# ---------------------------------------------------------------------------
# Complete TRAIN / VALIDATION preprocessing
# ---------------------------------------------------------------------------

def preprocess_train_validation(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessing on TRAIN and transform TRAIN and VALIDATION.

    This function enforces the project's anti-leakage rule:
    the preprocessor is fitted only on TRAIN.
    """

    logger.info(
        "Starting TRAIN / VALIDATION preprocessing"
    )

    X_train_processed, preprocessor = fit_transform_train(
        X_train
    )

    X_val_processed = transform_validation(
        X_val,
        preprocessor,
    )

    if list(X_train_processed.columns) != list(
        X_val_processed.columns
    ):
        logger.error(
            "TRAIN and VALIDATION processed feature columns do not match"
        )
        raise ValueError(
            "TRAIN and VALIDATION processed feature columns do not match."
        )

    logger.info(
        "TRAIN / VALIDATION preprocessing completed successfully"
    )

    return (
        X_train_processed,
        X_val_processed,
        preprocessor,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_preprocessed_datasets(
    X_train_processed: pd.DataFrame,
    X_val_processed: pd.DataFrame,
) -> None:
    """Verify processed datasets for basic ML-readiness."""

    logger.info("Verifying preprocessed datasets")

    train_missing = int(
        X_train_processed.isna().sum().sum()
    )
    val_missing = int(
        X_val_processed.isna().sum().sum()
    )

    train_infinite = int(
        np.isinf(
            X_train_processed.to_numpy()
        ).sum()
    )
    val_infinite = int(
        np.isinf(
            X_val_processed.to_numpy()
        ).sum()
    )

    same_features = (
        list(X_train_processed.columns)
        == list(X_val_processed.columns)
    )

    if train_missing or val_missing:
        logger.error(
            "Missing values detected - TRAIN=%d, VALIDATION=%d",
            train_missing,
            val_missing,
        )
        raise ValueError(
            "Missing values detected in preprocessed datasets."
        )

    if train_infinite or val_infinite:
        logger.error(
            "Infinite values detected - TRAIN=%d, VALIDATION=%d",
            train_infinite,
            val_infinite,
        )
        raise ValueError(
            "Infinite values detected in preprocessed datasets."
        )

    if not same_features:
        logger.error(
            "TRAIN and VALIDATION processed feature columns do not match"
        )
        raise ValueError(
            "TRAIN and VALIDATION processed feature columns do not match."
        )

    logger.info(
        "Preprocessed datasets verified successfully - "
        "features=%d, TRAIN=%d, VALIDATION=%d",
        X_train_processed.shape[1],
        len(X_train_processed),
        len(X_val_processed),
    )


# ---------------------------------------------------------------------------
# File-based convenience function
# ---------------------------------------------------------------------------

def load_and_preprocess(
    x_train_path: Path = X_TRAIN_PATH,
    x_val_path: Path = X_VAL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Load split datasets and preprocess TRAIN / VALIDATION."""

    logger.info("Loading X_train: %s", x_train_path)
    X_train = pd.read_csv(x_train_path)

    logger.info("Loading X_val: %s", x_val_path)
    X_val = pd.read_csv(x_val_path)

    logger.info(
        "Split datasets loaded - X_train=%d, X_val=%d",
        len(X_train),
        len(X_val),
    )

    (
        X_train_processed,
        X_val_processed,
        preprocessor,
    ) = preprocess_train_validation(
        X_train,
        X_val,
    )

    verify_preprocessed_datasets(
        X_train_processed,
        X_val_processed,
    )

    return (
        X_train_processed,
        X_val_processed,
        preprocessor,
    )


# ---------------------------------------------------------------------------
# Optional persistence of processed datasets
# ---------------------------------------------------------------------------

def save_processed_datasets(
    X_train_processed: pd.DataFrame,
    X_val_processed: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Save processed TRAIN and VALIDATION features."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_train_path = output_dir / "X_train_processed.csv"
    x_val_path = output_dir / "X_val_processed.csv"

    X_train_processed.to_csv(
        x_train_path,
        index=False,
    )

    X_val_processed.to_csv(
        x_val_path,
        index=False,
    )

    logger.info(
        "Processed X_train saved: %s",
        x_train_path,
    )

    logger.info(
        "Processed X_val saved: %s",
        x_val_path,
    )


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run preprocessing using the project's default split datasets."""

    logger.info(
        "PREPROCESSING PIPELINE STARTED"
    )

    try:
        (
            X_train_processed,
            X_val_processed,
            _preprocessor,
        ) = load_and_preprocess()

        save_processed_datasets(
            X_train_processed,
            X_val_processed,
        )

        logger.info(
            "Final processed shapes - "
            "X_train=%s, X_val=%s",
            X_train_processed.shape,
            X_val_processed.shape,
        )

        logger.info(
            "PREPROCESSING PIPELINE COMPLETED"
        )

    except Exception:
        logger.exception(
            "PREPROCESSING PIPELINE FAILED"
        )
        raise


if __name__ == "__main__":
    main()
