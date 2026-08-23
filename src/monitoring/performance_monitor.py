"""Utilities for preparing prediction data for performance monitoring."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.inference.prediction_pipeline import PredictionPipeline

import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "split"
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"

X_VAL_PATH = SPLIT_DIR / "X_val.csv"
Y_VAL_PATH = SPLIT_DIR / "y_val.csv"

PERFORMANCE_DATASET_PATH = (
    MONITORING_DIR / "performance_dataset.csv"
)

PERFORMANCE_REPORT_PATH = (
    MONITORING_DIR / "performance_report.json"
)

BASELINE_REPORT_PATH = (
    MONITORING_DIR / "baseline_metrics.json"
)

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

def prepare_performance_dataset() -> pd.DataFrame:
    """Generate predictions for validation data and pair them with actual ETA."""

    logger.info(
        "Preparing performance monitoring dataset"
    )

    logger.info(
        "Loading validation features: %s",
        X_VAL_PATH,
    )

    X_val = pd.read_csv(X_VAL_PATH)

    logger.info(
        "Loading validation ground truth: %s",
        Y_VAL_PATH,
    )

    y_val = pd.read_csv(Y_VAL_PATH)

    if len(X_val) != len(y_val):
        raise ValueError(
            "Validation feature and target row counts do not match."
        )

    missing_features = sorted(
        set(FEATURE_COLUMNS) - set(X_val.columns)
    )

    if missing_features:
        raise ValueError(
            "Validation data is missing required features: "
            f"{missing_features}"
        )

    if "trip_duration" not in y_val.columns:
        raise ValueError(
            "Validation target must contain 'trip_duration'."
        )

    X_val = X_val[FEATURE_COLUMNS].copy()

    logger.info(
        "Generating predictions for %d validation records",
        len(X_val),
    )

    prediction_pipeline = PredictionPipeline()
    #predictions = prediction_pipeline.predict(X_val)
    predictions = prediction_pipeline.predict_batch(X_val)
    performance_dataset = X_val.copy()

    performance_dataset["predicted_eta"] = predictions
    performance_dataset["actual_eta"] = y_val[
        "trip_duration"
    ].to_numpy()

    if performance_dataset[
        ["predicted_eta", "actual_eta"]
    ].isna().any().any():
        raise ValueError(
            "Performance dataset contains missing prediction "
            "or ground-truth values."
        )

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance_dataset.to_csv(
        PERFORMANCE_DATASET_PATH,
        index=False,
    )

    logger.info(
        "Performance monitoring dataset saved: %s",
        PERFORMANCE_DATASET_PATH,
    )

    logger.info(
        "Performance dataset shape: %s",
        performance_dataset.shape,
    )

    return performance_dataset


def calculate_performance_metrics(
    performance_dataset: pd.DataFrame,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and R² for a monitoring dataset."""

    required_columns = {
        "predicted_eta",
        "actual_eta",
    }

    missing_columns = sorted(
        required_columns - set(performance_dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Performance dataset is missing required columns: "
            f"{missing_columns}"
        )

    if performance_dataset.empty:
        raise ValueError(
            "Performance dataset cannot be empty."
        )

    predictions = performance_dataset[
        "predicted_eta"
    ]

    actuals = performance_dataset[
        "actual_eta"
    ]

    if predictions.isna().any() or actuals.isna().any():
        raise ValueError(
            "Performance dataset contains missing prediction "
            "or actual ETA values."
        )

    mae = mean_absolute_error(
        actuals,
        predictions,
    )

    rmse = mean_squared_error(
        actuals,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        actuals,
        predictions,
    )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    logger.info(
        "Performance metrics calculated - "
        "MAE=%.4f, RMSE=%.4f, R2=%.4f",
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
    )

    return metrics


def generate_performance_report(
    performance_dataset: pd.DataFrame,
) -> dict[str, object]:
    """Generate and persist the performance monitoring report."""

    metrics = calculate_performance_metrics(
        performance_dataset
    )

    report = {
        "dataset": {
            "rows": int(len(performance_dataset)),
            "source": str(PERFORMANCE_DATASET_PATH),
        },
        "metrics": metrics,
    }

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PERFORMANCE_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "Performance report saved: %s",
        PERFORMANCE_REPORT_PATH,
    )

    return report


def establish_baseline(
    performance_dataset: pd.DataFrame,
) -> dict[str, float]:
    """Create baseline metrics once and preserve them thereafter."""

    if BASELINE_REPORT_PATH.exists():
        logger.info(
            "Existing baseline found: %s",
            BASELINE_REPORT_PATH,
        )

        with BASELINE_REPORT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            baseline = json.load(file)

        logger.info(
            "Existing baseline preserved - "
            "MAE=%.4f, RMSE=%.4f, R2=%.4f",
            baseline["mae"],
            baseline["rmse"],
            baseline["r2"],
        )

        return baseline

    metrics = calculate_performance_metrics(
        performance_dataset
    )

    baseline = {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    }

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with BASELINE_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            baseline,
            file,
            indent=4,
        )

    logger.info(
        "Baseline metrics established - "
        "MAE=%.4f, RMSE=%.4f, R2=%.4f",
        baseline["mae"],
        baseline["rmse"],
        baseline["r2"],
    )

    logger.info(
        "Baseline metrics saved: %s",
        BASELINE_REPORT_PATH,
    )

    return baseline

# Main method
if __name__ == "__main__":
    dataset = prepare_performance_dataset()

    generate_performance_report(
        dataset
    )

    establish_baseline(
        dataset
    )