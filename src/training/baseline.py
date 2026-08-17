"""Baseline regression model for Phase 4 model development."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mlflow
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
    validate_modeling_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_metrics.json"

MLFLOW_EXPERIMENT_NAME = "NYC-Taxi-ETA-Model-Development"

def calculate_regression_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Calculate the standard Phase 4 regression metrics.

    Metrics:
        RMSE - Primary metric.
        MAE  - Secondary metric.
        R2   - Context metric.
    """

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def save_baseline_metrics(
    metrics: dict[str, float],
    strategy: str,
    training_rows: int,
    validation_rows: int,
) -> None:
    """Persist baseline metrics as a JSON report."""

    BASELINE_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model": "DummyRegressor",
        "strategy": strategy,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "metrics": metrics,
    }

    with BASELINE_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    logger.info(
        "Baseline metrics saved: %s",
        BASELINE_REPORT_PATH,
    )

def configure_mlflow() -> None:
    """Configure the MLflow experiment used for Phase 4."""

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    logger.info(
        "MLflow experiment configured: %s",
        MLFLOW_EXPERIMENT_NAME,
    )

def train_baseline(
    X_train,
    y_train,
) -> DummyRegressor:
    """Train the mean-based DummyRegressor baseline."""

    logger.info(
        "Training baseline model: DummyRegressor(strategy='mean')"
    )

    model = DummyRegressor(
        strategy="mean"
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Baseline model training completed"
    )

    return model


def run_baseline() -> dict[str, float]:
    """Run the complete Phase 4.2 baseline workflow."""

    logger.info("=" * 70)
    logger.info(
        "PHASE 4.2 - BASELINE MODEL"
    )
    logger.info("=" * 70)

    # ----------------------------------------------------------
    # Load and validate Phase 4 configuration
    # ----------------------------------------------------------

    config = load_modeling_config()

    validate_modeling_config(
        config
    )

    # ----------------------------------------------------------
    # Load and validate Phase 3 handover data
    # ----------------------------------------------------------

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = load_training_data(
        config
    )

    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )

    target_column = config["problem"]["target"]

    y_train_series = y_train[target_column]
    y_val_series = y_val[target_column]

    # ----------------------------------------------------------
    # Configure MLflow
    # ----------------------------------------------------------

    configure_mlflow()

    # ----------------------------------------------------------
    # Train baseline and track the experiment
    # ----------------------------------------------------------

    with mlflow.start_run(
        run_name="baseline_dummy_regressor_mean"
    ) as run:

        model = train_baseline(
            X_train,
            y_train_series,
        )

        # ------------------------------------------------------
        # Validation prediction
        # ------------------------------------------------------

        logger.info(
            "Generating baseline predictions on validation data"
        )

        y_val_pred = model.predict(
            X_val
        )

        # ------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------

        metrics = calculate_regression_metrics(
            y_val_series,
            y_val_pred,
        )

        logger.info(
            "Baseline RMSE: %.4f",
            metrics["rmse"],
        )

        logger.info(
            "Baseline MAE: %.4f",
            metrics["mae"],
        )

        logger.info(
            "Baseline R²: %.6f",
            metrics["r2"],
        )

        # ------------------------------------------------------
        # MLflow parameters
        # ------------------------------------------------------

        mlflow.log_param(
            "model_type",
            "DummyRegressor",
        )

        mlflow.log_param(
            "strategy",
            "mean",
        )

        mlflow.log_param(
            "problem_type",
            config["problem"]["type"],
        )

        mlflow.log_param(
            "target",
            target_column,
        )

        mlflow.log_param(
            "training_rows",
            len(X_train),
        )

        mlflow.log_param(
            "validation_rows",
            len(X_val),
        )

        mlflow.log_param(
            "feature_count",
            X_train.shape[1],
        )

        # ------------------------------------------------------
        # MLflow metrics
        # ------------------------------------------------------

        mlflow.log_metric(
            "rmse",
            metrics["rmse"],
        )

        mlflow.log_metric(
            "mae",
            metrics["mae"],
        )

        mlflow.log_metric(
            "r2",
            metrics["r2"],
        )

        # ------------------------------------------------------
        # Run tags
        # ------------------------------------------------------

        mlflow.set_tag(
            "phase",
            "4.2",
        )

        mlflow.set_tag(
            "model_role",
            "baseline",
        )

        mlflow.set_tag(
            "metric_priority",
            "rmse",
        )

        run_id = run.info.run_id

        logger.info(
            "MLflow run completed: %s",
            run_id,
        )

    # ----------------------------------------------------------
    # Save local baseline evidence
    # ----------------------------------------------------------

    save_baseline_metrics(
        metrics=metrics,
        strategy="mean",
        training_rows=len(X_train),
        validation_rows=len(X_val),
    )

    logger.info("=" * 70)
    logger.info(
        "PHASE 4.2 BASELINE COMPLETED SUCCESSFULLY"
    )
    logger.info("=" * 70)

    return metrics

if __name__ == "__main__":
    run_baseline()