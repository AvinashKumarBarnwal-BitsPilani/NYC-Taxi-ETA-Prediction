"""Candidate model training workflow for Phase 4.3."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import mlflow
import joblib

from src.training.baseline import calculate_regression_metrics
from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
    validate_modeling_config,
)
from src.training.model_factory import create_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_EXPERIMENT_NAME = "NYC-Taxi-ETA-Model-Development"

def configure_mlflow() -> None:
    """Configure the MLflow experiment used for candidate training."""

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info(
        "MLflow experiment configured: %s",
        MLFLOW_EXPERIMENT_NAME,
    )

def get_enabled_candidates(config: dict) -> dict[str, dict[str, Any]]:
    """Return enabled candidate models and their parameters."""

    candidates = config.get("candidates", {})
    enabled_candidates: dict[str, dict[str, Any]] = {}

    for model_name, model_config in candidates.items():
        if model_config.get("enabled", False):
            enabled_candidates[model_name] = model_config

    if not enabled_candidates:
        raise ValueError("No enabled candidate models found in modeling.yaml.")

    return enabled_candidates


def train_candidate(
    model_name: str,
    model_config: dict[str, Any],
    X_train,
    y_train,
):
    """Create and train one candidate model."""

    params = model_config.get("params", {})

    logger.info(
        "Creating candidate model: %s",
        model_name,
    )

    model = create_model(
        model_name=model_name,
        params=params,
    )

    logger.info(
        "Training candidate model: %s",
        model_name,
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Candidate model training completed: %s",
        model_name,
    )

    return model

# This function will persist the trained candidate model to disk for later comparison and tuning. 
def persist_candidate_model(
    model_name: str,
    model: Any,
    config: dict,
) -> Path:
    """Persist a trained candidate model for later comparison and tuning."""

    model_directory = (
        PROJECT_ROOT / config["artifacts"]["model_directory"]
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = model_directory / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    logger.info(
        "Candidate model persisted: %s",
        model_path,
    )
    return model_path


def run_candidate_training() -> None:
    """Run the Phase 4.3 candidate training workflow."""

    logger.info("=" * 70)
    logger.info("PHASE 4.3 - CANDIDATE MODEL TRAINING")
    logger.info("=" * 70)

    # ----------------------------------------------------------
    # 1. Load and validate Phase 4 configuration
    # ----------------------------------------------------------
    config = load_modeling_config()
    validate_modeling_config(config)

    # ----------------------------------------------------------
    # 2. Load and validate Phase 3 ML-ready data
    # ----------------------------------------------------------
    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = load_training_data(config)

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
    # 3. Identify enabled candidate models
    # ----------------------------------------------------------
    enabled_candidates = get_enabled_candidates(config)
    logger.info(
        "Enabled candidate models: %s",
        list(enabled_candidates.keys()),
    )

    # ----------------------------------------------------------
    # 4. Configure MLflow
    # ----------------------------------------------------------
    configure_mlflow()

    # ----------------------------------------------------------
    # 5. Train each candidate
    # ----------------------------------------------------------
    for model_name, model_config in enabled_candidates.items():

        logger.info("-" * 70)
        logger.info(
            "Training candidate: %s",
            model_name,
        )
        logger.info("-" * 70)

        with mlflow.start_run(
            run_name=f"candidate_{model_name}",
        ):

            model = train_candidate(
                model_name=model_name,
                model_config=model_config,
                X_train=X_train,
                y_train=y_train_series,
            )

            # --------------------------------------------------
            # Validation prediction
            # --------------------------------------------------
            logger.info(
                "Generating validation predictions: %s",
                model_name,
            )

            y_val_pred = model.predict(
                X_val,
            )

            # --------------------------------------------------
            # Evaluation
            # --------------------------------------------------
            metrics = calculate_regression_metrics(
                y_val_series,
                y_val_pred,
            )

            logger.info(
                "%s RMSE: %.4f",
                model_name,
                metrics["rmse"],
            )

            logger.info(
                "%s MAE: %.4f",
                model_name,
                metrics["mae"],
            )

            logger.info(
                "%s R²: %.6f",
                model_name,
                metrics["r2"],
            )

            # --------------------------------------------------
            # MLflow parameters
            # --------------------------------------------------
            mlflow.log_param(
                "model_type",
                model_name,
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

            # --------------------------------------------------
            # MLflow metrics
            # --------------------------------------------------
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

            # --------------------------------------------------
            # MLflow tags
            # --------------------------------------------------
            mlflow.set_tag(
                "phase",
                "4.3",
            )

            mlflow.set_tag(
                "model_role",
                "candidate",
            )

            mlflow.set_tag(
                "metric_priority",
                "rmse",
            )

            # Call the function to persist the candidate model to disk
            model_path = persist_candidate_model(
            model_name=model_name,
            model=model,
            config=config,
            )
            
            logger.info(
                "MLflow run completed: %s",
                mlflow.active_run().info.run_id,
            )

    logger.info("=" * 70)
    logger.info("PHASE 4.3 CANDIDATE TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_candidate_training()