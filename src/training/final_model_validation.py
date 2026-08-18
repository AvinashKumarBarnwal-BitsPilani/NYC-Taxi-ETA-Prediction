"""Phase 4.6.4 - Final model validation workflow."""
# Here we are doing (train + validate)

from __future__ import annotations

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
)

from src.training.final_model import (
    load_best_tuning_params,
    train_final_model,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def persist_validation_metrics(
    metrics: dict[str, float],
    output_path: str = "artifacts/final/metrics/final_validation_metrics.json",
) -> Path:
    """Persist final validation metrics for downstream artifact generation."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info(
        "Final validation metrics persisted: %s",
        path,
    )

    return path

def evaluate_final_model(
    model,
    X_val,
    y_val,
) -> dict[str, float]:
    """Evaluate the final model on the validation dataset."""

    predictions = model.predict(X_val)

    rmse = float(
        np.sqrt(mean_squared_error(y_val, predictions))
    )

    mae = float(
        mean_absolute_error(y_val, predictions)
    )

    r2 = float(
        r2_score(y_val, predictions)
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def run_final_model_validation() -> dict[str, float]:
    """Train and validate the final selected model."""

    logger.info("=" * 70)
    logger.info("PHASE 4.6.4 - FINAL MODEL VALIDATION")
    logger.info("=" * 70)

    config = load_modeling_config()
    X_train, X_val, y_train, y_val = load_training_data(config)

    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )

    logger.info("Training final model for validation")
    selected_params = load_best_tuning_params()

    logger.info(
    "Loaded selected XGBoost parameters: %s",
    selected_params,
    )

    final_model = train_final_model(
    X_train=X_train,
    y_train=y_train,
    selected_params=selected_params,
    )

    logger.info("Evaluating final model on validation data")

    metrics = evaluate_final_model(
        final_model,
        X_val,
        y_val,
    )

    logger.info("Final model RMSE: %.4f", metrics["rmse"])
    logger.info("Final model MAE: %.4f", metrics["mae"])
    logger.info("Final model R²: %.6f", metrics["r2"])

    logger.info("=" * 70)
    logger.info("PHASE 4.6.4 FINAL MODEL VALIDATION COMPLETED")
    logger.info("=" * 70)

    return metrics


if __name__ == "__main__":
    run_final_model_validation()