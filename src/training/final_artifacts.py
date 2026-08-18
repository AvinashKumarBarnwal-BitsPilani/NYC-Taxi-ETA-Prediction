"""Phase 4.7.4 - Final model metrics and metadata artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import joblib

from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
)
from src.training.final_model import load_best_tuning_params
from src.training.final_model_validation import evaluate_final_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_json(data: dict, output_path: str) -> Path:
    """Persist a dictionary as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    logger.info("JSON artifact persisted: %s", path)

    return path


def save_final_metrics(
    metrics: dict[str, float],
    output_path: str = "artifacts/final/metrics/final_metrics.json",
) -> Path:
    """Persist actual final model validation metrics."""

    data = {
        "primary_metric": "rmse",
        "secondary_metric": "mae",
        "context_metric": "r2",
        "metrics": metrics,
    }

    return save_json(data, output_path)


def save_model_metadata(
    selected_params: dict,
    metrics: dict[str, float],
    training_rows: int,
    training_features: int,
    validation_rows: int,
    output_path: str = "artifacts/final/metadata/model_metadata.json",
) -> Path:
    """Persist final model metadata and actual configuration."""

    metadata = {
        "model_name": "xgboost",
        "model_type": "XGBRegressor",
        "model_stage": "final",
        "selection_metric": "rmse",
        "selection_direction": "minimize",
        "selected_hyperparameters": selected_params,
        "training_rows": training_rows,
        "training_features": training_features,
        "validation_rows": validation_rows,
        "feature_count": training_features,
        "validation_metrics": metrics,
        "artifacts": {
            "model": "artifacts/final/model/final_model.joblib",
            "preprocessor": (
                "artifacts/final/preprocessing/preprocessor.joblib"
            ),
            "metrics": "artifacts/final/metrics/final_metrics.json",
        },
    }

    return save_json(metadata, output_path)


def create_final_metadata_artifacts() -> tuple[Path, Path]:
    """Create final metrics and model metadata artifacts dynamically."""

    logger.info("=" * 70)
    logger.info("PHASE 4.7.4 - FINAL METRICS & MODEL METADATA")
    logger.info("=" * 70)

    # 1. Load Phase 4 modeling configuration
    config = load_modeling_config()

    # 2. Load Phase 3 ML-ready datasets
    X_train, X_val, y_train, y_val = load_training_data(config)

    # 3. Validate Phase 3 → Phase 4 data contract
    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )

    # 4. Load the persisted final model
    model_path = Path(
        "artifacts/final/model/final_model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final model artifact not found: {model_path}. "
            "Run Phase 4.6.3 final model training first."
        )

    logger.info("Loading persisted final model: %s", model_path)

    final_model = joblib.load(model_path)

    # 5. Load the actual hyperparameters selected by Optuna
    selected_params = load_best_tuning_params()

    logger.info(
        "Loaded selected XGBoost parameters: %s",
        selected_params,
    )

    # 6. Evaluate the persisted final model
    logger.info(
        "Evaluating persisted final model on validation data"
    )

    metrics = evaluate_final_model(
        final_model,
        X_val,
        y_val,
    )

    logger.info("Final model RMSE: %.4f", metrics["rmse"])
    logger.info("Final model MAE: %.4f", metrics["mae"])
    logger.info("Final model R²: %.6f", metrics["r2"])

    # 7. Persist actual validation metrics
    metrics_path = save_final_metrics(metrics)

    # 8. Persist actual model metadata
    metadata_path = save_model_metadata(
        selected_params=selected_params,
        metrics=metrics,
        training_rows=len(X_train),
        training_features=X_train.shape[1],
        validation_rows=len(X_val),
    )

    logger.info("Final metrics artifact: %s", metrics_path)
    logger.info("Model metadata artifact: %s", metadata_path)

    logger.info("=" * 70)
    logger.info("PHASE 4.7.4 COMPLETED")
    logger.info("=" * 70)

    return metrics_path, metadata_path


if __name__ == "__main__":
    create_final_metadata_artifacts()