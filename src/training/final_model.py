"""Phase 4.6.3 - Final model retraining workflow."""
# Here we are only doing train (No validate)

# Final model is getting trained using the Best Hyperparameter we got in Phase 4.5

from __future__ import annotations

import xgboost as xgb

import json
from pathlib import Path
import joblib
import shutil

from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# This will persist (Save) final model (Phase - 4.7.2)
def persist_final_model(
    model,
    output_path: str = "artifacts/final/model/final_model.joblib",
) -> Path:
    """Persist the validated final model for Phase 5 consumption."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    logger.info("Final model persisted: %s", path)

    return path


# The Phase 3 preprocessor already exists here: data/processed/preprocessor.joblib
# So we will NOT fit or create another preprocessor. We'll copy the exact Phase 3 artifact into the Phase 4 handover area.
# This function was added as part of Phase 4.7.3
def persist_preprocessor(
    source_path: str = "data/processed/preprocessor.joblib",
    output_path: str = "artifacts/final/preprocessing/preprocessor.joblib",
) -> Path:
    """Persist the Phase 3 preprocessor for Phase 5 consumption."""

    source = Path(source_path)
    destination = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(
            f"Phase 3 preprocessor not found: {source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source, destination) # Why copy2()? Because we're not modifying or reserializing the preprocessor.
                                      # We're saying: This exact artifact produced by Phase 3 is the artifact Phase 5 should use.
    logger.info(
        "Preprocessor persisted: %s",
        destination,
    )

    return destination

def load_best_tuning_params(
    input_path: str = "artifacts/tuning/best_params.json",
) -> dict:
    """Load the best hyperparameters produced by Phase 4.5 tuning."""

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Tuning artifact not found: {path}. "
            "Run Phase 4.5 hyperparameter tuning first."
        )

    with path.open("r", encoding="utf-8") as file:
        artifact = json.load(file)

    if artifact.get("model") != "xgboost":
        raise ValueError(
            f"Expected XGBoost tuning artifact, "
            f"found model={artifact.get('model')!r}."
        )

    if "best_params" not in artifact:
        raise ValueError(
            f"Tuning artifact does not contain 'best_params': {path}"
        )

    logger.info(
        "Loaded selected XGBoost hyperparameters from: %s",
        path,
    )

    return artifact["best_params"]

def train_final_model(
    X_train,
    y_train,
    selected_params: dict,
):
    """Train the final XGBoost model using Phase 4.5 tuning output."""

    logger.info("Creating final XGBoost model")
    logger.info(
        "Using selected XGBoost hyperparameters: %s",
        selected_params,
    )

    model = xgb.XGBRegressor(
        **selected_params
    )

    logger.info("Training final XGBoost model")
    model.fit(
        X_train,
        y_train,
    )

    logger.info("Final XGBoost model training completed")
    return model

def run_final_model_training() -> xgb.XGBRegressor:
    """Run the Phase 4.6.3 final model retraining workflow."""

    logger.info("=" * 70)
    logger.info("PHASE 4.6.3 - FINAL MODEL RETRAINING")
    logger.info("=" * 70)

    # 1. Load configuration
    config = load_modeling_config()

    # 2. Load Phase 3 training and validation data
    X_train, X_val, y_train, y_val = load_training_data(config)

    # 3. Validate Phase 3 → Phase 4 data contract
    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )

    selected_params = load_best_tuning_params()

    logger.info(
    "Loaded selected XGBoost parameters: %s",
    selected_params,
    )

    # 4. Train fresh final model
    final_model = train_final_model(
        X_train=X_train,
        y_train=y_train,
    )

    #5. Persist final model final_model.joblib (Phase 4.7.2)
    model_path = persist_final_model(final_model)

    # 6. Persist Phase 3 preprocessor (Phase 4.7.3)
    preprocessor_path = persist_preprocessor()

    logger.info("=" * 70)
    logger.info("PHASE 4.6.3 FINAL MODEL RETRAINING COMPLETED")
    logger.info("=" * 70)

    return final_model


if __name__ == "__main__":
    run_final_model_training()