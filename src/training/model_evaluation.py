"""Phase 4.4 model evaluation workflow."""

from __future__ import annotations
from pathlib import Path
import json

from typing import Any

import joblib
import pandas as pd

from src.training.baseline import calculate_regression_metrics
from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
    validate_modeling_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_candidate_models(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load persisted candidate models from the configured model directory."""

    model_directory = (
        PROJECT_ROOT / config["artifacts"]["model_directory"]
    )

    candidates = config.get("candidates", {})
    models: dict[str, Any] = {}

    for model_name, model_config in candidates.items():
        if not model_config.get("enabled", False):
            continue

        model_path = model_directory / f"{model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Candidate model artifact not found: {model_path}"
            )

        logger.info(
            "Loading candidate model: %s",
            model_path,
        )

        models[model_name] = joblib.load(model_path)

    if not models:
        raise ValueError(
            "No enabled candidate model artifacts were found."
        )

    return models

def evaluate_model(
    model_name: str,
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Load a candidate model,Generate validation predictions and calculate regression metrics."""

    logger.info(
        "Evaluating candidate model: %s",
        model_name,
    )

    predictions = model.predict(X_val)

    metrics = calculate_regression_metrics(
        y_val,
        predictions,
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

    return metrics

def save_comparison_report(results: dict, output_path: str) -> None:
    """Persist model evaluation results as a JSON comparison report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    logger.info("Model comparison report persisted: %s", path)

def run_model_evaluation() -> dict[str, dict[str, float]]:
    """Evaluate all persisted candidate models on the validation dataset."""

    logger.info("=" * 70)
    logger.info("PHASE 4.4 - MODEL EVALUATION")
    logger.info("=" * 70)

    # ----------------------------------------------------------
    # 1. Load and validate configuration
    # ----------------------------------------------------------
    config = load_modeling_config()
    validate_modeling_config(config)

    # ----------------------------------------------------------
    # 2. Load Phase 4 validation data
    # ----------------------------------------------------------
    (
        _X_train,
        X_val,
        _y_train,
        y_val,
    ) = load_training_data(config)

    validate_data_contract(
        _X_train,
        X_val,
        _y_train,
        y_val,
        config,
    )

    target_column = config["problem"]["target"]
    y_val_series = y_val[target_column]

    # ----------------------------------------------------------
    # 3. Load persisted candidate models
    # ----------------------------------------------------------
    models = load_candidate_models(config)

    logger.info(
        "Candidate models available for evaluation: %s",
        list(models.keys()),
    )

    # ----------------------------------------------------------
    # 4. Evaluate candidates
    # ----------------------------------------------------------
    results: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        results[model_name] = evaluate_model(
            model_name=model_name,
            model=model,
            X_val=X_val,
            y_val=y_val_series,
        )

    logger.info("=" * 70)
    logger.info("PHASE 4.4 MODEL EVALUATION COMPLETED")
    logger.info("=" * 70)

    report = {
        "primary_metric": "rmse",
        "secondary_metric": "mae",
        "context_metric": "r2",
        "baseline": {
            "rmse": 3258.366782573327,
            "mae": 641.4636558199387,
            "r2": -0.00040622054028571775,
        },
        "candidates": results,
    }

    save_comparison_report(
        report,
        "reports/candidate_model_comparison.json",
    )

    return results

if __name__ == "__main__":
    run_model_evaluation()