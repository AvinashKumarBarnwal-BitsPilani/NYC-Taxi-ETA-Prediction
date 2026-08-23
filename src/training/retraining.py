"""Phase 6.7 - Retraining pipeline."""

# Reuse the existing Phase 4 data pipeline and validation for retraining.
# Loads the ML-ready train/validation data and selected XGBoost parameters
# without modifying or overwriting the existing production model.

from __future__ import annotations

import logging
from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
)
from src.training.final_model import (
    load_best_tuning_params,
    train_final_model,
)

from src.training.final_model_validation import evaluate_final_model

from src.utils.logger import get_logger

import shutil
import json
import joblib
from pathlib import Path

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "final"
    / "model"
    / "final_model.joblib"
)

PRODUCTION_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "final"
    / "metrics"
    / "final_metrics.json"
)

RETRAINED_MODEL_DIR = (
    PROJECT_ROOT / "artifacts" / "retrained" / "v1"
)

RETRAINED_MODEL_PATH = (
    RETRAINED_MODEL_DIR / "model.joblib"
)


def load_retraining_data():
    """Load and validate the existing Phase 3 ML-ready data for retraining."""

    logger.info("RETRAINING DATA PREPARATION STARTED")

    # ----------------------------------------------------------
    # 1. Load existing modeling configuration
    # ----------------------------------------------------------
    config = load_modeling_config()

    # ----------------------------------------------------------
    # 2. Load existing Phase 3 ML-ready datasets
    # ----------------------------------------------------------
    X_train, X_val, y_train, y_val = load_training_data(config)

    validate_retraining_preprocessing(
    X_train,
    X_val,
    )

    logger.info(
        "Retraining data loaded - X_train=%s, X_val=%s",
        X_train.shape,
        X_val.shape,
    )

    # ----------------------------------------------------------
    # 3. Reuse existing Phase 4 data contract validation
    # ----------------------------------------------------------
    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )

    # ----------------------------------------------------------
    # 4. Reuse existing selected XGBoost parameters
    # ----------------------------------------------------------
    selected_params = load_best_tuning_params()

    logger.info(
        "Selected XGBoost parameters loaded for retraining"
    )

    logger.info("RETRAINING DATA PREPARATION COMPLETED")

    return (
        config,
        X_train,
        X_val,
        y_train,
        y_val,
        selected_params,
    )


def validate_retraining_preprocessing(
    X_train,
    X_val,
) -> None:
    """Verify that retraining uses the existing processed feature data."""

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Retraining train and validation feature counts must match."
        )

    if X_train.isnull().any().any():
        raise ValueError(
            "Retraining training data contains missing values."
        )

    if X_val.isnull().any().any():
        raise ValueError(
            "Retraining validation data contains missing values."
        )

    logger.info(
        "Existing preprocessing reused - "
        "training features=%d, validation features=%d",
        X_train.shape[1],
        X_val.shape[1],
    )


def train_candidate_model(
    X_train,
    y_train,
    selected_params,
):
    """Train and persist a versioned retrained candidate model."""

    logger.info("CANDIDATE MODEL TRAINING STARTED")
    candidate_model = train_final_model(
        X_train,
        y_train,
        selected_params,
    )

    RETRAINED_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        candidate_model,
        RETRAINED_MODEL_PATH,
    )

    logger.info(
        "Candidate model saved: %s",
        RETRAINED_MODEL_PATH,
    )
    logger.info("CANDIDATE MODEL TRAINING COMPLETED")

    return candidate_model


def persist_candidate_metrics(
    metrics: dict[str, float],
) -> Path:
    """Persist candidate model evaluation metrics."""

    metrics_path = (
        RETRAINED_MODEL_DIR / "metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    logger.info(
        "Candidate metrics saved: %s",
        metrics_path,
    )

    return metrics_path


def evaluate_candidate_model(
    candidate_model,
    X_val,
    y_val,
) -> dict[str, float]:
    """Evaluate the retrained candidate using the existing Phase 4 evaluator."""

    logger.info("CANDIDATE MODEL EVALUATION STARTED")

    metrics = evaluate_final_model(
        candidate_model,
        X_val,
        y_val,
    )

    logger.info(
        "Candidate MAE: %.4f",
        metrics["mae"],
    )
    logger.info(
        "Candidate RMSE: %.4f",
        metrics["rmse"],
    )
    logger.info(
        "Candidate R²: %.6f",
        metrics["r2"],
    )

    logger.info("CANDIDATE MODEL EVALUATION COMPLETED")

    return metrics


def compare_candidate_with_production(
    candidate_metrics: dict[str, float],
) -> dict:
    """Compare candidate metrics against the Phase 4 production baseline."""

    logger.info("CANDIDATE VS PRODUCTION COMPARISON STARTED")

    with PRODUCTION_METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        production_metrics = json.load(file)

    production_mae = production_metrics["metrics"]["mae"]
    production_rmse = production_metrics["metrics"]["rmse"]
    production_r2 = production_metrics["metrics"]["r2"]

    candidate_mae = candidate_metrics["mae"]
    candidate_rmse = candidate_metrics["rmse"]
    candidate_r2 = candidate_metrics["r2"]

    mae_improvement = (
        (production_mae - candidate_mae)
        / production_mae
    )

    rmse_improvement = (
        (production_rmse - candidate_rmse)
        / production_rmse
    )

    r2_improvement = (
        (candidate_r2 - production_r2)
        / abs(production_r2)
    )

    candidate_better = (
        candidate_mae < production_mae
        and candidate_rmse < production_rmse
        and candidate_r2 > production_r2
    )

    comparison = {
        "production": {
            "mae": production_mae,
            "rmse": production_rmse,
            "r2": production_r2,
        },
        "candidate": {
            "mae": candidate_mae,
            "rmse": candidate_rmse,
            "r2": candidate_r2,
        },
        "improvement": {
            "mae": mae_improvement,
            "rmse": rmse_improvement,
            "r2": r2_improvement,
        },
        "candidate_better": candidate_better,
    }

    logger.info(
        "Production MAE: %.4f | Candidate MAE: %.4f",
        production_mae,
        candidate_mae,
    )

    logger.info(
        "Production RMSE: %.4f | Candidate RMSE: %.4f",
        production_rmse,
        candidate_rmse,
    )

    logger.info(
        "Production R²: %.6f | Candidate R²: %.6f",
        production_r2,
        candidate_r2,
    )

    logger.info(
        "Candidate better than production: %s",
        candidate_better,
    )

    logger.info("CANDIDATE VS PRODUCTION COMPARISON COMPLETED")

    return comparison


def persist_comparison_report(
    comparison: dict,
) -> Path:
    """Persist candidate vs production comparison."""

    comparison_path = (
        RETRAINED_MODEL_DIR
        / "comparison_report.json"
    )

    with comparison_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
        )

    logger.info(
        "Comparison report saved: %s",
        comparison_path,
    )

    return comparison_path


def decide_model_promotion(
    comparison: dict,
) -> dict:
    """Promote candidate only when it outperforms production."""

    logger.info("MODEL PROMOTION DECISION STARTED")
    candidate_better = comparison["candidate_better"]

    if candidate_better:
        logger.info(
            "Candidate is better - promoting candidate model"
        )

        PRODUCTION_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            RETRAINED_MODEL_PATH,
            PRODUCTION_MODEL_PATH,
        )

        decision = "PROMOTE_CANDIDATE"
        production_model_changed = True

    else:
        logger.info(
            "Candidate is not better - retaining production model"
        )

        decision = "RETAIN_PRODUCTION"
        production_model_changed = False

    promotion_decision = {
        "decision": decision,
        "candidate_better": candidate_better,
        "production_model_changed": production_model_changed,
    }

    logger.info(
        "Model promotion decision: %s",
        decision,
    )
    logger.info("MODEL PROMOTION DECISION COMPLETED")

    return promotion_decision


def persist_promotion_decision(
    promotion_decision: dict,
) -> Path:
    """Persist the model promotion decision."""

    decision_path = (
        RETRAINED_MODEL_DIR
        / "promotion_decision.json"
    )

    with decision_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            promotion_decision,
            file,
            indent=4,
        )

    logger.info(
        "Promotion decision saved: %s",
        decision_path,
    )

    return decision_path


def main() -> None:
    """Prepare data and train the retrained candidate model."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    (
        config,
        X_train,
        X_val,
        y_train,
        y_val,
        selected_params,
    ) = load_retraining_data()

    candidate_model = train_candidate_model(
    X_train,
    y_train,
    selected_params,
)

    candidate_metrics = evaluate_candidate_model(
    candidate_model,
    X_val,
    y_val,
)

    persist_candidate_metrics(
    candidate_metrics,
)  

    comparison = compare_candidate_with_production(
    candidate_metrics,
)

    persist_comparison_report(
    comparison,
)

    promotion_decision = decide_model_promotion(
    comparison,
)

    persist_promotion_decision(
    promotion_decision,
)

if __name__ == "__main__":
    main()