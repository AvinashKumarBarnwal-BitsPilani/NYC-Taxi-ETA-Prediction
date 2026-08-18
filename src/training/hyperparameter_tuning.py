"""Phase 4.5.3 - Optuna hyperparameter tuning workflow."""

from __future__ import annotations

import mlflow
import optuna
from sklearn.metrics import mean_squared_error

import json
from pathlib import Path

from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
)
from src.training.model_factory import create_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_objective(
    config: dict,
    X_train,
    X_val,
    y_train,
    y_val,
):
    """Create the Optuna objective function for XGBoost tuning.

    Parameters
    ----------
    config:
        Phase 4 modeling configuration.
    X_train, X_val:
        Processed training and validation features.
    y_train, y_val:
        Training and validation target values.

    Returns
    -------
    callable
        Optuna objective function that returns validation RMSE.
    """

    search_space = config["tuning"]["xgboost"]["search_space"]

    def objective(trial: optuna.Trial) -> float:
        """Run one XGBoost tuning trial and return validation RMSE."""

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                search_space["n_estimators"]["min"],
                search_space["n_estimators"]["max"],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                search_space["max_depth"]["min"],
                search_space["max_depth"]["max"],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"]["min"],
                search_space["learning_rate"]["max"],
            ),
            "subsample": trial.suggest_float(
                "subsample",
                search_space["subsample"]["min"],
                search_space["subsample"]["max"],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                search_space["colsample_bytree"]["min"],
                search_space["colsample_bytree"]["max"],
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                search_space["min_child_weight"]["min"],
                search_space["min_child_weight"]["max"],
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                search_space["reg_alpha"]["min"],
                search_space["reg_alpha"]["max"],
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                search_space["reg_lambda"]["min"],
                search_space["reg_lambda"]["max"],
            ),
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "tree_method": "hist",
        }

        model = create_model("xgboost", params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        rmse = mean_squared_error(
            y_val,
            predictions,
        ) ** 0.5

        logger.info(
            "Trial %d completed - RMSE: %.4f",
            trial.number,
            rmse,
        )

        return float(rmse)

    return objective

def save_best_tuning_result(
    study: optuna.Study,
    config: dict,
    output_path: str = "artifacts/tuning/best_params.json",
) -> Path:
    """Persist the best Optuna tuning result."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    result = {
    "model": config["tuning"]["model"],
    "metric": config["tuning"]["metric"],
    "direction": config["tuning"]["direction"],
    "best_value": float(study.best_value),
    "best_params": study.best_params,
}

    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    logger.info("Best tuning result persisted: %s", path)

    return path

def run_hyperparameter_tuning() -> optuna.Study:
    """Run the configured Optuna hyperparameter tuning study."""

    logger.info("=" * 70)
    logger.info("PHASE 4.5.3 - HYPERPARAMETER TUNING")
    logger.info("=" * 70)

    config = load_modeling_config()

    if config["tuning"]["model"] != "xgboost":
        raise ValueError(
            "Phase 4.5 tuning currently supports only xgboost."
    )

    X_train, X_val, y_train, y_val = load_training_data(config)

    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )

    n_trials = config["tuning"]["n_trials"]

    logger.info(
        "Tuning model: %s",
        config["tuning"]["model"],
    )
    logger.info(
        "Optimization metric: %s",
        config["tuning"]["metric"],
    )
    logger.info(
        "Optimization direction: %s",
        config["tuning"]["direction"],
    )
    logger.info(
        "Number of trials: %d",
        n_trials,
    )

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction=config["tuning"]["direction"],
        sampler=sampler,
    )

    objective = build_objective(
        config,
        X_train,
        X_val,
        y_train,
        y_val,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    mlflow.set_experiment("NYC-Taxi-ETA-Model-Development")

    with mlflow.start_run(run_name="hyperparameter_tuning_xgboost"):

        mlflow.set_tag("phase", "4.5")
        mlflow.set_tag("model", config["tuning"]["model"])
        mlflow.set_tag("run_type", "hyperparameter_tuning")
        mlflow.set_tag("sampler", config["tuning"]["sampler"])

        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optimization_metric", config["tuning"]["metric"])
        mlflow.log_param("optimization_direction", config["tuning"]["direction"])

        mlflow.log_metric("best_rmse", study.best_value)

        mlflow.log_params(study.best_params)

        logger.info(
            "Best RMSE: %.4f",
            study.best_value,
        )

        logger.info(
            "Best parameters: %s",
            study.best_params,
        )

        logger.info(
            "MLflow tuning run completed: %s",
            mlflow.active_run().info.run_id,
        )

        tuning_artifact_path = save_best_tuning_result(
        study,
        config,
        )

        logger.info(
        "Tuning artifact created: %s",
         tuning_artifact_path,
        )

    logger.info(
        "Best RMSE: %.4f",
        study.best_value,
    )

    logger.info(
        "Best parameters: %s",
        study.best_params,
    )

    logger.info("=" * 70)
    logger.info("PHASE 4.5.3 COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return study

if __name__ == "__main__":
    run_hyperparameter_tuning()