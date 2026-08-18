"""Tests for Phase 4.5 hyperparameter tuning."""

import optuna

from src.training.data_contract import load_modeling_config
from src.training.hyperparameter_tuning import build_objective


def test_tuning_configuration_is_valid():
    """Verify the configured tuning strategy and search space."""

    config = load_modeling_config()
    tuning = config["tuning"]

    assert tuning["framework"] == "optuna"
    assert tuning["sampler"] == "tpe"
    assert tuning["n_trials"] == 20
    assert tuning["metric"] == "rmse"
    assert tuning["direction"] == "minimize"
    assert tuning["model"] == "xgboost"

    expected_parameters = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
    }

    assert set(tuning["xgboost"]["search_space"]) == expected_parameters


def test_optuna_objective_returns_numeric_rmse():
    """Verify that the Optuna objective can execute a trial."""

    config = load_modeling_config()

    X_train = [[1.0], [2.0], [3.0], [4.0]]
    X_val = [[1.5], [2.5]]
    y_train = [10.0, 20.0, 30.0, 40.0]
    y_val = [15.0, 25.0]

    objective = build_objective(
        config,
        X_train,
        X_val,
        y_train,
        y_val,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    value = objective(study.ask())

    assert isinstance(value, float)
    assert value >= 0.0