"""
Model factory for Phase 4 candidate model training.

This module is responsible only for constructing configured
candidate regression models.

It does not:
    - load data
    - train models
    - evaluate models
    - perform hyperparameter tuning
    - log MLflow runs
    - save model artifacts
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


SUPPORTED_MODELS = {
    "linear_regression",
    "random_forest",
    "xgboost",
}


def create_model(model_name: str, params: dict[str, Any] | None = None) -> Any:
    """
    Create a configured regression model.

    Parameters
    ----------
    model_name:
        Name of the candidate model.

    params:
        Model-specific hyperparameters.

    Returns
    -------
    Any
        Configured scikit-learn compatible regression estimator.

    Raises
    ------
    ValueError
        If the model name is not supported.
    """

    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models: {supported}"
        )

    params = params or {}

    if model_name == "linear_regression":
        return LinearRegression(**params)

    if model_name == "random_forest":
        return RandomForestRegressor(**params)

    if model_name == "xgboost":
        return XGBRegressor(**params)

    # Defensive fallback.
    raise RuntimeError(
        f"Model '{model_name}' is listed as supported "
        "but has no implementation."
    )