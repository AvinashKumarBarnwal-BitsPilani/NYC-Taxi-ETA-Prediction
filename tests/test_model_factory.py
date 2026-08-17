from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.training.model_factory import create_model

def test_create_linear_regression():
    model = create_model("linear_regression")

    assert isinstance(model, LinearRegression)

def test_create_random_forest():
    model = create_model(
        "random_forest",
        {
            "n_estimators": 10,
            "max_depth": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
    )

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 10
    assert model.max_depth == 5

def test_create_xgboost():
    model = create_model(
        "xgboost",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "tree_method": "hist",
        },
    )

    assert isinstance(model, XGBRegressor)
    assert model.n_estimators == 10
    assert model.max_depth == 3


def test_unsupported_model_raises_error():
    try:
        create_model("unsupported_model")
        assert False, "Expected ValueError was not raised"
    except ValueError as exc:
        assert "Unsupported model" in str(exc)