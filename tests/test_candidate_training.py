"""Focused tests for Phase 4.3 candidate training utilities."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

import src.training.candidate_training as candidate_training
from src.training.data_contract import load_modeling_config


def test_enabled_candidates_are_loaded_from_configuration():
    """Verify enabled candidate models are read from modeling.yaml."""

    config = load_modeling_config()

    candidates = candidate_training.get_enabled_candidates(config)

    assert set(candidates) == {
        "linear_regression",
        "random_forest",
        "xgboost",
    }


def test_candidate_model_is_persisted_and_reloadable(tmp_path, monkeypatch):
    """Verify a trained candidate model can be persisted and loaded back."""

    monkeypatch.setattr(candidate_training, "PROJECT_ROOT", Path(tmp_path))

    config = {
        "artifacts": {
            "model_directory": "models"
        }
    }

    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.fit(X, y)

    model_path = candidate_training.persist_candidate_model(
        model_name="linear_regression",
        model=model,
        config=config,
    )

    assert model_path.exists()
    assert model_path.name == "linear_regression.joblib"

    loaded_model = joblib.load(model_path)

    predictions = loaded_model.predict(np.array([[5]]))

    assert predictions.shape == (1,)
    assert predictions[0] == 10