"""Tests for the Phase 4.4 model evaluation workflow."""

import json
#from pathlib import Path

from src.training.model_evaluation import (
    evaluate_model,
    save_comparison_report,
)


def test_evaluate_model_returns_required_metrics():
    """Evaluation should return RMSE, MAE, and R²."""

    class DummyModel:
        def predict(self, X):
            return [10.0, 20.0, 30.0]

    metrics = evaluate_model(
        model_name="test_model",
        model=DummyModel(),
        X_val=[[1], [2], [3]],
        y_val=[12.0, 18.0, 31.0],
    )

    assert set(metrics.keys()) == {"rmse", "mae", "r2"}

    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_save_comparison_report_creates_valid_json(tmp_path):
    """Comparison results should be persisted as valid JSON."""

    report = {
        "primary_metric": "rmse",
        "secondary_metric": "mae",
        "context_metric": "r2",
        "baseline": {
            "rmse": 3258.3668,
            "mae": 641.4637,
            "r2": -0.0004,
        },
        "candidates": {
            "linear_regression": {
                "rmse": 3205.0706,
                "mae": 467.0761,
                "r2": 0.0321,
            },
            "random_forest": {
                "rmse": 3236.4813,
                "mae": 464.3314,
                "r2": 0.0130,
            },
            "xgboost": {
                "rmse": 3203.3086,
                "mae": 451.0966,
                "r2": 0.0331,
            },
        },
    }

    output_path = tmp_path / "candidate_model_comparison.json"

    save_comparison_report(
        report,
        str(output_path),
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert loaded_report["primary_metric"] == "rmse"
    assert "baseline" in loaded_report
    assert "candidates" in loaded_report

    assert set(loaded_report["candidates"]) == {
        "linear_regression",
        "random_forest",
        "xgboost",
    }


def test_comparison_report_contains_all_required_metrics(tmp_path):
    """Each candidate must contain RMSE, MAE, and R²."""

    report = {
        "baseline": {
            "rmse": 3258.3668,
            "mae": 641.4637,
            "r2": -0.0004,
        },
        "candidates": {
            "linear_regression": {
                "rmse": 3205.0706,
                "mae": 467.0761,
                "r2": 0.0321,
            },
            "random_forest": {
                "rmse": 3236.4813,
                "mae": 464.3314,
                "r2": 0.0130,
            },
            "xgboost": {
                "rmse": 3203.3086,
                "mae": 451.0966,
                "r2": 0.0331,
            },
        },
    }

    output_path = tmp_path / "comparison.json"

    save_comparison_report(
        report,
        str(output_path),
    )

    with output_path.open("r", encoding="utf-8") as file:
        loaded_report = json.load(file)

    for metrics in loaded_report["candidates"].values():
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}