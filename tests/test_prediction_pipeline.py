"""Tests for Phase 5.1 prediction pipeline."""

from pathlib import Path
import pandas as pd
import pytest

from src.inference.prediction_pipeline import PredictionPipeline

RAW_FEATURES = {
    "vendor_id": [2],
    "passenger_count": [1],
    "store_and_fwd_flag": ["N"],
    "pickup_hour": [1],
    "pickup_day_of_week": [1],
    "pickup_month": [5],
    "is_weekend": [0],
    "distance_km": [9.529875],
}

def test_prediction_pipeline_loads_artifacts():
    """Verify the pipeline can load the persisted artifacts."""

    pipeline = PredictionPipeline()
    pipeline.load_artifacts()

    assert pipeline.model is not None
    assert pipeline.preprocessor is not None

def test_prediction_pipeline_returns_numeric_prediction():
    """Verify valid raw input produces a numeric prediction."""

    features = pd.DataFrame(RAW_FEATURES)
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(features)

    assert isinstance(prediction, float)
    assert prediction >= 0.0

def test_prediction_pipeline_rejects_non_dataframe_input():
    """Verify prediction input must be a pandas DataFrame."""

    pipeline = PredictionPipeline()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        pipeline.predict(RAW_FEATURES)

def test_prediction_pipeline_missing_model_fails():
    """Verify missing model artifact raises FileNotFoundError."""

    pipeline = PredictionPipeline(
        model_path="artifacts/final/model/missing_model.joblib",
    )

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        pipeline.load_artifacts()

def test_prediction_pipeline_missing_preprocessor_fails():
    """Verify missing preprocessor artifact raises FileNotFoundError."""

    pipeline = PredictionPipeline(
        preprocessor_path=(
            "artifacts/final/preprocessing/missing_preprocessor.joblib"
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Preprocessor artifact not found",
    ):
        pipeline.load_artifacts()