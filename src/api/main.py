"""Phase 5.2 - FastAPI application."""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI
from src.api.schemas import PredictionRequest
from src.inference.prediction_pipeline import PredictionPipeline

from src.monitoring.prediction_logger import log_prediction

app = FastAPI(
    title="NYC Taxi ETA Prediction API",
    description="REST API for NYC Taxi ETA prediction.",
    version="1.0.0",
)

prediction_pipeline = PredictionPipeline()

@app.get("/")
def health_check() -> dict[str, str]:
    """Return API health status."""

    return {
        "status": "healthy",
        "service": "nyc-taxi-eta-prediction",
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    """Generate an ETA prediction from raw taxi features."""

    features = pd.DataFrame([request.model_dump()])
    prediction = prediction_pipeline.predict(features)

    log_prediction(
        features=features,
        predicted_eta=prediction,
    )

    return {
        "predicted_eta": prediction,
    }