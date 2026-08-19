"""Tests for Phase 5.2 REST API."""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Verify GET / returns a healthy API response."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "nyc-taxi-eta-prediction",
    }

def test_prediction_endpoint_returns_success():
    """Verify POST /predict returns a successful response."""

    payload = {
        "vendor_id": 2,
        "passenger_count": 1,
        "store_and_fwd_flag": "N",
        "pickup_hour": 1,
        "pickup_day_of_week": 1,
        "pickup_month": 5,
        "is_weekend": 0,
        "distance_km": 9.529875,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_prediction_endpoint_returns_eta():
    """Verify POST /predict returns a numeric ETA."""

    payload = {
        "vendor_id": 2,
        "passenger_count": 1,
        "store_and_fwd_flag": "N",
        "pickup_hour": 1,
        "pickup_day_of_week": 1,
        "pickup_month": 5,
        "is_weekend": 0,
        "distance_km": 9.529875,
    }

    response = client.post("/predict", json=payload)
    data = response.json()

    assert "predicted_eta" in data
    assert isinstance(data["predicted_eta"], float)
    assert data["predicted_eta"] >= 0.0

def test_prediction_rejects_invalid_pickup_month():
    """Verify invalid pickup month is rejected."""

    payload = {
        "vendor_id": 2,
        "passenger_count": 1,
        "store_and_fwd_flag": "N",
        "pickup_hour": 1,
        "pickup_day_of_week": 1,
        "pickup_month": 13,
        "is_weekend": 0,
        "distance_km": 9.529875,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "pickup_month"]


def test_prediction_rejects_negative_distance():
    """Verify negative distance is rejected."""

    payload = {
        "vendor_id": 2,
        "passenger_count": 1,
        "store_and_fwd_flag": "N",
        "pickup_hour": 1,
        "pickup_day_of_week": 1,
        "pickup_month": 5,
        "is_weekend": 0,
        "distance_km": -1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "distance_km"]