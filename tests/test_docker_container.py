"""Phase 5.4.6 - Docker container smoke tests."""

from __future__ import annotations

import json
import subprocess
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
IMAGE_NAME = "avinashkumarb6/nyc-taxi-eta-prediction:4.0"
CONTAINER_NAME = "pedantic_dubinsky"


VALID_PAYLOAD = {
    "vendor_id": 2,
    "passenger_count": 1,
    "store_and_fwd_flag": "N",
    "pickup_hour": 1,
    "pickup_day_of_week": 1,
    "pickup_month": 5,
    "is_weekend": 0,
    "distance_km": 9.529875,
}


def test_docker_image_exists() -> None:
    """Verify the Docker image exists locally."""

    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Docker image not found: {IMAGE_NAME}\n"
        f"{result.stderr}"
    )


def test_docker_container_is_running() -> None:
    """Verify the expected Docker container is running."""

    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}",
            CONTAINER_NAME,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Docker container not found: {CONTAINER_NAME}\n"
        f"{result.stderr}"
    )

    assert result.stdout.strip() == "true"


def test_docker_health_endpoint() -> None:
    """Verify the containerized health endpoint."""

    with urlopen(f"{BASE_URL}/", timeout=5) as response:
        body = json.loads(response.read().decode())

    assert response.status == 200
    assert body["status"] == "healthy"
    assert body["service"] == "nyc-taxi-eta-prediction"


def test_docker_prediction_endpoint() -> None:
    """Verify the containerized prediction endpoint."""

    request = Request(
        f"{BASE_URL}/predict",
        data=json.dumps(VALID_PAYLOAD).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=10) as response:
        body = json.loads(response.read().decode())

    assert response.status == 200
    assert "predicted_eta" in body
    assert isinstance(body["predicted_eta"], (int, float))


def test_docker_validation_endpoint() -> None:
    """Verify invalid input is rejected by the containerized API."""

    invalid_payload = {
        **VALID_PAYLOAD,
        "pickup_month": 13,
    }

    request = Request(
        f"{BASE_URL}/predict",
        data=json.dumps(invalid_payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        urlopen(request, timeout=5)
        raise AssertionError("Expected HTTP 422 validation error")

    except HTTPError as exc:
        assert exc.code == 422