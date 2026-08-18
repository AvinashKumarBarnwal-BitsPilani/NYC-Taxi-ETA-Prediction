"""Tests for Phase 4.7 final handover artifacts."""

import json
from pathlib import Path


FINAL_ARTIFACTS = Path("artifacts/final")


def test_required_handover_artifacts_exist():
    """Verify all required final handover artifacts exist."""

    required_artifacts = [
        FINAL_ARTIFACTS / "model" / "final_model.joblib",
        FINAL_ARTIFACTS / "preprocessing" / "preprocessor.joblib",
        FINAL_ARTIFACTS / "metrics" / "final_metrics.json",
        FINAL_ARTIFACTS / "metadata" / "model_metadata.json",
        FINAL_ARTIFACTS / "manifest.json",
    ]

    for artifact in required_artifacts:
        assert artifact.exists(), f"Missing handover artifact: {artifact}"


def test_final_manifest_references_required_artifacts():
    """Verify the handover manifest references all required artifacts."""

    manifest_path = FINAL_ARTIFACTS / "manifest.json"

    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    assert manifest["artifact_stage"] == "final"
    assert manifest["model_type"] == "XGBRegressor"

    artifacts = manifest["artifacts"]

    assert artifacts["model"] == (
        "artifacts/final/model/final_model.joblib"
    )

    assert artifacts["preprocessor"] == (
        "artifacts/final/preprocessing/preprocessor.joblib"
    )

    assert artifacts["metrics"] == (
        "artifacts/final/metrics/final_metrics.json"
    )

    assert artifacts["metadata"] == (
        "artifacts/final/metadata/model_metadata.json"
    )


def test_final_metrics_artifact_is_valid():
    """Verify final metrics contain the required metrics."""

    metrics_path = FINAL_ARTIFACTS / "metrics" / "final_metrics.json"

    with metrics_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["primary_metric"] == "rmse"
    assert data["secondary_metric"] == "mae"
    assert data["context_metric"] == "r2"

    metrics = data["metrics"]

    assert set(metrics.keys()) == {"rmse", "mae", "r2"}

    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["r2"], float)


def test_model_metadata_is_valid():
    """Verify final model metadata contains required information."""

    metadata_path = FINAL_ARTIFACTS / "metadata" / "model_metadata.json"

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["model_name"] == "xgboost"
    assert metadata["model_type"] == "XGBRegressor"
    assert metadata["model_stage"] == "final"

    assert "selected_hyperparameters" in metadata
    assert "validation_metrics" in metadata
    assert "artifacts" in metadata

    assert metadata["training_rows"] > 0
    assert metadata["training_features"] > 0
    assert metadata["validation_rows"] > 0