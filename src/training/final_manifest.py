"""Phase 4.7.5 - Final model handover manifest."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


FINAL_ARTIFACTS = {
    "model": "artifacts/final/model/final_model.joblib",
    "preprocessor": "artifacts/final/preprocessing/preprocessor.joblib",
    "metrics": "artifacts/final/metrics/final_metrics.json",
    "metadata": "artifacts/final/metadata/model_metadata.json",
}


def create_handover_manifest(
    output_path: str = "artifacts/final/manifest.json",
) -> Path:
    """Create the final handover manifest after validating artifacts."""

    logger.info("Validating final handover artifacts")

    missing_artifacts = []

    for artifact_name, artifact_path in FINAL_ARTIFACTS.items():
        path = Path(artifact_path)

        if not path.exists():
            missing_artifacts.append(
                f"{artifact_name}: {artifact_path}"
            )
        else:
            logger.info(
                "Handover artifact found - %s: %s",
                artifact_name,
                path,
            )

    if missing_artifacts:
        raise FileNotFoundError(
            "Required handover artifacts are missing:\n"
            + "\n".join(missing_artifacts)
        )

    manifest = {
        "artifact_stage": "final",
        "model_type": "XGBRegressor",
        "artifacts": FINAL_ARTIFACTS,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    logger.info("Handover manifest persisted: %s", path)

    return path


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("PHASE 4.7.5 - CREATE HANDOVER MANIFEST")
    logger.info("=" * 70)

    manifest_path = create_handover_manifest()

    logger.info("Handover manifest created: %s", manifest_path)

    logger.info("=" * 70)
    logger.info("PHASE 4.7.5 COMPLETED")
    logger.info("=" * 70)