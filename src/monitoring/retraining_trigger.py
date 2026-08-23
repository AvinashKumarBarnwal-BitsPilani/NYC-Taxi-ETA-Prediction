# Retraining is considered only when significant feature drift
# or meaningful model-performance degradation is observed.
# These thresholds align with the alerting thresholds in Phase 6.5.

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"

DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"
PERFORMANCE_REPORT_PATH = MONITORING_DIR / "performance_report.json"
BASELINE_METRICS_PATH = MONITORING_DIR / "baseline_metrics.json"

RETRAINING_TRIGGER_REPORT_PATH = (
    MONITORING_DIR / "retraining_trigger_report.json"
)

RETRAINING_PSI_THRESHOLD = 0.25
RETRAINING_MAE_DEGRADATION_THRESHOLD = 0.20

from typing import Any


def evaluate_retraining_trigger(
    drift_report: dict[str, Any],
    performance_report: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Determine whether the model is a retraining candidate."""

    drifted_features = [
        feature
        for feature, result in drift_report["psi"].items()
        if result["psi"] > RETRAINING_PSI_THRESHOLD
    ]

    current_mae = performance_report["metrics"]["mae"]
    baseline_mae = baseline_metrics["mae"]

    if baseline_mae <= 0:
        raise ValueError("Baseline MAE must be greater than zero.")

    mae_degradation = (
        (current_mae - baseline_mae)
        / baseline_mae
    )

    performance_degraded = (
        mae_degradation
        > RETRAINING_MAE_DEGRADATION_THRESHOLD
    )

    retraining_candidate = (
        bool(drifted_features)
        or performance_degraded
    )

    return {
        "retraining_candidate": retraining_candidate,
        "drift_trigger": {
            "triggered": bool(drifted_features),
            "threshold": RETRAINING_PSI_THRESHOLD,
            "features": drifted_features,
        },
        "performance_trigger": {
            "triggered": performance_degraded,
            "baseline_mae": baseline_mae,
            "current_mae": current_mae,
            "degradation": mae_degradation,
            "threshold": RETRAINING_MAE_DEGRADATION_THRESHOLD,
        },
    }

def load_json_report(path: Path) -> dict[str, Any]:
    """Load a monitoring JSON report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Monitoring report not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_retraining_trigger_report() -> dict[str, Any]:
    """Generate and persist the retraining trigger report."""

    drift_report = load_json_report(DRIFT_REPORT_PATH)
    performance_report = load_json_report(
        PERFORMANCE_REPORT_PATH
    )
    baseline_metrics = load_json_report(
        BASELINE_METRICS_PATH
    )

    decision = evaluate_retraining_trigger(
        drift_report=drift_report,
        performance_report=performance_report,
        baseline_metrics=baseline_metrics,
    )

    report = {
        "thresholds": {
            "psi": RETRAINING_PSI_THRESHOLD,
            "mae_degradation": (
                RETRAINING_MAE_DEGRADATION_THRESHOLD
            ),
        },
        **decision,
    }

    with RETRAINING_TRIGGER_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "Retraining trigger report saved: %s",
        RETRAINING_TRIGGER_REPORT_PATH,
    )

    return report


def main() -> None:
    """Evaluate and report the retraining decision."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    logger.info("RETRAINING TRIGGER EVALUATION STARTED")
    report = generate_retraining_trigger_report()

    logger.info(
        "Retraining candidate: %s",
        report["retraining_candidate"],
    )
    logger.info("RETRAINING TRIGGER EVALUATION COMPLETED")


if __name__ == "__main__":
    main()