"""Threshold-based alert evaluation for model monitoring."""

import logging
from typing import Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"

DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"
PERFORMANCE_REPORT_PATH = MONITORING_DIR / "performance_report.json"
BASELINE_METRICS_PATH = MONITORING_DIR / "baseline_metrics.json"
ALERT_REPORT_PATH = MONITORING_DIR / "alert_report.json"


PSI_WARNING_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25
MAE_DEGRADATION_THRESHOLD = 0.20


def classify_psi_alert(psi: float) -> str:
    """Classify PSI value into an alert severity."""

    if psi > PSI_ALERT_THRESHOLD:
        return "ALERT"

    if psi >= PSI_WARNING_THRESHOLD:
        return "WARNING"

    return "NO_ALERT"


def evaluate_mae_alert(
    current_mae: float,
    baseline_mae: float,
) -> dict[str, Any]:
    """Evaluate MAE degradation against the established baseline."""

    if baseline_mae <= 0:
        raise ValueError("Baseline MAE must be greater than zero.")

    degradation = (
        (current_mae - baseline_mae)
        / baseline_mae
    )

    return {
        "baseline_mae": baseline_mae,
        "current_mae": current_mae,
        "degradation": degradation,
        "threshold": MAE_DEGRADATION_THRESHOLD,
        "alert": degradation > MAE_DEGRADATION_THRESHOLD,
    }


def evaluate_alerts(
    drift_report: dict[str, Any],
    performance_report: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate drift and performance monitoring alerts."""

    psi_alerts = {}

    for feature, result in drift_report["psi"].items():
        psi = result["psi"]

        psi_alerts[feature] = {
            "psi": psi,
            "severity": classify_psi_alert(psi),
        }

    current_mae = performance_report["metrics"]["mae"]
    baseline_mae = baseline_metrics["mae"]

    mae_result = evaluate_mae_alert(
        current_mae=current_mae,
        baseline_mae=baseline_mae,
    )

    psi_alert_triggered = any(
        result["severity"] == "ALERT"
        for result in psi_alerts.values()
    )

    overall_alert = (
        psi_alert_triggered
        or mae_result["alert"]
    )

    return {
        "psi_alerts": psi_alerts,
        "mae": mae_result,
        "overall_alert": overall_alert,
    }


def load_json_report(path: Path) -> dict[str, Any]:
    """Load a monitoring JSON report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Monitoring report not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

    
def generate_alert_report() -> dict[str, Any]:
    """Generate and persist the combined alert report."""

    drift_report = load_json_report(DRIFT_REPORT_PATH)
    performance_report = load_json_report(
        PERFORMANCE_REPORT_PATH
    )
    baseline_metrics = load_json_report(
        BASELINE_METRICS_PATH
    )

    alert_result = evaluate_alerts(
        drift_report=drift_report,
        performance_report=performance_report,
        baseline_metrics=baseline_metrics,
    )

    report = {
        "thresholds": {
            "psi_warning": PSI_WARNING_THRESHOLD,
            "psi_alert": PSI_ALERT_THRESHOLD,
            "mae_degradation": MAE_DEGRADATION_THRESHOLD,
        },
        **alert_result,
    }

    with ALERT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "Alert report saved: %s",
        ALERT_REPORT_PATH,
    )

    return report


def main() -> None:
    """Generate the monitoring alert report."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    logger.info("ALERT EVALUATION STARTED")
    report = generate_alert_report()

    logger.info(
        "Overall alert: %s",
        report["overall_alert"],
    )
    logger.info("ALERT EVALUATION COMPLETED")

if __name__ == "__main__":
    main()