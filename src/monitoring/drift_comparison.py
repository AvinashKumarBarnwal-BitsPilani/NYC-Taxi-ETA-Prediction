"""Compare normal and simulated drift detection results."""

import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"

NORMAL_REPORT_PATH = (
    MONITORING_DIR / "normal_drift_report.json"
)

DRIFTED_REPORT_PATH = (
    MONITORING_DIR / "drifted_drift_report.json"
)

COMPARISON_REPORT_PATH = (
    MONITORING_DIR / "drift_comparison_report.json"
)


def load_report(path: Path) -> dict:
    """Load a drift report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Drift report not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_comparison() -> dict:
    """Compare normal and drifted datasets."""

    normal = load_report(NORMAL_REPORT_PATH)
    drifted = load_report(DRIFTED_REPORT_PATH)

    comparison = {
        "normal": {
            "rows": normal["evaluation"]["current_rows"],
            "overall_drift_detected": (
                normal["overall_drift_detected"]
            ),
            "psi": {
                feature: result["psi"]
                for feature, result
                in normal["psi"].items()
            },
            "ks_distance_km": normal["ks_test"]["distance_km"],
        },
        "drifted": {
            "rows": drifted["evaluation"]["current_rows"],
            "overall_drift_detected": (
                drifted["overall_drift_detected"]
            ),
            "psi": {
                feature: result["psi"]
                for feature, result
                in drifted["psi"].items()
            },
            "ks_distance_km": drifted["ks_test"]["distance_km"],
        },
    }

    with COMPARISON_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
        )

    logger.info(
        "Drift comparison report saved: %s",
        COMPARISON_REPORT_PATH,
    )

    return comparison


def main() -> None:
    """Run drift comparison."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    logger.info("DRIFT COMPARISON STARTED")

    comparison = create_comparison()

    logger.info(
        "Normal drift detected: %s",
        comparison["normal"]["overall_drift_detected"],
    )

    logger.info(
        "Drifted drift detected: %s",
        comparison["drifted"]["overall_drift_detected"],
    )

    logger.info("DRIFT COMPARISON COMPLETED")


if __name__ == "__main__":
    main()