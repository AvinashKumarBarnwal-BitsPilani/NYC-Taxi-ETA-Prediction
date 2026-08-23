"""Feature drift baseline and detection utilities."""

# Build the drift baseline from X_train by capturing the reference distribution of all monitored features. 
# Store value proportions for discrete/categorical features and quantile-based bin distributions for continuous distance_km.

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "split" / "X_train.csv"
)

MONITORING_DIR = (
    PROJECT_ROOT / "data" / "monitoring"
)

DRIFT_BASELINE_PATH = (
    MONITORING_DIR / "drift_baseline.json"
)

DRIFT_REPORT_PATH = (
    MONITORING_DIR / "drift_report.json"
)

MONITORED_FEATURES = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "distance_km",
]

CATEGORICAL_FEATURES = [
    "vendor_id",
    "store_and_fwd_flag",
]

DISCRETE_FEATURES = [
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
]

CONTINUOUS_FEATURES = [
    "distance_km",
]


# Drift thresholds used to classify feature distribution changes.
# Alerting and automated actions are handled in later phases.

# 0.20 PSI → drift detection threshold
# 0.10 / 0.25 PSI → alert severity thresholds
# 20% MAE degradation → performance alert threshold
# 0.05 KS p-value → statistical drift detection threshold

PSI_DRIFT_THRESHOLD = 0.20
PSI_WARNING_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

MAE_DEGRADATION_THRESHOLD = 0.20

KS_P_VALUE_THRESHOLD = 0.05

def validate_baseline_features(
    df: pd.DataFrame,
) -> None:
    """Validate that all monitored features are present."""

    missing_features = [
        feature
        for feature in MONITORED_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required drift features: "
            f"{missing_features}"
        )


def build_discrete_distribution(
    series: pd.Series,
) -> dict[str, float]:
    """Build value-to-proportion distribution."""

    proportions = (
        series.value_counts(normalize=True)
        .sort_index()
    )

    return {
        str(value): float(proportion)
        for value, proportion in proportions.items()
    }


def build_continuous_distribution(
    series: pd.Series,
    bins: int = 10,
) -> dict:
    """Build quantile-based distribution for continuous data."""

    clean_series = series.dropna()

    _, bin_edges = pd.qcut(
        clean_series,
        q=bins,
        retbins=True,
        duplicates="drop",
    )

    bin_counts = pd.cut(
        clean_series,
        bins=bin_edges,
        include_lowest=True,
    ).value_counts(
        normalize=True,
        sort=False,
    )

    return {
        "bin_edges": [
            float(edge)
            for edge in bin_edges
        ],
        "proportions": [
            float(proportion)
            for proportion in bin_counts
        ],
    }


def create_drift_baseline() -> dict:
    """Create the reference feature distributions from X_train."""

    logger.info(
        "Creating drift baseline from TRAIN data: %s",
        TRAIN_DATA_PATH,
    )

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: "
            f"{TRAIN_DATA_PATH}"
        )

    df = pd.read_csv(TRAIN_DATA_PATH)

    logger.info(
        "Drift baseline source loaded - shape=%s",
        df.shape,
    )

    if df.empty:
        raise ValueError(
            "Drift baseline source cannot be empty."
        )

    validate_baseline_features(df)

    distributions = {}

    for feature in (
        CATEGORICAL_FEATURES
        + DISCRETE_FEATURES
    ):
        distributions[feature] = {
            "type": "discrete",
            "distribution": build_discrete_distribution(
                df[feature]
            ),
        }

    for feature in CONTINUOUS_FEATURES:
        distributions[feature] = {
            "type": "continuous",
            "distribution": build_continuous_distribution(
                df[feature]
            ),
        }

    baseline = {
        "source": str(TRAIN_DATA_PATH),
        "rows": len(df),
        "features": MONITORED_FEATURES,
        "distributions": distributions,
    }

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DRIFT_BASELINE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            baseline,
            file,
            indent=4,
        )

    logger.info(
        "Drift baseline saved: %s",
        DRIFT_BASELINE_PATH,
    )

    return baseline


PREDICTION_LOG_PATH = (
    MONITORING_DIR / "prediction_logs.csv"
)

CURRENT_DATA_PATH = (
    MONITORING_DIR / "current_production_data.csv"
)


def prepare_current_production_data() -> pd.DataFrame:
    """Prepare current production features for drift analysis."""

    logger.info(
        "Preparing current production data from: %s",
        PREDICTION_LOG_PATH,
    )

    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found: "
            f"{PREDICTION_LOG_PATH}"
        )

    df = pd.read_csv(PREDICTION_LOG_PATH)

    logger.info(
        "Prediction logs loaded - shape=%s",
        df.shape,
    )

    if df.empty:
        raise ValueError(
            "Current production data cannot be empty."
        )

    validate_baseline_features(df)

    current_data = df[MONITORED_FEATURES].copy()

    logger.info(
    "Current production records available for drift analysis: %d",
    len(current_data),
    )

    if current_data.empty:
        raise ValueError(
            "Current production feature data cannot be empty."
        )

    current_data.to_csv(
        CURRENT_DATA_PATH,
        index=False,
    )

    logger.info(
        "Current production data saved: %s",
        CURRENT_DATA_PATH,
    )

    logger.info(
        "Current production feature data shape=%s",
        current_data.shape,
    )

    return current_data


# Compare baseline and current feature distributions using PSI.
# PSI provides the drift magnitude; drift thresholds are evaluated later.
def calculate_psi(
    baseline_proportions: list[float],
    current_proportions: list[float],
) -> float:
    """Calculate Population Stability Index (PSI)."""

    baseline = np.asarray(
        baseline_proportions,
        dtype=float,
    )

    current = np.asarray(
        current_proportions,
        dtype=float,
    )

    if baseline.shape != current.shape:
        raise ValueError(
            "Baseline and current distributions must "
            "have the same shape."
        )

    epsilon = 1e-6

    baseline = np.clip(
        baseline,
        epsilon,
        None,
    )

    current = np.clip(
        current,
        epsilon,
        None,
    )

    return float(
        np.sum(
            (current - baseline)
            * np.log(current / baseline)
        )
    )


def calculate_discrete_psi(
    baseline_distribution: dict[str, float],
    current_series: pd.Series,
) -> float:
    """Calculate PSI for a discrete/categorical feature."""

    current_distribution = {
    str(value): float(proportion)
    for value, proportion in (
        current_series.value_counts(
            normalize=True
        ).items()
    )
    }

    categories = set(
        baseline_distribution.keys()
    ) | {
        str(value)
        for value in current_distribution.keys()
    }

    baseline_proportions = []
    current_proportions = []

    for category in sorted(categories):
        baseline_proportions.append(
            float(
                baseline_distribution.get(
                    category,
                    0.0,
                )
            )
        )

        current_proportions.append(
            float(
                current_distribution.get(
                    category,
                    0.0,
                )
            )
        )

    return calculate_psi(
        baseline_proportions,
        current_proportions,
    )


def calculate_continuous_psi(
    baseline_distribution: dict,
    current_series: pd.Series,
) -> float:
    """Calculate PSI using baseline bins for a continuous feature (distance_km)."""

    bin_edges = np.asarray(
        baseline_distribution["bin_edges"],
        dtype=float,
    )

    current_values = current_series.dropna().to_numpy()

    if len(current_values) == 0:
        raise ValueError(
            "Current continuous feature contains no valid values."
        )

    # Assign values outside the baseline range to
    # the first/last baseline bin.
    bin_indices = np.digitize(
        current_values,
        bin_edges[1:-1],
        right=False,
    )

    bin_indices = np.clip(
        bin_indices,
        0,
        len(bin_edges) - 2,
    )

    current_counts = np.bincount(
        bin_indices,
        minlength=len(bin_edges) - 1,
    )

    current_proportions = (
        current_counts / len(current_values)
    )

    baseline_proportions = np.asarray(
        baseline_distribution["proportions"],
        dtype=float,
    )

    return calculate_psi(
        baseline_proportions.tolist(),
        current_proportions.tolist(),
    )


def calculate_feature_psi(
    baseline: dict,
    current_data: pd.DataFrame,
) -> dict[str, float]:
    """Calculate PSI for all monitored features."""

    psi_results = {}

    for feature in MONITORED_FEATURES:
        feature_config = baseline[
            "distributions"
        ][feature]

        if feature_config["type"] == "discrete":
            psi_results[feature] = (
                calculate_discrete_psi(
                    feature_config["distribution"],
                    current_data[feature],
                )
            )

        elif feature_config["type"] == "continuous":
            psi_results[feature] = (
                calculate_continuous_psi(
                    feature_config["distribution"],
                    current_data[feature],
                )
            )

        else:
            raise ValueError(
                f"Unsupported feature type for "
                f"{feature}: "
                f"{feature_config['type']}"
            )

    return psi_results


def run_psi_analysis() -> dict[str, float]:
    """Load baseline/current data and calculate feature PSI."""

    if not DRIFT_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Drift baseline not found: "
            f"{DRIFT_BASELINE_PATH}"
        )

    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Current production data not found: "
            f"{CURRENT_DATA_PATH}"
        )

    with DRIFT_BASELINE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        baseline = json.load(file)

    current_data = pd.read_csv(
        CURRENT_DATA_PATH
    )

    validate_baseline_features(
        current_data
    )

    psi_results = calculate_feature_psi(
        baseline,
        current_data,
    )

    for feature, psi in psi_results.items():
        logger.info(
            "PSI - %s: %.6f",
            feature,
            psi,
        )

    return psi_results


# Compare baseline and current distance_km distributions using
# the two-sample KS test. Threshold interpretation is handled later.
def calculate_ks_test(
    baseline_series: pd.Series,
    current_series: pd.Series,
) -> dict[str, float]:
    """Run KS two-sample test for a continuous feature."""

    baseline_values = (
        baseline_series
        .dropna()
        .to_numpy()
    )

    current_values = (
        current_series
        .dropna()
        .to_numpy()
    )

    if len(baseline_values) == 0:
        raise ValueError(
            "Baseline continuous feature contains no valid values."
        )

    if len(current_values) == 0:
        raise ValueError(
            "Current continuous feature contains no valid values."
        )

    statistic, p_value = ks_2samp(
        baseline_values,
        current_values,
    )

    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
    }


def run_ks_analysis() -> dict[str, float]:
    """Run KS test for distance_km."""

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: "
            f"{TRAIN_DATA_PATH}"
        )

    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Current production data not found: "
            f"{CURRENT_DATA_PATH}"
        )

    baseline_data = pd.read_csv(
        TRAIN_DATA_PATH
    )

    current_data = pd.read_csv(
        CURRENT_DATA_PATH
    )

    validate_baseline_features(
        baseline_data
    )

    validate_baseline_features(
        current_data
    )

    ks_result = calculate_ks_test(
        baseline_data["distance_km"],
        current_data["distance_km"],
    )

    logger.info(
        "KS Test - distance_km: statistic=%.6f, p_value=%.6f",
        ks_result["ks_statistic"],
        ks_result["p_value"],
    )

    return ks_result


def is_psi_drift(psi: float) -> bool:
    """Return whether PSI indicates significant drift."""

    return psi >= PSI_DRIFT_THRESHOLD

def is_ks_drift(p_value: float) -> bool:
    """Return whether KS test indicates significant drift."""

    return p_value < KS_P_VALUE_THRESHOLD


# Combine PSI and KS results with configured thresholds into
# a machine-readable drift report for downstream monitoring.
def generate_drift_report(
    psi_results: dict[str, float],
    ks_result: dict[str, float],
) -> dict:
    """Generate and persist the current drift report."""

    psi_report = {
        feature: {
            "psi": float(psi),
            "drift_detected": is_psi_drift(psi),
        }
        for feature, psi in psi_results.items()
    }

    ks_report = {
        "distance_km": {
            "ks_statistic": ks_result["ks_statistic"],
            "p_value": ks_result["p_value"],
            "drift_detected": is_ks_drift(
                ks_result["p_value"]
            ),
        }
    }

    overall_drift_detected = (
        any(
            result["drift_detected"]
            for result in psi_report.values()
        )
        or any(
            result["drift_detected"]
            for result in ks_report.values()
        )
    )

    report = {
        "evaluation": {
            "baseline_source": str(
                TRAIN_DATA_PATH
            ),
            "current_source": str(
                CURRENT_DATA_PATH
            ),
            "current_rows": len(
                pd.read_csv(CURRENT_DATA_PATH)
            ),
        },
        "psi": psi_report,
        "ks_test": ks_report,
        "thresholds": {
            "psi": PSI_DRIFT_THRESHOLD,
            "ks_p_value": KS_P_VALUE_THRESHOLD,
        },
        "overall_drift_detected": overall_drift_detected,
    }

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DRIFT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "Drift report saved: %s",
        DRIFT_REPORT_PATH,
    )

    logger.info(
        "Overall drift detected: %s",
        overall_drift_detected,
    )

    return report


def main() -> None:
    """Create drift baseline and prepare current production data."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    logger.info(
        "DRIFT BASELINE AND CURRENT DATA PREPARATION STARTED"
    )

    create_drift_baseline()
    prepare_current_production_data()

    psi_results = run_psi_analysis()
    ks_result = run_ks_analysis()

    generate_drift_report(
    psi_results,
    ks_result,
    )

    logger.info(
        "DRIFT BASELINE AND CURRENT DATA PREPARATION COMPLETED"
    )


if __name__ == "__main__":
    main()