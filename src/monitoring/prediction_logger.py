"""Prediction logging for the NYC Taxi ETA monitoring workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
PREDICTION_LOG_PATH = MONITORING_DIR / "prediction_logs.csv"

LOG_COLUMNS = [
    "timestamp",
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "distance_km",
    "predicted_eta",
]

def log_prediction(
    features: pd.DataFrame,
    predicted_eta: float,
) -> None:
    """Append a successful prediction to the monitoring log."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "Prediction logging input must be a pandas DataFrame."
        )

    if len(features) != 1:
        raise ValueError(
            "Prediction logging expects exactly one prediction record."
        )

    missing_columns = sorted(
        set(LOG_COLUMNS[1:-1]) - set(features.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Prediction logging input is missing columns: "
            f"{missing_columns}"
        )

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = features.iloc[0].to_dict()

    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    row["predicted_eta"] = float(predicted_eta)

    log_record = pd.DataFrame(
        [row],
        columns=LOG_COLUMNS,
    )

    file_exists = PREDICTION_LOG_PATH.exists()

    log_record.to_csv(
        PREDICTION_LOG_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )

    logger.info(
        "Prediction logged successfully: eta=%s",
        predicted_eta,
    )