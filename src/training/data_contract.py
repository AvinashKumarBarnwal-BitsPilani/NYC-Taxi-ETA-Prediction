"""Phase 4 data contract validation for the NYC Taxi ETA project."""

from pathlib import Path

import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELING_CONFIG_PATH = PROJECT_ROOT / "configs" / "modeling.yaml"

def load_modeling_config() -> dict:
    """Load the Phase 4 modeling configuration."""

    logger.info(
        "Loading Phase 4 modeling configuration: %s",
        MODELING_CONFIG_PATH,
    )

    if not MODELING_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Modeling configuration not found: {MODELING_CONFIG_PATH}"
        )

    with MODELING_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return config

def validate_modeling_config(config: dict) -> None:
    """Validate the required Phase 4 modeling configuration."""

    required_problem_keys = {"type", "target"}
    required_metric_keys = {"primary", "secondary", "context"}
    required_data_keys = {
        "train_features",
        "validation_features",
        "train_target",
        "validation_target",
    }

    if "problem" not in config:
        raise KeyError("Missing required configuration section: problem")

    if "features" not in config:
        raise KeyError("Missing required configuration section: features")

    if "metrics" not in config:
        raise KeyError("Missing required configuration section: metrics")

    if "data" not in config:
        raise KeyError("Missing required configuration section: data")

    missing_problem_keys = required_problem_keys - set(config["problem"])
    if missing_problem_keys:
        raise KeyError(
            f"Missing problem configuration keys: {missing_problem_keys}"
        )

    missing_metric_keys = required_metric_keys - set(config["metrics"])
    if missing_metric_keys:
        raise KeyError(
            f"Missing metric configuration keys: {missing_metric_keys}"
        )

    missing_data_keys = required_data_keys - set(config["data"])
    if missing_data_keys:
        raise KeyError(
            f"Missing data configuration keys: {missing_data_keys}"
        )

    if config["problem"]["type"] != "regression":
        raise ValueError(
            "Phase 4 currently supports only regression problems."
        )

    if config["problem"]["target"] != "trip_duration":
        raise ValueError(
            "Phase 4 target must be 'trip_duration'."
        )

    expected_metrics = {
        "primary": "rmse",
        "secondary": "mae",
        "context": "r2",
    }

    if config["metrics"] != expected_metrics:
        raise ValueError(
            "Phase 4 metrics do not match the approved modeling strategy. "
            f"Expected: {expected_metrics}"
        )

    if len(config["features"]) != 5:
        raise ValueError(
            "Phase 4 expects exactly 5 processed model features."
        )

def load_training_data(config: dict) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Load Phase 3 ML-ready training and validation datasets."""

    data_config = config["data"]

    X_train_path = PROJECT_ROOT / data_config["train_features"]
    X_val_path = PROJECT_ROOT / data_config["validation_features"]
    y_train_path = PROJECT_ROOT / data_config["train_target"]
    y_val_path = PROJECT_ROOT / data_config["validation_target"]

    paths = {
        "X_train": X_train_path,
        "X_val": X_val_path,
        "y_train": y_train_path,
        "y_val": y_val_path,
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}"
            )

    logger.info("Loading Phase 3 training and validation datasets")

    X_train = pd.read_csv(X_train_path)
    X_val = pd.read_csv(X_val_path)
    y_train = pd.read_csv(y_train_path)
    y_val = pd.read_csv(y_val_path)

    return X_train, X_val, y_train, y_val


def validate_data_contract(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    config: dict,
) -> None:
    """Validate the Phase 3 → Phase 4 data contract."""

    expected_features = config["features"]
    target_column = config["problem"]["target"]

    logger.info("Validating Phase 3 → Phase 4 data contract")

    # ----------------------------------------------------------
    # Feature columns
    # ----------------------------------------------------------

    if list(X_train.columns) != expected_features:
        raise ValueError(
            "X_train feature columns do not match the Phase 4 contract.\n"
            f"Expected: {expected_features}\n"
            f"Actual:   {list(X_train.columns)}"
        )

    if list(X_val.columns) != expected_features:
        raise ValueError(
            "X_val feature columns do not match the Phase 4 contract.\n"
            f"Expected: {expected_features}\n"
            f"Actual:   {list(X_val.columns)}"
        )

    # ----------------------------------------------------------
    # Target
    # ----------------------------------------------------------

    if list(y_train.columns) != [target_column]:
        raise ValueError(
            "y_train does not contain the expected target column."
        )

    if list(y_val.columns) != [target_column]:
        raise ValueError(
            "y_val does not contain the expected target column."
        )

    # ----------------------------------------------------------
    # Row alignment
    # ----------------------------------------------------------

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train row counts do not match."
        )

    if len(X_val) != len(y_val):
        raise ValueError(
            "X_val and y_val row counts do not match."
        )

    # ----------------------------------------------------------
    # Missing values
    # ----------------------------------------------------------

    if X_train.isna().any().any():
        raise ValueError("X_train contains missing values.")

    if X_val.isna().any().any():
        raise ValueError("X_val contains missing values.")

    if y_train.isna().any().any():
        raise ValueError("y_train contains missing values.")

    if y_val.isna().any().any():
        raise ValueError("y_val contains missing values.")

    # ----------------------------------------------------------
    # Numeric model inputs
    # ----------------------------------------------------------

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in X_train.dtypes):
        raise TypeError(
            "All Phase 4 training features must be numeric."
        )

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in X_val.dtypes):
        raise TypeError(
            "All Phase 4 validation features must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(y_train[target_column]):
        raise TypeError(
            "Training target must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(y_val[target_column]):
        raise TypeError(
            "Validation target must be numeric."
        )

    logger.info(
        "Phase 4 data contract validation PASSED"
    )

    logger.info(
        "Training features: %d rows × %d columns",
        len(X_train),
        X_train.shape[1],
    )

    logger.info(
        "Validation features: %d rows × %d columns",
        len(X_val),
        X_val.shape[1],
    )

    logger.info(
        "Training target: %d rows",
        len(y_train),
    )

    logger.info(
        "Validation target: %d rows",
        len(y_val),
    )