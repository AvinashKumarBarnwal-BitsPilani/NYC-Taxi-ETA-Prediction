"""Phase 4.1 modeling strategy and data-contract verification."""

from src.training.data_contract import (
    load_modeling_config,
    load_training_data,
    validate_data_contract,
    validate_modeling_config,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_modeling_strategy() -> None:
    """Validate the Phase 4 modeling strategy and data contract."""

    logger.info("=" * 70)
    logger.info("PHASE 4.1 - MODELING STRATEGY & DATA CONTRACT")
    logger.info("=" * 70)

    config = load_modeling_config()

    validate_modeling_config(config)

    logger.info(
        "Problem type: %s",
        config["problem"]["type"],
    )

    logger.info(
        "Target: %s",
        config["problem"]["target"],
    )

    logger.info(
        "Primary metric: %s",
        config["metrics"]["primary"],
    )

    logger.info(
        "Secondary metric: %s",
        config["metrics"]["secondary"],
    )

    logger.info(
        "Context metric: %s",
        config["metrics"]["context"],
    )

    X_train, X_val, y_train, y_val = load_training_data(config)

    validate_data_contract(
        X_train,
        X_val,
        y_train,
        y_val,
        config,
    )

    logger.info(
        "Expected model features: %s",
        config["features"],
    )

    logger.info("=" * 70)
    logger.info("PHASE 4.1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_modeling_strategy()