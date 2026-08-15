"""Reusable data-contract validation module for the NYC Taxi project.

The YAML data contract is the single source of truth for schema and
validation rules. This module validates raw TRAIN and TEST datasets
without modifying or deleting any records.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_PATH = PROJECT_ROOT / "configs" / "data_contract.yaml"

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"


# ---------------------------------------------------------------------------
# Contract loading
# ---------------------------------------------------------------------------

def load_contract(
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and return the YAML data-validation contract."""

    with contract_path.open("r", encoding="utf-8") as file:
        contract = yaml.safe_load(file)

    if not contract:
        raise ValueError("Data contract is empty or invalid.")

    return contract


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_datasets(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw TRAIN and TEST datasets using contract-defined datetime fields."""

    train = pd.read_csv(
        train_path,
        parse_dates=["pickup_datetime", "dropoff_datetime"],
    )

    test = pd.read_csv(
        test_path,
        parse_dates=["pickup_datetime"],
    )

    return train, test


# ---------------------------------------------------------------------------
# Validation result helper
# ---------------------------------------------------------------------------

def add_result(
    results: list[dict[str, Any]],
    dataset: str,
    rule: str,
    invalid_count: int,
) -> None:
    """Append one validation result to the report."""

    results.append(
        {
            "dataset": dataset,
            "rule": rule,
            "invalid_records": int(invalid_count),
            "status": "PASS" if invalid_count == 0 else "FAIL",
        }
    )


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    dataset_name: str,
    contract: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Validate that all contract-required columns are present."""

    required_columns = contract["datasets"][
        dataset_name.lower()
    ]["required_columns"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    add_result(
        results,
        dataset_name,
        "Required columns present",
        len(missing_columns),
    )


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def validate_missing_values(
    df: pd.DataFrame,
    dataset_name: str,
    contract: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Validate the contract's required no-null rule."""

    rule = contract["missing_value_rules"]

    if not rule.get("required_columns_must_not_be_null", False):
        return

    required_columns = contract["datasets"][
        dataset_name.lower()
    ]["required_columns"]

    available_required_columns = [
        column for column in required_columns
        if column in df.columns
    ]

    missing_records = (
        df[available_required_columns]
        .isnull()
        .any(axis=1)
        .sum()
    )

    add_result(
        results,
        dataset_name,
        "Required columns contain no null values",
        missing_records,
    )


# ---------------------------------------------------------------------------
# Categorical rules
# ---------------------------------------------------------------------------

def validate_categorical_rules(
    df: pd.DataFrame,
    dataset_name: str,
    contract: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Validate categorical columns against allowed contract values."""

    for column, rule in contract["categorical_rules"].items():

        if column not in df.columns:
            continue

        allowed_values = rule["allowed_values"]

        invalid_count = (
            ~df[column].isin(allowed_values)
        ).sum()

        add_result(
            results,
            dataset_name,
            f"{column} contains only allowed values",
            invalid_count,
        )


# ---------------------------------------------------------------------------
# Numeric rules
# ---------------------------------------------------------------------------

def validate_numeric_rules(
    df: pd.DataFrame,
    dataset_name: str,
    contract: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Validate numeric ranges defined in the contract."""

    for column, rule in contract["numeric_rules"].items():

        if column not in df.columns:
            continue

        if "min_exclusive" in rule:
            invalid_count = (
                df[column] <= rule["min_exclusive"]
            ).sum()

            add_result(
                results,
                dataset_name,
                f"{column} > {rule['min_exclusive']}",
                invalid_count,
            )

        if "min" in rule and "max" in rule:
            invalid_count = (
                ~df[column].between(
                    rule["min"],
                    rule["max"],
                )
            ).sum()

            add_result(
                results,
                dataset_name,
                f"{column} within [{rule['min']}, {rule['max']}]",
                invalid_count,
            )


# ---------------------------------------------------------------------------
# Datetime rules
# ---------------------------------------------------------------------------

def validate_datetime_rules(
    train: pd.DataFrame,
    contract: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Validate TRAIN datetime rules defined in the contract."""

    datetime_rules = contract["datetime_rules"]

    if datetime_rules.get("chronological_order", {}).get("rule"):
        invalid_count = (
            train["pickup_datetime"]
            >= train["dropoff_datetime"]
        ).sum()

        add_result(
            results,
            "TRAIN",
            datetime_rules["chronological_order"]["rule"],
            invalid_count,
        )


# ---------------------------------------------------------------------------
# Target rules
# ---------------------------------------------------------------------------

def validate_target_rules(
    train: pd.DataFrame,
    contract: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Validate TRAIN target rules defined in the contract."""

    target_column = contract["target_rules"]["target_column"]

    target_rules = contract["target_rules"]["rules"]

    if f"{target_column} > 0" in target_rules:
        invalid_count = (
            train[target_column] <= 0
        ).sum()

        add_result(
            results,
            "TRAIN",
            f"{target_column} > 0",
            invalid_count,
        )

    calculated_duration = (
        train["dropoff_datetime"]
        - train["pickup_datetime"]
    ).dt.total_seconds()

    duration_match_rule = (
        f"{target_column} == "
        "(dropoff_datetime - pickup_datetime) in seconds"
    )

    if duration_match_rule in target_rules:
        invalid_count = (
            train[target_column] != calculated_duration
        ).sum()

        add_result(
            results,
            "TRAIN",
            duration_match_rule,
            invalid_count,
        )


# ---------------------------------------------------------------------------
# Duplicate rules
# ---------------------------------------------------------------------------

def validate_duplicates(
    df: pd.DataFrame,
    dataset_name: str,
    contract: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Validate duplicate ID and complete-row rules from the contract."""

    duplicate_rules = contract["duplicate_rules"]

    if duplicate_rules["id"].get("must_be_unique", False):
        duplicate_ids = df["id"].duplicated().sum()

        add_result(
            results,
            dataset_name,
            "IDs are unique",
            duplicate_ids,
        )

    if not duplicate_rules["complete_rows"].get(
        "duplicates_allowed",
        True,
    ):
        duplicate_rows = df.duplicated().sum()

        add_result(
            results,
            dataset_name,
            "No duplicate rows",
            duplicate_rows,
        )


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------

def validate_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    """Validate TRAIN and TEST against the complete data contract."""

    results: list[dict[str, Any]] = []

    datasets = [
        ("TRAIN", train),
        ("TEST", test),
    ]

    for dataset_name, df in datasets:
        validate_required_columns(
            df,
            dataset_name,
            contract,
            results,
        )

        validate_missing_values(
            df,
            dataset_name,
            contract,
            results,
        )

        validate_categorical_rules(
            df,
            dataset_name,
            contract,
            results,
        )

        validate_numeric_rules(
            df,
            dataset_name,
            contract,
            results,
        )

        validate_duplicates(
            df,
            dataset_name,
            contract,
            results,
        )

    validate_datetime_rules(
        train,
        contract,
        results,
    )

    validate_target_rules(
        train,
        contract,
        results,
    )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Public validation entry point
# ---------------------------------------------------------------------------

def validate_raw_data(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    contract_path: Path = CONTRACT_PATH,
) -> pd.DataFrame:
    """Load contract and raw datasets, then return the validation report."""

    contract = load_contract(contract_path)
    train, test = load_datasets(train_path, test_path)

    return validate_datasets(
        train,
        test,
        contract,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def log_validation_report(results_df: pd.DataFrame) -> bool:
    """Log validation results and return True when all rules pass."""

    logger.info("DATA CONTRACT VALIDATION REPORT")

    for _, row in results_df.iterrows():
        logger.info(
            "%s | %s | invalid_records=%d | %s",
            row["dataset"],
            row["rule"],
            int(row["invalid_records"]),
            row["status"],
        )

    failed = int((results_df["status"] == "FAIL").sum())

    if failed == 0:
        logger.info("RESULT: PASS")
        logger.info("All implemented data-contract rules passed.")
    else:
        logger.error("RESULT: FAIL - Failed rules: %d", failed)

    return failed == 0


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run data-contract validation using the project's default paths."""

    logger.info("DATA CONTRACT VALIDATION STARTED")

    try:
        results_df = validate_raw_data()

        passed = log_validation_report(results_df)

        if not passed:
            raise ValueError("Data contract validation failed.")

        logger.info("DATA CONTRACT VALIDATION COMPLETED")

    except Exception:
        logger.exception("Data contract validation failed")
        raise


if __name__ == "__main__":
    main()
