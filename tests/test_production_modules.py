"""
7.3b – Test Production Modules

These smoke tests verify that the reusable production modules can be imported
successfully and expose their expected public functions.

The tests check the structural health of the production modules, including:
- Cleaning
- Data contract validation
- Feature engineering
- Train/validation splitting
- Preprocessing

These tests do not execute the full data-processing pipeline or validate
business logic. Detailed behavioral and data-invariant tests are covered
in the subsequent 7.3 steps.
"""

"""Smoke tests for reusable production modules."""

from src.ingestion.cleaning import clean_raw_data, log_summary
from src.validation.validate_contract import (
    validate_datasets,
    validate_raw_data,
)
from src.features.feature_engineering import (
    engineer_features,
    engineer_train_test,
)
from src.pipelines.train_validation_split import (
    split_train_validation,
)
from src.pipelines.preprocessing import (
    preprocess_train_validation,
)


def test_cleaning_module_imports():
    """Verify the production cleaning module exposes its public functions."""
    assert callable(clean_raw_data)
    assert callable(log_summary)


def test_validation_module_imports():
    """Verify the production validation module exposes its public functions."""
    assert callable(validate_raw_data)
    assert callable(validate_datasets)


def test_feature_engineering_module_imports():
    """Verify the production feature-engineering module exposes its public functions."""
    assert callable(engineer_features)
    assert callable(engineer_train_test)


def test_train_validation_split_module_imports():
    """Verify the production split module exposes its public function."""
    assert callable(split_train_validation)


def test_preprocessing_module_imports():
    """Verify the production preprocessing module exposes its public function."""
    assert callable(preprocess_train_validation)