# Validate PSI, KS test calculations and configured drift thresholds.
import pandas as pd
import pytest

from src.monitoring.drift_detection import (
    calculate_psi,
    calculate_ks_test,
    is_psi_drift,
    is_ks_drift,
)


def test_psi_identical_distributions():
    """Identical distributions should have zero PSI."""

    baseline = [0.5, 0.3, 0.2]
    current = [0.5, 0.3, 0.2]

    psi = calculate_psi(
        baseline,
        current,
    )

    assert psi == pytest.approx(0.0)


def test_psi_detects_distribution_change():
    """A significantly changed distribution should have positive PSI."""

    baseline = [0.8, 0.15, 0.05]
    current = [0.1, 0.2, 0.7]

    psi = calculate_psi(
        baseline,
        current,
    )

    assert psi > 0.20


def test_ks_test_returns_valid_result():
    """KS test should return statistic and p-value."""

    baseline = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    current = pd.Series(
        [1.1, 2.1, 3.1, 4.1, 5.1]
    )

    result = calculate_ks_test(
        baseline,
        current,
    )

    assert 0.0 <= result["ks_statistic"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0


def test_psi_threshold_classification():
    """PSI threshold should classify drift correctly."""

    assert is_psi_drift(0.10) is False
    assert is_psi_drift(0.19) is False
    assert is_psi_drift(0.20) is True
    assert is_psi_drift(0.50) is True


def test_ks_threshold_classification():
    """KS p-value threshold should classify drift correctly."""

    assert is_ks_drift(0.10) is False
    assert is_ks_drift(0.05) is False
    assert is_ks_drift(0.049) is True
    assert is_ks_drift(0.01) is True