import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.performance_monitor import (
    calculate_performance_metrics,
    establish_baseline,
)


def test_calculate_performance_metrics():
    """Verify MAE, RMSE, and R2 calculations."""

    df = pd.DataFrame(
        {
            "actual_eta": [100.0, 200.0, 300.0],
            "predicted_eta": [110.0, 190.0, 310.0],
        }
    )

    metrics = calculate_performance_metrics(df)

    expected_mae = 10.0
    expected_rmse = np.sqrt(
        (10.0**2 + 10.0**2 + 10.0**2) / 3
    )

    assert metrics["mae"] == pytest.approx(
        expected_mae
    )

    assert metrics["rmse"] == pytest.approx(
        expected_rmse
    )

    assert metrics["r2"] == pytest.approx(
        0.985
    )


def test_missing_required_columns_fails():
    """Verify required monitoring columns are enforced."""

    df = pd.DataFrame(
        {
            "actual_eta": [100.0, 200.0],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        calculate_performance_metrics(df)


def test_missing_values_fail():
    """Verify missing prediction/actual values are rejected."""

    df = pd.DataFrame(
        {
            "actual_eta": [100.0, np.nan],
            "predicted_eta": [110.0, 190.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing prediction or actual ETA",
    ):
        calculate_performance_metrics(df)


def test_empty_dataset_fails():
    """Verify empty monitoring datasets are rejected."""

    df = pd.DataFrame(
        {
            "actual_eta": [],
            "predicted_eta": [],
        }
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        calculate_performance_metrics(df)


def test_baseline_is_created_once(
    tmp_path,
    monkeypatch,
):
    """Verify an existing baseline is preserved."""

    import src.monitoring.performance_monitor as monitor

    baseline_path = (
        tmp_path / "baseline_metrics.json"
    )

    monkeypatch.setattr(
        monitor,
        "BASELINE_REPORT_PATH",
        baseline_path,
    )

    df = pd.DataFrame(
        {
            "actual_eta": [100.0, 200.0, 300.0],
            "predicted_eta": [110.0, 190.0, 310.0],
        }
    )

    first_baseline = establish_baseline(df)

    assert baseline_path.exists()

    original_content = baseline_path.read_text(
        encoding="utf-8"
    )

    changed_df = pd.DataFrame(
        {
            "actual_eta": [1000.0, 2000.0, 3000.0],
            "predicted_eta": [100.0, 200.0, 300.0],
        }
    )

    second_baseline = establish_baseline(
        changed_df
    )

    assert second_baseline == first_baseline

    assert baseline_path.read_text(
        encoding="utf-8"
    ) == original_content

    stored = json.loads(
        baseline_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored == first_baseline