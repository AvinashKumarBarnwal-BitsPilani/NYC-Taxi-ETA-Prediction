"""
7.3e – End-to-End Pipeline Test

Verifies that the complete Phase 3 production data-engineering pipeline
can execute successfully and produce the expected ML-ready artifacts.

This test invokes the actual production pipeline rather than duplicating
its internal processing logic.
"""

from pathlib import Path
import subprocess
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_PIPELINE_MODULE = "src.pipelines.run_pipeline"

SPLIT_DIR = PROJECT_ROOT / "data" / "split"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_VAL_PATH = SPLIT_DIR / "X_val.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
Y_VAL_PATH = SPLIT_DIR / "y_val.csv"

X_TRAIN_PROCESSED_PATH = (
    PROCESSED_DIR / "X_train_processed.csv"
)

X_VAL_PROCESSED_PATH = (
    PROCESSED_DIR / "X_val_processed.csv"
)


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline():
    """
    Execute the complete production pipeline and verify its outputs.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            RUN_PIPELINE_MODULE,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    # Pipeline must complete successfully.
    assert result.returncode == 0, (
        "End-to-end pipeline failed.\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    # Verify expected pipeline completion message.
    assert (
    "PHASE 3 - END-TO-END DATA ENGINEERING PIPELINE COMPLETED"
    in result.stderr
    )   

    # Verify split artifacts exist.
    expected_split_files = [
        X_TRAIN_PATH,
        X_VAL_PATH,
        Y_TRAIN_PATH,
        Y_VAL_PATH,
    ]

    for path in expected_split_files:
        assert path.exists(), (
            f"Expected split artifact was not created: {path}"
        )

    # Verify processed artifacts exist.
    expected_processed_files = [
        X_TRAIN_PROCESSED_PATH,
        X_VAL_PROCESSED_PATH,
    ]

    for path in expected_processed_files:
        assert path.exists(), (
            f"Expected processed artifact was not created: {path}"
        )

    # Verify final dataset shapes.
    X_train = pd.read_csv(X_TRAIN_PROCESSED_PATH)
    X_val = pd.read_csv(X_VAL_PROCESSED_PATH)

    assert X_train.shape == (1_166_833, 10)
    assert X_val.shape == (291_709, 10)