#Data Cleaning Pipeline

# This script transforms the raw TRAIN and TEST datasets into cleaned datasets by applying the data-quality rules defined during Phase 3B.
# The raw datasets are never modified. 
# # Cleaned datasets and an audit summary are generated under data/interim/ for downstream ML processing.

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

TRAIN_INPUT = RAW_DIR / "train.csv"
TEST_INPUT = RAW_DIR / "test.csv"

TRAIN_OUTPUT = INTERIM_DIR / "train_clean.csv"
TEST_OUTPUT = INTERIM_DIR / "test_clean.csv"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
COMMON_DTYPES = { # Data types shared by both TRAIN and TEST.
    "id": "string",
    "vendor_id": "Int64",
    "passenger_count": "Int64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "store_and_fwd_flag": "string",
}

TRAIN_DTYPES = {
    **COMMON_DTYPES, # Reuse common column types.
    "trip_duration": "Int64",
}

# ---------------------------------------------------------------------------
# Load data Function. Will be called from main()
# ---------------------------------------------------------------------------
# Imp Note: pickup_datetime is there in both TRAIN and TEST datasets, but dropoff_datetime is only present in TRAIN dataset. 
# So, we will parse pickup_datetime for both datasets, but dropoff_datetime only for TRAIN dataset.

# The reason why we have not added pickup_datetime in COMMON_DTYPES & dropoff_datetime in TRAIN_DTYPES is because 
# datetime columns are handled separately using the parse_dates argument of pd.read_csv().

def load_data():
    """Load raw TRAIN and TEST datasets with explicit data types."""

    train = pd.read_csv(
        TRAIN_INPUT,
        dtype=TRAIN_DTYPES,
        parse_dates=["pickup_datetime", "dropoff_datetime"], # parse_dates argument ensures that the datetime columns are read as datetime objects instead of strings.
    )

    test = pd.read_csv(
        TEST_INPUT,
        dtype=COMMON_DTYPES,
        parse_dates=["pickup_datetime"],
    )

    return train, test

# ------------------------------------------------------------------------------------------------------
#  ************* Cleaning Function is also called from main() for both TRAIN and TEST datasets *************
# ------------------------------------------------------------------------------------------------------
def clean_dataset(df, dataset_name, has_target=False):
    """
    Apply the Phase 3B cleaning rules and return the cleaned dataset
    together with an audit summary.
    """

    original_count = len(df)

    audit = {
        "original_records": original_count,
        "missing_value_removals": 0,
        "duplicate_removals": 0,
        "invalid_passenger_removals": 0,
        "invalid_coordinate_removals": 0,
        "invalid_timestamp_removals": 0,
        "invalid_target_removals": 0,
        "extreme_outlier_removals": 0,
    }

    # -----------------------------------------------------------------------
    # 1. Missing values: Remove rows with any missing values in any column.
    # -----------------------------------------------------------------------
    missing_mask = df.isnull().any(axis=1)
    audit["missing_value_removals"] = int(missing_mask.sum())

    df = df.loc[~missing_mask].copy()

    # -----------------------------------------------------------------------
    # 2. Duplicate rows: Remove rows that are exact duplicates of other rows.
    # -----------------------------------------------------------------------
    duplicate_mask = df.duplicated()
    audit["duplicate_removals"] = int(duplicate_mask.sum())

    df = df.loc[~duplicate_mask].copy()

    # -----------------------------------------------------------------------
    # 3. Duplicate IDs: Remove rows that have duplicate values in the "id" column, keeping only the first occurrence.
    # -----------------------------------------------------------------------
    duplicate_id_mask = df["id"].duplicated(keep="first")
    duplicate_id_count = int(duplicate_id_mask.sum())

    audit["duplicate_removals"] += duplicate_id_count
    df = df.loc[~duplicate_id_mask].copy()

    # -----------------------------------------------------------------------
    # 4. Invalid passenger counts: Remove rows with invalid passenger counts. <= 0.
    # -----------------------------------------------------------------------
    invalid_passenger_mask = df["passenger_count"] <= 0
    audit["invalid_passenger_removals"] = int(
        invalid_passenger_mask.sum()
    )

    df = df.loc[~invalid_passenger_mask].copy()

    # -----------------------------------------------------------------------
    # 5. Invalid geographic coordinates: Remove rows with pickup or dropoff coordinates outside the broad NYC geographic boundary.
    # Broad NYC geographic boundary:
        # Longitude: -75 to -72
        # Latitude : 40 to 42
    # -----------------------------------------------------------------------
    invalid_coordinate_mask = (
        ~df["pickup_longitude"].between(-75, -72)
        | ~df["pickup_latitude"].between(40, 42)
        | ~df["dropoff_longitude"].between(-75, -72)
        | ~df["dropoff_latitude"].between(40, 42)
    )

    audit["invalid_coordinate_removals"] = int(
        invalid_coordinate_mask.sum()
    )

    df = df.loc[~invalid_coordinate_mask].copy()

    # -----------------------------------------------------------------------
    # 6. Impossible timestamps: Remove rows where pickup datetime is after dropoff datetime.
    # # TRAIN only: For TEST, this step is skipped since dropoff_datetime is not available.
    # -----------------------------------------------------------------------
    if has_target:

        invalid_timestamp_mask = (
            df["pickup_datetime"] >= df["dropoff_datetime"]
        )

        audit["invalid_timestamp_removals"] = int(
            invalid_timestamp_mask.sum()
        )

        df = df.loc[~invalid_timestamp_mask].copy()

    # -----------------------------------------------------------------------
    # 7. Invalid target values: Remove rows with invalid trip_duration values (<= 0 or not equal to the difference between dropoff and pickup datetimes).
    # TRAIN only. For TEST, this step is skipped since trip_duration is not available.
    # -----------------------------------------------------------------------
    if has_target:
        calculated_duration = (
            df["dropoff_datetime"] - df["pickup_datetime"]
        ).dt.total_seconds()

        invalid_target_mask = (
            (df["trip_duration"] <= 0)
            | (df["trip_duration"] != calculated_duration)
        )

        audit["invalid_target_removals"] = int(
            invalid_target_mask.sum()
        )

        df = df.loc[~invalid_target_mask].copy()

     # ---------------------------------------------------------------
     # 8. Extreme outliers: Remove rows with trip_duration greater than 24 hours (86,400 seconds).
     # Remove clearly unrealistic trips > 24 hours.

     # TRAIN only: For TEST, this step is skipped since trip_duration is not available.
     # ---------------------------------------------------------------
    if has_target:

        extreme_outlier_mask = df["trip_duration"] > 86400

        audit["extreme_outlier_removals"] = int(
                extreme_outlier_mask.sum()
            )

        df = df.loc[~extreme_outlier_mask].copy()

    # -----------------------------------------------------------------------
    # Final count: Record the final count of records after cleaning and the total number of records removed.
    # -----------------------------------------------------------------------
    audit["final_records"] = len(df)
    audit["total_removed"] = original_count - len(df)

    return df, audit

# ---------------------------------------------------------------------------
# Print audit summary
# ---------------------------------------------------------------------------
def print_summary(dataset_name, audit):
    """Print cleaning audit information."""

    print()
    print("=" * 70)
    print(f"{dataset_name} CLEANING SUMMARY")
    print("=" * 70)

    for key, value in audit.items():
        print(f"{key.replace('_', ' ').title():30}: {value}")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 3B – DATA CLEANING PIPELINE")
    print("=" * 70)

    train, test = load_data()

    print(f"\nRaw TRAIN records: {len(train)}")
    print(f"Raw TEST records : {len(test)}")

    # Clean TRAIN
    train_clean, train_audit = clean_dataset(
        train,
        dataset_name="TRAIN",
        has_target=True,
    )

    # Clean TEST
    test_clean, test_audit = clean_dataset(
        test,
        dataset_name="TEST",
        has_target=False,
    )

    # Save cleaned datasets
    train_clean.to_csv(TRAIN_OUTPUT, index=False)
    test_clean.to_csv(TEST_OUTPUT, index=False)

    # Print summaries
    print_summary("TRAIN", train_audit)
    print_summary("TEST", test_audit)

    print()
    print("=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)

    print(f"Clean TRAIN dataset: {TRAIN_OUTPUT}")
    print(f"Clean TEST dataset : {TEST_OUTPUT}")


if __name__ == "__main__":
    main()