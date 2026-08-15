# Evaluate existing candidate features (vendor_id, passenger_count, and
# store_and_fwd_flag) by inspecting their distributions and relationship
# with trip_duration to decide whether they should be retained as features.

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INTERIM_DIR = Path("data/interim")
TRAIN_PATH = INTERIM_DIR / "train_clean.csv"

# ---------------------------------------------------------------------------
# Load cleaned TRAIN dataset
# ---------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)

# ---------------------------------------------------------------------------
# 1. Vendor ID
# ---------------------------------------------------------------------------
print("=== Vendor ID Distribution ===")

print(
    train_df["vendor_id"]
    .value_counts()
    .sort_index()
)

print("\n=== Trip Duration by Vendor ===")

print(
    train_df.groupby("vendor_id")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# ---------------------------------------------------------------------------
# 2. Passenger Count
# ---------------------------------------------------------------------------
print("\n=== Passenger Count Distribution ===")

print(
    train_df["passenger_count"]
    .value_counts()
    .sort_index()
)

print("\n=== Trip Duration by Passenger Count ===")

print(
    train_df.groupby("passenger_count")["trip_duration"]
    .agg(["count", "mean", "median"])
    .sort_index()
    .round(2)
)

# ---------------------------------------------------------------------------
# 3. Store and Forward Flag
# ---------------------------------------------------------------------------
print("\n=== Store and Forward Flag Distribution ===")

print(
    train_df["store_and_fwd_flag"]
    .value_counts()
)

print("\n=== Trip Duration by Store and Forward Flag ===")

print(
    train_df.groupby("store_and_fwd_flag")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# ---------------------------------------------------------------------------
# 4. Basic data type check
# ---------------------------------------------------------------------------
print("\n=== Feature Data Types ===")

print(
    train_df[
        [
            "vendor_id",
            "passenger_count",
            "store_and_fwd_flag"
        ]
    ].dtypes
)