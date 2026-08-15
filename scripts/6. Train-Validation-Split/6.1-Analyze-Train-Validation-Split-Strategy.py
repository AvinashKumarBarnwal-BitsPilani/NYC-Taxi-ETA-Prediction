import pandas as pd
from pathlib import Path

# Analyze the temporal distribution of the cleaned TRAIN dataset to determine
# whether a Random or Time-based train/validation split better represents the
# project's prediction scenario.

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
# Convert pickup_datetime
# ---------------------------------------------------------------------------
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"],
    errors="coerce"
)

# ---------------------------------------------------------------------------
# Overall date range
# ---------------------------------------------------------------------------
print("=== Overall Pickup Datetime Range ===")
print("Minimum:", train_df["pickup_datetime"].min())
print("Maximum:", train_df["pickup_datetime"].max())

# ---------------------------------------------------------------------------
# Records by month
# ---------------------------------------------------------------------------
train_df["pickup_month_period"] = (
    train_df["pickup_datetime"].dt.to_period("M")
)

print("\n=== Records by Month ===")
print(
    train_df["pickup_month_period"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------------------------
# Records by date
# ---------------------------------------------------------------------------
train_df["pickup_date"] = (
    train_df["pickup_datetime"].dt.date
)

daily_counts = (
    train_df["pickup_date"]
    .value_counts()
    .sort_index()
)

print("\n=== Daily Record Statistics ===")
print("Number of unique dates:", daily_counts.size)
print("Minimum records/day:", daily_counts.min())
print("Maximum records/day:", daily_counts.max())
print("Mean records/day:", round(daily_counts.mean(), 2))

# ---------------------------------------------------------------------------
# Check for missing calendar dates
# ---------------------------------------------------------------------------
full_date_range = pd.date_range(
    start=train_df["pickup_datetime"].min().normalize(),
    end=train_df["pickup_datetime"].max().normalize(),
    freq="D"
)

observed_dates = pd.to_datetime(
    train_df["pickup_date"]
).dt.normalize().unique()

missing_dates = full_date_range.difference(
    observed_dates
)

print("\n=== Calendar Date Continuity ===")
print("Expected calendar days:", len(full_date_range))
print("Observed calendar days:", len(observed_dates))
print("Missing calendar days:", len(missing_dates))

if len(missing_dates) > 0:
    print("Missing dates:")
    print(missing_dates)
else:
    print("No missing calendar dates.")

# ---------------------------------------------------------------------------
# Chronological sample
# ---------------------------------------------------------------------------
print("\n=== Earliest 5 Records ===")

print(
    train_df[
        ["pickup_datetime", "trip_duration"]
    ]
    .sort_values("pickup_datetime")
    .head()
)

print("\n=== Latest 5 Records ===")
print(
    train_df[
        ["pickup_datetime", "trip_duration"]
    ]
    .sort_values("pickup_datetime")
    .tail()
)