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
# Convert pickup_datetime to datetime
# ---------------------------------------------------------------------------
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"],
    errors="coerce"
)

# ---------------------------------------------------------------------------
# Create datetime features for analysis
# ---------------------------------------------------------------------------
train_df["pickup_hour"] = train_df["pickup_datetime"].dt.hour

train_df["pickup_day_of_week"] = (
    train_df["pickup_datetime"].dt.dayofweek
)

train_df["pickup_month"] = (
    train_df["pickup_datetime"].dt.month
)

train_df["is_weekend"] = (
    train_df["pickup_day_of_week"]
    .isin([5, 6])
    .astype(int)
)


# ---------------------------------------------------------------------------
# Analyze trip duration by hour
# ---------------------------------------------------------------------------
print("=== Trip Duration by Hour ===")

print(
    train_df.groupby("pickup_hour")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# ---------------------------------------------------------------------------
# Analyze trip duration by day of week
# ---------------------------------------------------------------------------
print("\n=== Trip Duration by Day of Week ===")
print(
    train_df.groupby("pickup_day_of_week")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# ---------------------------------------------------------------------------
# Analyze trip duration by month
# ---------------------------------------------------------------------------
print("\n=== Trip Duration by Month ===")

print(
    train_df.groupby("pickup_month")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# ---------------------------------------------------------------------------
# Analyze trip duration: weekday vs weekend
# ---------------------------------------------------------------------------
print("\n=== Trip Duration: Weekday vs Weekend ===")

print(
    train_df.groupby("is_weekend")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)