# Evaluate whether defining peak commuting periods as a binary rush_hour feature
# provides useful information beyond pickup_hour by comparing trip duration
# during rush and non-rush hours.
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
# Convert pickup_datetime and extract pickup_hour
# ---------------------------------------------------------------------------
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"],
    errors="coerce"
)

train_df["pickup_hour"] = (
    train_df["pickup_datetime"].dt.hour
)

# ---------------------------------------------------------------------------
# Analyze trip duration by hour
# ---------------------------------------------------------------------------
print("=== Trip Duration by Hour ===")

hour_analysis = (
    train_df.groupby("pickup_hour")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print(hour_analysis)

# ---------------------------------------------------------------------------
# Define candidate rush-hour periods
#
# Morning: 07:00–09:59
# Evening: 16:00–18:59
# ---------------------------------------------------------------------------
rush_hours = [7, 8, 9, 16, 17, 18]

train_df["rush_hour"] = (
    train_df["pickup_hour"]
    .isin(rush_hours)
    .astype(int)
)

# ---------------------------------------------------------------------------
# Analyze trip duration: rush vs non-rush
# ---------------------------------------------------------------------------
print("\n=== Trip Duration: Rush Hour vs Non-Rush Hour ===")
print(
    train_df.groupby("rush_hour")["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# ---------------------------------------------------------------------------
# Analyze rush-hour distribution
# ---------------------------------------------------------------------------
print("\n=== Rush Hour Distribution ===")
print(
    train_df["rush_hour"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------------------------
# Display rush-hour definition
# ---------------------------------------------------------------------------
print("\n=== Rush Hour Definition ===")

print("Morning Rush : 07:00–09:59")
print("Evening Rush : 16:00–18:59")
print("Rush Hours   :", rush_hours)