import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INTERIM_DIR = Path("data/interim")

TRAIN_PATH = INTERIM_DIR / "train_clean.csv"
TEST_PATH = INTERIM_DIR / "test_clean.csv"

# ---------------------------------------------------------------------------
# Load cleaned datasets
# ---------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ---------------------------------------------------------------------------
# Geographic columns
# ---------------------------------------------------------------------------
geo_columns = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude"
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
print("=== Geographic Column Data Types ===")
print(train_df[geo_columns].dtypes)

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
print("\n=== Missing Geographic Values ===")
print("TRAIN:")
print(train_df[geo_columns].isna().sum())

print("\nTEST:")
print(test_df[geo_columns].isna().sum())

# ---------------------------------------------------------------------------
# Geographic ranges
# ---------------------------------------------------------------------------
print("\n=== TRAIN Geographic Ranges ===")
print(train_df[geo_columns].agg(["min", "max"]).T)

print("\n=== TEST Geographic Ranges ===")
print(test_df[geo_columns].agg(["min", "max"]).T)

# ---------------------------------------------------------------------------
# Sample records
# ---------------------------------------------------------------------------
print("\n=== Sample Geographic Records ===")
print(
    train_df[
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude"
        ]
    ].head()
)

# ---------------------------------------------------------------------------
# Identical pickup/dropoff locations
# ---------------------------------------------------------------------------
same_location = (
    (train_df["pickup_longitude"] == train_df["dropoff_longitude"]) &
    (train_df["pickup_latitude"] == train_df["dropoff_latitude"])
)

print("\n=== Identical Pickup/Dropoff Locations ===")
print("TRAIN:", same_location.sum())