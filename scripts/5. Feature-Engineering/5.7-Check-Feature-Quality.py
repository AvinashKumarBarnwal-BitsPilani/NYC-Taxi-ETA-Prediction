import pandas as pd
import numpy as np
from pathlib import Path

# Check the quality of the current candidate features by validating missing
# values, infinite values, expected ranges, data types, and TRAIN/TEST feature
# consistency before moving to preprocessing.

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
# Create datetime features
# ---------------------------------------------------------------------------
for df in [train_df, test_df]:

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce"
    )

    df["pickup_hour"] = df["pickup_datetime"].dt.hour

    df["pickup_day_of_week"] = (
        df["pickup_datetime"].dt.dayofweek
    )

    df["pickup_month"] = (
        df["pickup_datetime"].dt.month
    )

    df["is_weekend"] = (
        df["pickup_day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

# ---------------------------------------------------------------------------
# Create Haversine distance
# ---------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0

    return earth_radius_km * c


for df in [train_df, test_df]:

    df["distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )

# ---------------------------------------------------------------------------
# Candidate feature list
# ---------------------------------------------------------------------------
feature_columns = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "distance_km"
]

# ---------------------------------------------------------------------------
# 1. Feature columns present in TRAIN and TEST
# ---------------------------------------------------------------------------
print("=== Feature Column Consistency ===")

print("\nMissing from TRAIN:")
print([
    column
    for column in feature_columns
    if column not in train_df.columns
])

print("\nMissing from TEST:")
print([
    column
    for column in feature_columns
    if column not in test_df.columns
])

# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------
print("\n=== Missing Values ===")

print("\nTRAIN:")
print(train_df[feature_columns].isna().sum())

print("\nTEST:")
print(test_df[feature_columns].isna().sum())

# ---------------------------------------------------------------------------
# 3. Infinite values
# ---------------------------------------------------------------------------
print("\n=== Infinite Values ===")

numeric_features = [
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "distance_km"
]

print("\nTRAIN:")

for column in numeric_features:
    print(
        f"{column}: "
        f"{np.isinf(train_df[column]).sum()}"
    )

print("\nTEST:")

for column in numeric_features:
    print(
        f"{column}: "
        f"{np.isinf(test_df[column]).sum()}"
    )

# ---------------------------------------------------------------------------
# 4. Expected range checks
# ---------------------------------------------------------------------------
print("\n=== Range Checks ===")

print("\nTRAIN:")

print(
    "pickup_hour:",
    train_df["pickup_hour"].min(),
    "to",
    train_df["pickup_hour"].max()
)

print(
    "pickup_day_of_week:",
    train_df["pickup_day_of_week"].min(),
    "to",
    train_df["pickup_day_of_week"].max()
)

print(
    "pickup_month:",
    train_df["pickup_month"].min(),
    "to",
    train_df["pickup_month"].max()
)

print(
    "is_weekend:",
    sorted(train_df["is_weekend"].unique())
)

print(
    "passenger_count:",
    train_df["passenger_count"].min(),
    "to",
    train_df["passenger_count"].max()
)

print(
    "distance_km:",
    train_df["distance_km"].min(),
    "to",
    train_df["distance_km"].max()
)


print("\nTEST:")

print(
    "pickup_hour:",
    test_df["pickup_hour"].min(),
    "to",
    test_df["pickup_hour"].max()
)

print(
    "pickup_day_of_week:",
    test_df["pickup_day_of_week"].min(),
    "to",
    test_df["pickup_day_of_week"].max()
)

print(
    "pickup_month:",
    test_df["pickup_month"].min(),
    "to",
    test_df["pickup_month"].max()
)

print(
    "is_weekend:",
    sorted(test_df["is_weekend"].unique())
)

print(
    "passenger_count:",
    test_df["passenger_count"].min(),
    "to",
    test_df["passenger_count"].max()
)

print(
    "distance_km:",
    test_df["distance_km"].min(),
    "to",
    test_df["distance_km"].max()
)

# ---------------------------------------------------------------------------
# 5. Categorical values
# ---------------------------------------------------------------------------
print("\n=== Categorical Values ===")

print("\nTRAIN:")
print("vendor_id:", sorted(train_df["vendor_id"].unique()))
print(
    "store_and_fwd_flag:",
    sorted(train_df["store_and_fwd_flag"].unique())
)

print("\nTEST:")
print("vendor_id:", sorted(test_df["vendor_id"].unique()))
print(
    "store_and_fwd_flag:",
    sorted(test_df["store_and_fwd_flag"].unique())
)

# ---------------------------------------------------------------------------
# 6. Feature data types
# ---------------------------------------------------------------------------
print("\n=== Feature Data Types ===")

print("\nTRAIN:")
print(train_df[feature_columns].dtypes)

print("\nTEST:")
print(test_df[feature_columns].dtypes)