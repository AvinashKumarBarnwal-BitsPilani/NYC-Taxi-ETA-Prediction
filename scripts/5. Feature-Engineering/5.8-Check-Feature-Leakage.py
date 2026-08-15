import pandas as pd
import numpy as np
from pathlib import Path

# Perform a final leakage audit by verifying that candidate features are based
# only on prediction-time information and that target/post-trip columns are
# excluded from the model feature set.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INTERIM_DIR = Path("data/interim")
TRAIN_PATH = INTERIM_DIR / "train_clean.csv"
TEST_PATH = INTERIM_DIR / "test_clean.csv"

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ---------------------------------------------------------------------------
# Recreate datetime features
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
# Recreate Haversine distance
# ---------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two geographic points.
    Returns distance in kilometers.
    """

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
# Define final candidate feature set
# ---------------------------------------------------------------------------
candidate_features = [
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
# Define target and post-trip columns
# ---------------------------------------------------------------------------
target_column = "trip_duration"

post_trip_columns = [
    "dropoff_datetime"
]

# ---------------------------------------------------------------------------
# Define prediction-time source columns
# ---------------------------------------------------------------------------
prediction_time_columns = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "vendor_id",
    "store_and_fwd_flag"
]

# ---------------------------------------------------------------------------
# Feature → Source mapping
# ---------------------------------------------------------------------------
feature_sources = {
    "vendor_id": ["vendor_id"],
    "passenger_count": ["passenger_count"],
    "store_and_fwd_flag": ["store_and_fwd_flag"],
    "pickup_hour": ["pickup_datetime"],
    "pickup_day_of_week": ["pickup_datetime"],
    "pickup_month": ["pickup_datetime"],
    "is_weekend": ["pickup_datetime"],
    "distance_km": [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude"
    ]
}

# ---------------------------------------------------------------------------
# 1. Candidate feature availability
# ---------------------------------------------------------------------------
print("=== Candidate Feature Availability ===")

missing_train = [
    feature
    for feature in candidate_features
    if feature not in train_df.columns
]

missing_test = [
    feature
    for feature in candidate_features
    if feature not in test_df.columns
]

print("\nMissing from TRAIN:")
print(missing_train)

print("\nMissing from TEST:")
print(missing_test)

# ---------------------------------------------------------------------------
# 2. Check target and post-trip columns
# ---------------------------------------------------------------------------
print("\n=== Leakage / Post-Trip Columns ===")

print(
    f"Target column '{target_column}': "
    f"TRAIN={'Present' if target_column in train_df.columns else 'Absent'}, "
    f"TEST={'Present' if target_column in test_df.columns else 'Absent'}"
)

for column in post_trip_columns:
    print(
        f"Post-trip column '{column}': "
        f"TRAIN={'Present' if column in train_df.columns else 'Absent'}, "
        f"TEST={'Present' if column in test_df.columns else 'Absent'}"
    )

# ---------------------------------------------------------------------------
# 3. Check accidental inclusion of target/post-trip columns
# ---------------------------------------------------------------------------
print("\n=== Candidate Feature Leakage Check ===")

leakage_in_candidates = [
    feature
    for feature in candidate_features
    if feature == target_column or feature in post_trip_columns
]

if leakage_in_candidates:
    print("Potential leakage detected:")
    print(leakage_in_candidates)
else:
    print(
        "No target or post-trip columns are included in candidate features."
    )

# ---------------------------------------------------------------------------
# 4. Display feature → source mapping
# ---------------------------------------------------------------------------
print("\n=== Feature Source Mapping ===")

for feature, sources in feature_sources.items():
    print(f"{feature} <- {sources}")

# ---------------------------------------------------------------------------
# 5. Verify feature sources are prediction-time information
# ---------------------------------------------------------------------------
print("\n=== Source Availability Check ===")

invalid_sources = {}

for feature, sources in feature_sources.items():

    unavailable_sources = [
        source
        for source in sources
        if source not in prediction_time_columns
    ]

    if unavailable_sources:
        invalid_sources[feature] = unavailable_sources


if invalid_sources:
    print("Potentially unavailable sources detected:")
    print(invalid_sources)
else:
    print(
        "All feature sources are available at prediction time."
    )

# ---------------------------------------------------------------------------
# 6. Final leakage status
# ---------------------------------------------------------------------------
print("\n=== Final Leakage Status ===")

if (
    not missing_train
    and not missing_test
    and not leakage_in_candidates
    and not invalid_sources
):
    print("PASS - No feature leakage identified.")
else:
    print("REVIEW REQUIRED - Potential leakage or feature issue detected.")