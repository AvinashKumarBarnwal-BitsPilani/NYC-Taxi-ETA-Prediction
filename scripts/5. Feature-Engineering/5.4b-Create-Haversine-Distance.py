# Create a Haversine distance feature (distance_km) using pickup and dropoff latitude/longitude coordinates. 
# The same calculation is applied to both TRAIN and TEST without using the target, ensuring a leakage-free feature.

import pandas as pd
import numpy as np
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
# Haversine distance function
# ---------------------------------------------------------------------------
def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance between two geographic points.

    Returns distance in kilometers.
    """

    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    # Earth's approximate radius in kilometers
    earth_radius_km = 6371.0

    return earth_radius_km * c

# ---------------------------------------------------------------------------
# Create distance feature
# ---------------------------------------------------------------------------
for df in [train_df, test_df]:

    df["distance_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"]
    )

# ---------------------------------------------------------------------------
# Display sample results
# ---------------------------------------------------------------------------
print("=== TRAIN Distance Feature ===")
print(
    train_df[
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "distance_km"
        ]
    ].head()
)

print("\n=== TEST Distance Feature ===")
print(
    test_df[
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "distance_km"
        ]
    ].head()
)

# ---------------------------------------------------------------------------
# Distance summary
# ---------------------------------------------------------------------------
print("\n=== TRAIN Distance Summary ===")
print(
    train_df["distance_km"]
    .describe()
    .round(3)
)

print("\n=== TEST Distance Summary ===")

print(
    test_df["distance_km"]
    .describe()
    .round(3)
)

# ---------------------------------------------------------------------------
# Zero-distance trips
# ---------------------------------------------------------------------------
print("\n=== Zero-Distance Trips ===")
print(
    "TRAIN:",
    (train_df["distance_km"] == 0).sum()
)

print(
    "TEST:",
    (test_df["distance_km"] == 0).sum()
)