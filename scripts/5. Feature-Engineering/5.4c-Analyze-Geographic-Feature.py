# Analyze the relationship between Haversine distance (distance_km) and trip_duration.
# Also inspect zero-distance trips and extreme-distance trips to identify potential
# anomalies before deciding whether distance_km should be retained as a model feature.

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
# Create Haversine distance
# ---------------------------------------------------------------------------
import numpy as np

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


train_df["distance_km"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"]
)

# ---------------------------------------------------------------------------
# Analyze trip duration by distance range
# ---------------------------------------------------------------------------
distance_bins = [
    -0.01,
    1,
    2,
    5,
    10,
    20,
    float("inf")
]

distance_labels = [
    "0-1 km",
    "1-2 km",
    "2-5 km",
    "5-10 km",
    "10-20 km",
    "20+ km"
]

train_df["distance_range"] = pd.cut(
    train_df["distance_km"],
    bins=distance_bins,
    labels=distance_labels
)

print("=== Trip Duration by Distance Range ===")
print(
    train_df.groupby(
        "distance_range",
        observed=False
    )["trip_duration"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# ---------------------------------------------------------------------------
# Analyze zero-distance trips
# ---------------------------------------------------------------------------
zero_distance = train_df["distance_km"] == 0

print("\n=== Zero-Distance Trips ===")
print("Count:", zero_distance.sum())

print(
    train_df.loc[
        zero_distance,
        "trip_duration"
    ]
    .describe()
    .round(2)
)

# ---------------------------------------------------------------------------
# Correlation between distance and trip duration
# ---------------------------------------------------------------------------
correlation = train_df[
    ["distance_km", "trip_duration"]
].corr()

print("\n=== Distance vs Trip Duration Correlation ===")
print(correlation.round(4))

# ---------------------------------------------------------------------------
# Inspect longest-distance trips
# ---------------------------------------------------------------------------
print("\n=== Top 10 Longest-Distance Trips ===")

longest_trips = train_df.nlargest(
    10,
    "distance_km"
)[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "distance_km",
        "trip_duration"
    ]
]

print(longest_trips.to_string(index=False))