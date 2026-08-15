import pandas as pd
import numpy as np
from pathlib import Path

# Implement a chronological 80/20 train-validation split using pickup_datetime.
# The earlier observations are used for training and the latest observations
# are reserved for validation to simulate future-trip prediction.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INTERIM_DIR = Path("data/interim")
SPLIT_DIR = Path("data/split")

TRAIN_PATH = INTERIM_DIR / "train_clean.csv"

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_VAL_PATH = SPLIT_DIR / "X_val.csv"
Y_TRAIN_PATH = SPLIT_DIR / "y_train.csv"
Y_VAL_PATH = SPLIT_DIR / "y_val.csv"

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
# Create datetime features
# ---------------------------------------------------------------------------
train_df["pickup_hour"] = (
    train_df["pickup_datetime"].dt.hour
)

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
# Haversine distance
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


train_df["distance_km"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"]
)

# ---------------------------------------------------------------------------
# Final candidate features from Step 5.9
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

target_column = "trip_duration"

# ---------------------------------------------------------------------------
# Sort chronologically
# ---------------------------------------------------------------------------
train_df = train_df.sort_values(
    "pickup_datetime"
).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Calculate chronological 80/20 split
# ---------------------------------------------------------------------------
split_index = int(len(train_df) * 0.80)

train_part = train_df.iloc[:split_index].copy()
validation_part = train_df.iloc[split_index:].copy()

# ---------------------------------------------------------------------------
# Separate X and y
# ---------------------------------------------------------------------------
X_train = train_part[feature_columns].copy()
y_train = train_part[target_column].copy()

X_val = validation_part[feature_columns].copy()
y_val = validation_part[target_column].copy()

# ---------------------------------------------------------------------------
# Display split information
# ---------------------------------------------------------------------------
print("=== Train / Validation Split ===")

print("Total records     :", len(train_df))
print("Training records  :", len(X_train))
print("Validation records:", len(X_val))

print("\nTraining proportion  :", round(len(X_train) / len(train_df), 4))
print("Validation proportion:", round(len(X_val) / len(train_df), 4))

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------
print("\n=== Date Ranges ===")

print("Training:")
print("Minimum:", train_part["pickup_datetime"].min())
print("Maximum:", train_part["pickup_datetime"].max())

print("\nValidation:")
print("Minimum:", validation_part["pickup_datetime"].min())
print("Maximum:", validation_part["pickup_datetime"].max())

# ---------------------------------------------------------------------------
# Verify chronological ordering
# ---------------------------------------------------------------------------
print("\n=== Chronological Ordering Check ===")

print(
    "Training maximum < Validation minimum:",
    train_part["pickup_datetime"].max()
    < validation_part["pickup_datetime"].min()
)

# ---------------------------------------------------------------------------
# Target statistics
# ---------------------------------------------------------------------------
print("\n=== Target Statistics ===")

print("\nTraining:")
print(y_train.describe().round(2))

print("\nValidation:")
print(y_val.describe().round(2))

# ---------------------------------------------------------------------------
# Feature shape verification
# ---------------------------------------------------------------------------
print("\n=== Feature Shapes ===")

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)

print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)

# ---------------------------------------------------------------------------
# Create split output directory
# ---------------------------------------------------------------------------
SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------------------------
# Save split datasets
# ---------------------------------------------------------------------------
X_train.to_csv(
    X_TRAIN_PATH,
    index=False
)

X_val.to_csv(
    X_VAL_PATH,
    index=False
)

y_train.to_csv(
    Y_TRAIN_PATH,
    index=False
)

y_val.to_csv(
    Y_VAL_PATH,
    index=False
)

# ---------------------------------------------------------------------------
# Confirm output files
# ---------------------------------------------------------------------------
print("\n=== Output Files ===")

print(X_TRAIN_PATH)
print(X_VAL_PATH)
print(Y_TRAIN_PATH)
print(Y_VAL_PATH)

print("\nTrain/Validation split completed successfully.")