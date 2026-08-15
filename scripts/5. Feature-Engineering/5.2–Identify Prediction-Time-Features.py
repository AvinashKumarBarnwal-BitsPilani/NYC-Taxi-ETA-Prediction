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
# Define column categories
# ---------------------------------------------------------------------------

# Identifier: kept for record tracking, but not used as a model feature
identifier_columns = [
    "id"
]

# Candidate prediction-time features
candidate_features = [
    "vendor_id",
    "pickup_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag"
]

# Columns unavailable at prediction time / leakage
leakage_columns = [
    "dropoff_datetime"
]

# Target variable
target_column = [
    "trip_duration"
]


# ---------------------------------------------------------------------------
# Display classification
# ---------------------------------------------------------------------------
print("=== Prediction-Time Feature Classification ===")

print("\nIdentifier:")
for column in identifier_columns:
    print(f"  - {column}")

print("\nCandidate Prediction-Time Features:")
for column in candidate_features:
    print(f"  - {column}")

print("\nLeakage / Unavailable at Prediction Time:")
for column in leakage_columns:
    print(f"  - {column}")

print("\nTarget:")
for column in target_column:
    print(f"  - {column}")


# ---------------------------------------------------------------------------
# Verify classification against TRAIN and TEST schemas
# ---------------------------------------------------------------------------
train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

print("\n=== Schema Verification ===")

print("\nColumns present only in TRAIN:")
print(sorted(train_columns - test_columns))

print("\nColumns present in both TRAIN and TEST:")
print(sorted(train_columns & test_columns))


# ---------------------------------------------------------------------------
# Verify candidate features are available in both TRAIN and TEST
# ---------------------------------------------------------------------------
missing_candidate_features = [
    column
    for column in candidate_features
    if column not in train_columns or column not in test_columns
]

print("\n=== Candidate Feature Verification ===")

if not missing_candidate_features:
    print("All candidate prediction-time features are present in both TRAIN and TEST.")
else:
    print("Missing candidate features:")
    for column in missing_candidate_features:
        print(f"  - {column}")


# ---------------------------------------------------------------------------
# Verify leakage and target columns
# ---------------------------------------------------------------------------
print("\n=== Leakage / Target Verification ===")

for column in leakage_columns:
    print(
        f"{column}: "
        f"TRAIN={'Yes' if column in train_columns else 'No'}, "
        f"TEST={'Yes' if column in test_columns else 'No'}"
    )

for column in target_column:
    print(
        f"{column}: "
        f"TRAIN={'Yes' if column in train_columns else 'No'}, "
        f"TEST={'Yes' if column in test_columns else 'No'}"
    )


# ---------------------------------------------------------------------------
# Final model input definition
# ---------------------------------------------------------------------------
model_input_columns = candidate_features

print("\n=== Final Candidate Model Inputs ===")
print(model_input_columns)

print("\n=== Excluded from Model Inputs ===")
print("Identifier:", identifier_columns)
print("Leakage:", leakage_columns)
print("Target:", target_column)