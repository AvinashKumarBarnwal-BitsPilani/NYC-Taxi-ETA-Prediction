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
# Convert pickup_datetime to datetime
# ---------------------------------------------------------------------------
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"],
    errors="coerce"
)

test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"],
    errors="coerce"
)

# ---------------------------------------------------------------------------
# Validate datetime conversion
# ---------------------------------------------------------------------------
print("TRAIN invalid datetime conversions:",
      train_df["pickup_datetime"].isna().sum())

print("TEST invalid datetime conversions :",
      test_df["pickup_datetime"].isna().sum())

# ---------------------------------------------------------------------------
# Create datetime features
# ---------------------------------------------------------------------------
for df in [train_df, test_df]:

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
# Display sample of engineered features
# ---------------------------------------------------------------------------
datetime_columns = [
    "pickup_datetime",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend"
]

print("\n=== TRAIN Datetime Features ===")
print(train_df[datetime_columns].head())

print("\n=== TEST Datetime Features ===")
print(test_df[datetime_columns].head())

# ---------------------------------------------------------------------------
# Display unique values for validation
# ---------------------------------------------------------------------------
print("\n=== Unique Values ===")

for column in [
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend"
]:
    print(f"{column}: {sorted(train_df[column].unique())}")

# ---------------------------------------------------------------------------
# Display datetime ranges
# ---------------------------------------------------------------------------
print("\n=== Datetime Range ===")

print("TRAIN:")
print("Minimum:", train_df["pickup_datetime"].min())
print("Maximum:", train_df["pickup_datetime"].max())

print("\nTEST:")
print("Minimum:", test_df["pickup_datetime"].min())
print("Maximum:", test_df["pickup_datetime"].max())