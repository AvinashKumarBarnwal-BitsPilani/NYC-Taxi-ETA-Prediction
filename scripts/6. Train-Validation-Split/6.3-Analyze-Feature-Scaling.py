import pandas as pd
from pathlib import Path

# Analyze the numerical and discrete features in the training set to determine
# which features require scaling and which should be handled as categorical or
# binary features during preprocessing.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLIT_DIR = Path("data/split")
X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"

# ---------------------------------------------------------------------------
# Load training features only
# ---------------------------------------------------------------------------
X_train = pd.read_csv(X_TRAIN_PATH)

# ---------------------------------------------------------------------------
# Feature groups for analysis
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
# 1. Data types
# ---------------------------------------------------------------------------
print("=== Feature Data Types ===")
print(X_train[candidate_features].dtypes)

# ---------------------------------------------------------------------------
# 2. Numerical feature statistics
# ---------------------------------------------------------------------------
numeric_features = [
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "distance_km"
]

print("\n=== Numerical Feature Statistics ===")

print(
    X_train[numeric_features]
    .describe()
    .T
    .round(3)
)

# ---------------------------------------------------------------------------
# 3. Feature ranges
# ---------------------------------------------------------------------------
print("\n=== Feature Ranges ===")

for column in numeric_features:

    print(
        f"{column}: "
        f"min={X_train[column].min()}, "
        f"max={X_train[column].max()}, "
        f"range={X_train[column].max() - X_train[column].min():.3f}"
    )

# ---------------------------------------------------------------------------
# 4. Unique values for discrete / categorical-like features
# ---------------------------------------------------------------------------
print("\n=== Unique Values ===")

for column in [
    "vendor_id",
    "store_and_fwd_flag",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend"
]:

    print(
        f"{column}: "
        f"{X_train[column].nunique()} unique values"
    )

    print(
        sorted(X_train[column].unique())
    )

# ---------------------------------------------------------------------------
# 5. Missing values
# ---------------------------------------------------------------------------
print("\n=== Missing Values ===")

print(
    X_train[candidate_features]
    .isna()
    .sum()
)

# ---------------------------------------------------------------------------
# 6. Preliminary scaling classification
# ---------------------------------------------------------------------------
print("\n=== Preliminary Feature Classification ===")

print("\nCategorical:")
print("vendor_id")
print("store_and_fwd_flag")

print("\nDiscrete / temporal:")
print("passenger_count")
print("pickup_hour")
print("pickup_day_of_week")
print("pickup_month")
print("is_weekend")

print("\nContinuous numerical:")
print("distance_km")

# ---------------------------------------------------------------------------
# 7. Scaling observation
# ---------------------------------------------------------------------------
print("\n=== Scaling Observation ===")

print(
    "distance_km has the largest continuous numerical scale and is the "
    "primary candidate for numerical scaling."
)

print(
    "Categorical, binary, and temporal features should not be blindly "
    "standardized; their final treatment will depend on preprocessing "
    "and encoding decisions."
)