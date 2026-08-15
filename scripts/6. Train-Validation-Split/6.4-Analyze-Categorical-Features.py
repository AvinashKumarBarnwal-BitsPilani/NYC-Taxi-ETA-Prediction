import pandas as pd
from pathlib import Path

# Analyze categorical features in the training and validation datasets to
# determine the appropriate encoding strategy and verify category consistency.
# The analysis also checks whether validation contains unseen categories.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLIT_DIR = Path("data/split")

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_VAL_PATH = SPLIT_DIR / "X_val.csv"

# ---------------------------------------------------------------------------
# Load training and validation features
# ---------------------------------------------------------------------------
X_train = pd.read_csv(X_TRAIN_PATH)
X_val = pd.read_csv(X_VAL_PATH)

# ---------------------------------------------------------------------------
# Categorical features identified in Step 6.3
# ---------------------------------------------------------------------------
categorical_features = [
    "vendor_id",
    "store_and_fwd_flag"
]

# ---------------------------------------------------------------------------
# Analyze categorical features
# ---------------------------------------------------------------------------
print("=== Categorical Feature Analysis ===")

for column in categorical_features:

    print(f"\n--- {column} ---")

    print("Data type:")
    print(X_train[column].dtype)

    print("\nTRAIN categories:")
    print(sorted(X_train[column].unique()))

    print("\nVALIDATION categories:")
    print(sorted(X_val[column].unique()))

    print("\nTRAIN distribution:")
    print(X_train[column].value_counts().sort_index())

    print("\nVALIDATION distribution:")
    print(X_val[column].value_counts().sort_index())

# ---------------------------------------------------------------------------
# Check category consistency
# ---------------------------------------------------------------------------
print("\n=== Category Consistency Check ===")

for column in categorical_features:

    train_categories = set(X_train[column].unique())
    validation_categories = set(X_val[column].unique())

    unseen_in_validation = validation_categories - train_categories

    print(f"\n{column}")

    print(
        "Categories present in TRAIN but not VALIDATION:",
        sorted(train_categories - validation_categories)
    )

    print(
        "Categories present in VALIDATION but not TRAIN:",
        sorted(unseen_in_validation)
    )

    if unseen_in_validation:
        print("Status: REVIEW REQUIRED - unseen validation category detected.")
    else:
        print("Status: PASS - all validation categories exist in training.")

# ---------------------------------------------------------------------------
# Missing-value check
# ---------------------------------------------------------------------------
print("\n=== Missing Values ===")

print("TRAIN:")
print(X_train[categorical_features].isna().sum())

print("\nVALIDATION:")
print(X_val[categorical_features].isna().sum())

# ---------------------------------------------------------------------------
# Encoding strategy
# ---------------------------------------------------------------------------
print("\n=== Proposed Encoding Strategy ===")

print("Categorical features:")
for column in categorical_features:
    print(f"- {column}")

print("\nProposed encoder: OneHotEncoder")
print("Unknown-category handling: handle_unknown='ignore'")
print("Encoder fitting: TRAIN only")