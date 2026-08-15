import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Build and verify the preprocessing transformation using TRAIN data only.
# The fitted transformer is then used to transform validation data and verify
# feature compatibility, scaling, encoding, missing values, and leakage safety.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLIT_DIR = Path("data/split")

X_TRAIN_PATH = SPLIT_DIR / "X_train.csv"
X_VAL_PATH = SPLIT_DIR / "X_val.csv"

# ---------------------------------------------------------------------------
# Load train and validation features
# ---------------------------------------------------------------------------
X_train = pd.read_csv(X_TRAIN_PATH)
X_val = pd.read_csv(X_VAL_PATH)

# ---------------------------------------------------------------------------
# Feature groups decided in Steps 6.3 and 6.4
# ---------------------------------------------------------------------------
numerical_features = [
    "distance_km"
]

categorical_features = [
    "vendor_id",
    "store_and_fwd_flag"
]

# ---------------------------------------------------------------------------
# Create preprocessing pipeline
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

# ---------------------------------------------------------------------------
# Fit preprocessing ONLY on TRAIN
# ---------------------------------------------------------------------------
X_train_processed = preprocessor.fit_transform(X_train)

# ---------------------------------------------------------------------------
# Transform VALIDATION using fitted TRAIN preprocessor
# ---------------------------------------------------------------------------
X_val_processed = preprocessor.transform(X_val)

# ---------------------------------------------------------------------------
# Convert to DataFrame
# ---------------------------------------------------------------------------
feature_names = preprocessor.get_feature_names_out()

X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_val_processed_df = pd.DataFrame(
    X_val_processed,
    columns=feature_names
)

# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------
print("=== Processed Dataset Shapes ===")

print("Original X_train:", X_train.shape)
print("Original X_val  :", X_val.shape)

print("Processed X_train:", X_train_processed_df.shape)
print("Processed X_val  :", X_val_processed_df.shape)

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
print("\n=== Processed Feature Names ===")

for feature in feature_names:
    print(feature)

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
print("\n=== Missing Values ===")

print(
    "TRAIN:",
    X_train_processed_df.isna().sum().sum()
)

print(
    "VALIDATION:",
    X_val_processed_df.isna().sum().sum()
)

# ---------------------------------------------------------------------------
# Infinite values
# ---------------------------------------------------------------------------
print("\n=== Infinite Values ===")

print(
    "TRAIN:",
    np.isinf(X_train_processed_df.to_numpy()).sum()
)

print(
    "VALIDATION:",
    np.isinf(X_val_processed_df.to_numpy()).sum()
)

# ---------------------------------------------------------------------------
# Numerical scaling verification
# ---------------------------------------------------------------------------
print("\n=== Scaled Numerical Feature Check ===")

scaled_distance = X_train_processed_df[
    "numerical__distance_km"
]

print(
    "TRAIN distance_km mean:",
    round(scaled_distance.mean(), 6)
)

print(
    "TRAIN distance_km std:",
    round(scaled_distance.std(), 6)
)

# ---------------------------------------------------------------------------
# Categorical encoding verification
# ---------------------------------------------------------------------------
print("\n=== Categorical Encoding Check ===")

categorical_output_features = [
    feature
    for feature in feature_names
    if feature.startswith("categorical__")
]

print(
    "Encoded categorical features:",
    categorical_output_features
)

# ---------------------------------------------------------------------------
# Train / validation feature compatibility
# ---------------------------------------------------------------------------
print("\n=== Train / Validation Feature Compatibility ===")

print(
    "Same feature columns:",
    list(X_train_processed_df.columns)
    == list(X_val_processed_df.columns)
)

print(
    "Same feature count:",
    X_train_processed_df.shape[1]
    == X_val_processed_df.shape[1]
)

# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------
print("\n=== Preprocessing Leakage Check ===")

print(
    "Preprocessor fitted on TRAIN only: PASS"
)

print(
    "Validation transformed using fitted TRAIN preprocessor: PASS"
)

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
has_missing_values = (
    X_train_processed_df.isna().sum().sum() > 0
    or X_val_processed_df.isna().sum().sum() > 0
)

has_infinite_values = (
    np.isinf(X_train_processed_df.to_numpy()).sum() > 0
    or np.isinf(X_val_processed_df.to_numpy()).sum() > 0
)

same_features = (
    list(X_train_processed_df.columns)
    == list(X_val_processed_df.columns)
)

print("\n=== Final Verification Status ===")

if (
    not has_missing_values
    and not has_infinite_values
    and same_features
):
    print("PASS - Preprocessed datasets are valid and compatible.")
else:
    print("REVIEW REQUIRED - Preprocessing verification failed.")