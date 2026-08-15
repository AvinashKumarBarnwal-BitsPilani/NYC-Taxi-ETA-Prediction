import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INTERIM_DIR = Path("data/interim")

TRAIN_PATH = INTERIM_DIR / "train_clean.csv"
TEST_PATH = INTERIM_DIR / "test_clean.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("TRAIN columns:")
print(train_df.columns.tolist())

print("\nTEST columns:")
print(test_df.columns.tolist())

print("\nColumns present only in TRAIN:")
print(sorted(set(train_df.columns) - set(test_df.columns)))

print("\nColumns present in both TRAIN and TEST:")
print(sorted(set(train_df.columns) & set(test_df.columns)))