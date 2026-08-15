from pathlib import Path
import pandas as pd
# --------------------------------------------------
# Configuration
# --------------------------------------------------
RAW_FILE = Path("data/raw/train.csv")
OUTPUT_FILE = Path("data/interim/train_sample_10pct.csv")

SAMPLE_FRACTION = 0.10
RANDOM_STATE = 42

# --------------------------------------------------
# Validate input
# # --------------------------------------------------
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {RAW_FILE}")

# --------------------------------------------------
# Create output directory
# --------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load raw training data
# --------------------------------------------------
print("Loading training dataset...")
train = pd.read_csv(RAW_FILE)
print(f"Full training dataset: {len(train):,} rows")

# --------------------------------------------------
# Create reproducible 10% random sample
# --------------------------------------------------
sample = train.sample(
    frac=SAMPLE_FRACTION,
    random_state=RANDOM_STATE
)

# --------------------------------------------------
# Save development dataset
# --------------------------------------------------
sample.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------
print()
print("Development dataset created successfully.")
print(f"Sampling fraction : {SAMPLE_FRACTION:.0%}")
print(f"Random state      : {RANDOM_STATE}")
print(f"Sample rows       : {len(sample):,}")
print(f"Output file       : {OUTPUT_FILE}")