# Phase 3A - Data Understanding-Validation


## Table of Contnets
- [Step 1 - Understand the Dataset](#step-1--understand-the-dataset)
  - [1.1 Inventory the Raw Dataset](#11--inventory-the-raw-dataset)
  - [1.2 Understand the Dataset Schema](#12--understand-the-dataset-schema)
  - [1.3 Inspect Data Types](#13--inspect-data-types)
  - [1.4 Initial Missing-Value Check](#14--initial-missing-value-check)
  - [1.5 Initial Statistics & Cardinality](#15--initial-statistics--cardinality)
  - [1.6 Complete Dataset Row Counts](#16--complete-dataset-row-counts)
  - [1.7 Train vs Test Schema](#17--train-vs-test-schema)
  - [1.8 Complete Missing-Value Check](#18--complete-missing-value-check)
  - [1.9 Complete Target Analysis](#19--complete-target-analysis)
  - [1.10 Datetime Range & Consistency](#110--datetime-range--consistency)
  - [1.11 Create & Validate 10% Development Dataset](#111--create--validate-10-development-dataset)
- [Step 2 - Define the Data Contract & Validation Rules](#step-2--define-the-data-contract--validation-rules)
  - [2.1 Define Expected Schema/Columns](#21--define-expected-schemacolumns)
  - [2.2 Define Required Columns](#22--define-required-columns)
  - [2.3 Define Expected Data Types](#23--define-expected-data-types)
  - [2.4 Define Categorical Value Rules](#24--define-categorical-value-rules)
  - [2.5 Define Numeric Range Rules](#25--define-numeric-range-rules)
  - [2.6 Define Datetime Consistency Rules](#26--define-datetime-consistency-rules)
  - [2.7 Define Target Validation Rules](#27--define-target-validation-rules)
  - [2.8 Define Missing-Value Rules](#28--define-missing-value-rules)
  - [2.9 Define Duplicate Handling Rules](#29--define-duplicate-handling-rules)
  - [2.10 Define Invalid-Record Rules](#210--define-invalid-record-rules)
  - [2.11 Create Data Validation Contract](#211--create-data-validation-contract)
  - [2.12 Validate the Contract Against the Dataset](#212--validate-the-contract-against-the-dataset)

# Step 1 – Understand the Dataset

**Goal:** Build a clear understanding of what data we have before modifying anything.

## 1.1 – Inventory the Raw Dataset

Inspect the files available under:

```text
data/raw/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### Current Dataset

| Dataset | Records | Approx. Size |
|---|---:|---:|
| `train.csv` | 1,458,644 | ~200.6 MB |
| `test.csv` | 625,134 | ~70.8 MB |
| `sample_submission.csv` | 625,134 | ~8.75 MB |

### Expected Outcome

Understand the available datasets and their scale.

---

## 1.2 – Understand the Dataset Schema

### Training Dataset

The training dataset contains 11 columns:

```text
id
vendor_id
pickup_datetime
dropoff_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
trip_duration
```

### Test Dataset

The test dataset contains 9 columns:

```text
id
vendor_id
pickup_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
```

### Train-only Columns

```text
dropoff_datetime
trip_duration
```

`trip_duration` is the prediction target.

`dropoff_datetime` is available in the training dataset but not in the test dataset and therefore must not be used as a prediction feature.

### Expected Outcome

Clearly identify:

- Training features
- Test features
- Prediction target
- Identifier columns
- Fields requiring special handling

---

## 1.3 – Inspect Data Types

Current data types observed:

| Column | Data Type | Initial Classification |
|---|---|---|
| `id` | object | Identifier |
| `vendor_id` | int64 | Categorical |
| `pickup_datetime` | object | Datetime |
| `dropoff_datetime` | object | Datetime |
| `passenger_count` | int64 | Numerical |
| `pickup_longitude` | float64 | Geographic |
| `pickup_latitude` | float64 | Geographic |
| `dropoff_longitude` | float64 | Geographic |
| `dropoff_latitude` | float64 | Geographic |
| `store_and_fwd_flag` | object | Categorical |
| `trip_duration` | int64 | Target |

### Expected Outcome

Understand how each feature is currently represented and identify fields that may require transformation later.

---

## 1.4 – Initial Missing-Value Check

An initial missing-value check was performed on the training dataset using the first 100,000 records.

### Result

No missing values were observed in the initial sample.

```text
All columns → 0 missing values
```

This was followed by a complete dataset check in **Section 1.8**.

---

## 1.5 – Initial Statistics & Cardinality

Initial profiling was performed using 100,000 training records.

### Cardinality

| Column | Unique Values in Sample |
|---|---:|
| `vendor_id` | 2 |
| `store_and_fwd_flag` | 2 |
| `passenger_count` | 7 |
| `trip_duration` | 4,046 |
| `pickup_longitude` | 12,656 |
| `dropoff_longitude` | 15,490 |
| `pickup_latitude` | 25,839 |
| `dropoff_latitude` | 30,239 |
| `pickup_datetime` | 99,590 |
| `dropoff_datetime` | 99,625 |
| `id` | 100,000 |

### Initial Observations

- `vendor_id` has 2 unique values.
- `store_and_fwd_flag` has 2 unique values.
- `passenger_count` has 7 unique values.
- `id` behaves as an identifier.
- Geographic and datetime features have relatively high cardinality.
- `passenger_count` contains a minimum value of `0`.

The presence of `passenger_count = 0` is an observation only.

**No records are removed at this stage.**

Its validity will be investigated during:

**Step 2 – Define the Data Contract & Validation Rules**

---

## 1.6 – Complete Dataset Row Counts

The complete datasets were loaded to determine their actual sizes.

```text
Training records:        1,458,644
Test records:              625,134
Sample submissions:        625,134
```

### Observation

The training dataset contains approximately **1.46 million records** and is large enough that repeatedly using the complete dataset during development may increase experimentation time.

Therefore, a reproducible 10% development dataset will be created in **Section 1.11**.

---

## 1.7 – Train vs Test Schema

The training and test schemas were compared.

### TRAIN-only Columns

```text
dropoff_datetime
trip_duration
```

### TEST-only Columns

```text
None
```

Therefore:

```text
trip_duration
```

is confirmed as the prediction target.

`dropoff_datetime` is available only during training and cannot be used as a prediction feature because it will not be available when making predictions on the test dataset.

---

## 1.8 – Complete Missing-Value Check

A complete missing-value check was performed on both the training and test datasets.

### Training Dataset

```text
All columns → 0 missing values
```

### Test Dataset

```text
All columns → 0 missing values
```

### Conclusion

No missing values are present in the raw training or test datasets.

However, missing values may still need to be checked after data cleaning and feature engineering.

---

## 1.9 – Complete Target Analysis

The prediction target is:

```text
trip_duration
```

The complete training dataset was analyzed.

### Target Statistics

| Statistic | Value |
|---|---:|
| Count | 1,458,644 |
| Mean | ~959.49 sec |
| Std Dev | ~5237.43 sec |
| Minimum | 1 sec |
| 25th Percentile | 397 sec |
| Median | 662 sec |
| 75th Percentile | 1,075 sec |
| Maximum | 3,526,282 sec |

### Important Observations

The target distribution is highly skewed.

The maximum observed value is:

```text
3,526,282 seconds
```

which is approximately:

```text
40.8 days
```

Several other values are close to:

```text
86,400 seconds ≈ 24 hours
```

These values appear suspicious and require further investigation.

### Important Rule

Do **not** remove or modify these observations during Step 1.

They will be investigated during:

- **Step 2 – Define the Data Contract & Validation Rules**
- **Step 3 – Data Quality Analysis**
- **Step 4 – Data Cleaning**

---

## 1.10 – Datetime Range & Consistency

The complete dataset was analyzed for datetime coverage.

### Pickup Datetime

```text
2016-01-01 00:00:17
        ↓
2016-06-30 23:59:39
```

### Dropoff Datetime

```text
2016-01-01 00:03:31
        ↓
2016-07-01 23:02:03
```

### Datetime Consistency Check

The following condition was checked:

```text
dropoff_datetime < pickup_datetime
```

Result:

```text
0 violations
```

Therefore, no records were found where the drop-off time occurred before the pickup time.

### Initial Observation

Datetime information is likely to be important during feature engineering.

Potential future features may include:

- Pickup hour
- Day
- Day of week
- Month
- Weekend indicator
- Rush-hour indicator

These decisions will be made during:

**Step 5 – Feature Engineering**

---

### 1.11 – Create & Validate 10% Development Dataset

Since the full training dataset contains **1,458,644 records**, a reproducible **10% random development dataset** containing **145,864 records** was created using `random_state=42`.

The 10% development dataset was validated against the **100% raw training dataset** across the following dimensions:

- Row count and schema
- Unique IDs
- Target (`trip_duration`) distribution
- Categorical feature distributions
- Temporal distribution
- Geographic distribution

The comparison showed that the 10% sample closely represents the full training dataset.

Some comparative observations:

| Metric | 100% Raw Training | 10% Development |
|---|---:|---:|
| Records | 1,458,644 | 145,864 |
| `trip_duration` Mean | 959.49 sec | 964.34 sec |
| `trip_duration` Median | 662 sec | 662 sec |
| `vendor_id = 1` | 46.50% | 46.44% |
| `vendor_id = 2` | 53.50% | 53.56% |
| `store_and_fwd_flag = N` | 99.45% | 99.46% |
| `store_and_fwd_flag = Y` | 0.55% | 0.54% |

Temporal distributions across **month, hour, and day of week** were also found to be very similar. Geographic statistics showed very close **median and percentile values**, with differences mainly observed in rare extreme geographic values.

### Conclusion

The validation confirms that the **10% development dataset is a sufficiently representative sample of the full training dataset** for development and experimentation.

The 10% dataset will be used to accelerate development and experimentation, while the **100% training dataset remains the source of truth for final validation and model training**.

**Step 1.11 – Complete ✅**
---

# Step 1 – Expected Deliverables

By the end of Step 1, we should have:

- [x] Raw dataset inventory
- [x] Dataset schema understanding
- [x] Data-type classification
- [x] Initial missing-value analysis
- [x] Initial statistics and cardinality
- [x] Complete row counts
- [x] Train vs Test schema comparison
- [x] Complete missing-value check
- [x] Complete target analysis
- [x] Datetime range and consistency analysis
- [ ] Reproducible 10% development dataset
- [ ] Sample validation against the full training dataset

### Step 1 Output

```text
data/
├── raw/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
└── interim/
    └── train_sample_10pct.csv
```

---

# Step 2 – Define the Data Contract & Validation Rules

**Goal:** Decide what constitutes a valid record before cleaning the dataset.

Instead of randomly cleaning data, define explicit rules.

### Tasks

- [ ] Define expected columns
- [ ] Define expected data types
- [ ] Define required columns
- [ ] Define acceptable ranges
- [ ] Define missing-value expectations
- [ ] Define duplicate handling rules
- [ ] Define invalid-record rules

Examples:

```text
trip_duration > 0

passenger_count > 0

latitude / longitude within expected NYC ranges

pickup_datetime < dropoff_datetime

vendor_id must contain valid values

store_and_fwd_flag must contain expected categories
```

### Important Principle

Do not silently delete data.

Every major cleaning rule should have a reason.

For example:

```text
Invalid trip_duration
        ↓
Why invalid?
        ↓
Cannot represent a real taxi trip
        ↓
Remove / flag record
```

### Project Location

```text
configs/
src/
tests/
```

### Expected Output

A documented **data validation contract** that Phase 3 can enforce programmatically.

## Step 2 Roadmap

```text
Step 2 – Define the Data Contract & Validation Rules
│
├── 2.1 Define Expected Schema
│
├── 2.2 Define Required Columns
│
├── 2.3 Define Expected Data Types
│
├── 2.4 Define Categorical Value Rules
│
├── 2.5 Define Numeric Range Rules
│
├── 2.6 Define Datetime Consistency Rules
│
├── 2.7 Define Target Validation Rules
│
├── 2.8 Define Missing-Value Rules
│
├── 2.9 Define Duplicate Handling Rules
│
├── 2.10 Define Invalid-Record Rules
│
├── 2.11 Create Data Validation Contract
│
└── 2.12 Validate the Contract Against the Dataset
```

### 2.1 – Define Expected Schema/Columns

The first step in the data contract is to define the expected schema for both the training and test datasets.

The schema observed in Step 1 was verified again using the raw CSV files.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv', nrows=0); test=pd.read_csv('data/raw/test.csv', nrows=0); print('=== TRAIN SCHEMA ==='); print(train.dtypes); print('\n=== TEST SCHEMA ==='); print(test.dtypes); print('\n=== TRAIN COLUMNS ==='); print(train.columns.tolist()); print('\n=== TEST COLUMNS ==='); print(test.columns.tolist())"
```
#### Output

```text
=== TRAIN SCHEMA ===
id                    object
vendor_id             object
pickup_datetime       object
dropoff_datetime      object
passenger_count       object
pickup_longitude      object
pickup_latitude       object
dropoff_longitude     object
dropoff_latitude      object
store_and_fwd_flag    object
trip_duration         object
dtype: object

=== TEST SCHEMA ===
id                    object
vendor_id             object
pickup_datetime       object
passenger_count       object
pickup_longitude      object
pickup_latitude       object
dropoff_longitude     object
dropoff_latitude      object
store_and_fwd_flag    object
dtype: object

=== TRAIN COLUMNS ===
['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration']

=== TEST COLUMNS ===
['id', 'vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']
```

### 2.2 – Define Required Columns

The required columns for the training and test datasets were identified based on the schema validated in Step 2.1.

#### Training Dataset

The following columns are required:

```text
id
vendor_id
pickup_datetime
dropoff_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
trip_duration
```

#### Test Dataset

The following columns are required:

```text
id
vendor_id
pickup_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
```

#### Training vs Test Difference

The following columns are present only in the training dataset:

```text
dropoff_datetime
trip_duration
```

This difference is expected.

- `trip_duration` is the target variable and is therefore not available in the test dataset.
- `dropoff_datetime` is also not available in the test dataset because it would not be known at prediction time.
- Therefore, the absence of these columns from the test dataset is considered valid.

#### Required Column Contract

| Dataset | Required Columns |
|---|---|
| **TRAIN** | `id`, `vendor_id`, `pickup_datetime`, `dropoff_datetime`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `store_and_fwd_flag`, `trip_duration` |
| **TEST** | `id`, `vendor_id`, `pickup_datetime`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `store_and_fwd_flag` |

### Validation Rule

During data validation:

- All required training columns must be present.
- All required test columns must be present.
- No required column should be missing.
- The absence of `trip_duration` and `dropoff_datetime` from TEST is expected and must not be treated as an error.

**Status: Step 2.2 Complete ✅**

### 2.3 – Define Expected Data Types

The expected logical data type for each column was defined based on the actual dataset structure observed during Step 1.

> **Important:** The raw CSV schema inspection using `nrows=0` showed all columns as `object`. This is because only the headers were read. The `object` dtype at this stage does not mean that all values are strings.
>
> The data contract therefore defines the **expected logical data type** that each column must satisfy after parsing/validation.

#### Training Dataset

| Column | Expected Type | Role |
|---|---|---|
| `id` | String | Identifier |
| `vendor_id` | Integer / Categorical | Feature |
| `pickup_datetime` | Datetime | Feature |
| `dropoff_datetime` | Datetime | Training-only field |
| `passenger_count` | Integer | Feature |
| `pickup_longitude` | Float | Feature |
| `pickup_latitude` | Float | Feature |
| `dropoff_longitude` | Float | Feature |
| `dropoff_latitude` | Float | Feature |
| `store_and_fwd_flag` | Categorical | Feature |
| `trip_duration` | Integer / Numeric | Target |

#### Test Dataset

| Column | Expected Type | Role |
|---|---|---|
| `id` | String | Identifier |
| `vendor_id` | Integer / Categorical | Feature |
| `pickup_datetime` | Datetime | Feature |
| `passenger_count` | Integer | Feature |
| `pickup_longitude` | Float | Feature |
| `pickup_latitude` | Float | Feature |
| `dropoff_longitude` | Float | Feature |
| `dropoff_latitude` | Float | Feature |
| `store_and_fwd_flag` | Categorical | Feature |

### Expected Data Type Contract

```text
TRAIN
│
├── id                    → String
├── vendor_id             → Integer / Categorical
├── pickup_datetime       → Datetime
├── dropoff_datetime      → Datetime
├── passenger_count       → Integer
├── pickup_longitude      → Float
├── pickup_latitude       → Float
├── dropoff_longitude     → Float
├── dropoff_latitude      → Float
├── store_and_fwd_flag    → Categorical
└── trip_duration         → Integer / Numeric (Target)


TEST
│
├── id                    → String
├── vendor_id             → Integer / Categorical
├── pickup_datetime       → Datetime
├── passenger_count       → Integer
├── pickup_longitude      → Float
├── pickup_latitude       → Float
├── dropoff_longitude     → Float
├── dropoff_latitude      → Float
└── store_and_fwd_flag    → Categorical
```

### Validation Rules

The data validation process will later verify that:

1. Each required column can be converted to its expected logical type.
2. Datetime columns contain valid datetime values.
3. Numeric columns contain valid numeric values.
4. Integer fields contain valid integer-compatible values.
5. Categorical fields contain valid categorical values defined in the data contract.
6. Records that cannot satisfy the expected data type will be flagged as invalid rather than silently discarded.

### Important Principle

The data type definition is part of the **data contract**, not a cleaning decision.

At this stage:

```text
Raw Data
   ↓
Expected Logical Type
   ↓
Validation
   ↓
Valid / Invalid
```

No records are removed or modified as part of Step 2.3.

**Status: Step 2.3 Complete ✅**


### 2.4 – Define Categorical Value Rules

Categorical fields must contain only values that are defined as valid by the data contract.

The complete training dataset was inspected to identify all observed values for:

- `vendor_id`
- `store_and_fwd_flag`

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv', usecols=['vendor_id','store_and_fwd_flag']); print('=== vendor_id ==='); print(train['vendor_id'].value_counts(dropna=False).sort_index()); print('\n=== store_and_fwd_flag ==='); print(train['store_and_fwd_flag'].value_counts(dropna=False).sort_index())"
```

#### Output

```text
=== vendor_id ===
vendor_id
1    678342
2    780302
Name: count, dtype: int64

=== store_and_fwd_flag ===
store_and_fwd_flag
N    1450599
Y       8045
Name: count, dtype: int64
```

### Observations

The complete training dataset contains:

#### `vendor_id`

```text
Valid values:
1
2
```

No other `vendor_id` values were observed.

#### `store_and_fwd_flag`

```text
Valid values:
N
Y
```

No other values were observed.

The distribution is:

| Column | Value | Count |
|---|---:|---:|
| `vendor_id` | `1` | 678,342 |
| `vendor_id` | `2` | 780,302 |
| `store_and_fwd_flag` | `N` | 1,450,599 |
| `store_and_fwd_flag` | `Y` | 8,045 |

### Categorical Data Contract

Based on the complete training dataset, the following categorical rules are defined:

```text
vendor_id
    → Allowed values: {1, 2}

store_and_fwd_flag
    → Allowed values: {N, Y}
```

Any future record containing a value outside these defined categories should be treated as an **invalid categorical value** and flagged by the validation process.

### Validation Rule

```text
vendor_id
    ├── 1 → Valid
    └── 2 → Valid

store_and_fwd_flag
    ├── N → Valid
    └── Y → Valid
```

Values outside these sets must not be silently converted or ignored.

### Important Principle

The valid categories were derived from the **complete raw training dataset**, rather than assuming values based only on the 10% development dataset.

The 10% development dataset was previously validated as representative of the full training dataset in Step 1.11, but the complete dataset remains the source of truth for defining the data contract.

**Status: Step 2.4 Complete ✅**

### 2.5 – Define Numeric Range Rules

Numeric fields were inspected using the complete raw training dataset to understand their observed ranges before defining validation rules.

#### Command

```powershell
python -c "import pandas as pd; cols=['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']; train=pd.read_csv('data/raw/train.csv',usecols=cols); print(train[cols].describe().T[['count','min','25%','50%','75%','max']].to_string())"
```

#### Output

```text
                       count         min         25%         50%          75%           max
passenger_count    1458644.0    0.000000    1.000000    1.000000     2.000000  9.000000e+00
pickup_longitude   1458644.0 -121.933342  -73.991867  -73.981743   -73.967331 -6.133553e+01
pickup_latitude    1458644.0   34.359695   40.737347   40.754101    40.768360  5.188108e+01
dropoff_longitude  1458644.0 -121.933304  -73.991325  -73.979752   -73.963013 -6.133553e+01
dropoff_latitude   1458644.0   32.181141   40.735885   40.754524    40.769810  4.392103e+01
trip_duration      1458644.0    1.000000  397.000000  662.000000  1075.000000  3.526282e+06
```

### Observations

The complete dataset shows the following observed ranges:

| Column | Observed Minimum | Observed Maximum | Initial Observation |
|---|---:|---:|---|
| `passenger_count` | 0 | 9 | Contains `0`, which requires investigation |
| `pickup_longitude` | -121.933342 | -61.335529 | Most values are concentrated around NYC |
| `pickup_latitude` | 34.359695 | 51.881084 | Most values are concentrated around NYC |
| `dropoff_longitude` | -121.933304 | -61.335529 | Most values are concentrated around NYC |
| `dropoff_latitude` | 32.181141 | 43.921028 | Most values are concentrated around NYC |
| `trip_duration` | 1 sec | 3,526,282 sec | Contains extremely large values requiring investigation |

The percentile values show that the majority of the records are concentrated around the expected NYC geographic region.

For example:

```text
Pickup longitude
25%    → -73.991867
Median → -73.981743
75%    → -73.967331

Pickup latitude
25%    → 40.737347
Median → 40.754101
75%    → 40.768360
```

However, the observed minimum and maximum values show that the dataset contains geographic outliers.

Similarly, `trip_duration` has a maximum of **3,526,282 seconds**, which is far above the typical trip durations observed in the dataset.

These values will be investigated during the data validation and cleaning stage rather than being removed based only on their extreme values.

---

### Initial Numeric Data Contract

The following fundamental validity rules are defined.

#### `passenger_count`

```text
Rule:
passenger_count > 0
```

A taxi trip should contain at least one passenger.

Values equal to `0` or negative values will therefore be considered invalid.

The maximum observed value is `9`. No upper business limit is imposed at this stage; the observed value will be investigated during the cleaning stage if necessary.

---

#### Geographic Coordinates

Latitude and longitude must represent valid geographic coordinates.

```text
Latitude:
-90 <= latitude <= 90

Longitude:
-180 <= longitude <= 180
```

These are fundamental geographic validity limits.

The observed NYC-specific distribution will be investigated separately before introducing a tighter geographic boundary. Therefore, an unusual coordinate within the valid global range will **not automatically be removed** at this stage.

This distinction is important:

```text
Globally invalid coordinate
        ↓
Definitely invalid

Valid global coordinate
        ↓
May still be geographically unusual
        ↓
Investigate before removing
```

---

#### `trip_duration`

```text
Rule:
trip_duration > 0
```

A taxi trip must have a positive duration.

The dataset contains extremely large values, including:

```text
Maximum observed value = 3,526,282 seconds
```

These values are not automatically removed at this stage.

They will be investigated using the relationship between:

- `pickup_datetime`
- `dropoff_datetime`
- `trip_duration`

before a final upper-bound rule is defined.

---

### Numeric Data Contract Summary

| Column | Validation Rule | Status |
|---|---|---|
| `passenger_count` | `> 0` | Defined |
| `pickup_longitude` | `-180 <= value <= 180` | Defined |
| `pickup_latitude` | `-90 <= value <= 90` | Defined |
| `dropoff_longitude` | `-180 <= value <= 180` | Defined |
| `dropoff_latitude` | `-90 <= value <= 90` | Defined |
| `trip_duration` | `> 0` | Defined |
| Geographic NYC-specific boundary | To be investigated | Pending |
| Maximum valid `trip_duration` | To be investigated | Pending |

### Important Principle

Extreme values are **not automatically considered invalid**.

The validation process will distinguish between:

```text
Invalid by definition
        ↓
Can be rejected by the contract

Unusual but technically valid
        ↓
Requires investigation
        ↓
Cleaning decision made later
```

This prevents us from silently deleting potentially useful observations without understanding why they exist.

**Status: Step 2.5 Complete ✅**

### 2.6 – Define Datetime Consistency Rules

Datetime fields are critical for this project because `pickup_datetime`, `dropoff_datetime`, and `trip_duration` are directly related.

The complete raw training dataset was validated to establish datetime range and consistency rules.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['pickup_datetime','dropoff_datetime','trip_duration'],parse_dates=['pickup_datetime','dropoff_datetime']); calculated=(train['dropoff_datetime']-train['pickup_datetime']).dt.total_seconds(); print('=== DATETIME RANGE ==='); print('Pickup:',train['pickup_datetime'].min(),'to',train['pickup_datetime'].max()); print('Dropoff:',train['dropoff_datetime'].min(),'to',train['dropoff_datetime'].max()); print('\n=== DATETIME ORDER ==='); print('Dropoff <= Pickup:',(train['dropoff_datetime']<=train['pickup_datetime']).sum()); print('\n=== DURATION CONSISTENCY ==='); print('Calculated duration = trip_duration:',(calculated==train['trip_duration']).sum()); print('Mismatch:',(calculated!=train['trip_duration']).sum()); print('\n=== DURATION DIFFERENCE ==='); diff=(train['trip_duration']-calculated).abs(); print(diff.describe())"
```

#### Output

```text
=== DATETIME RANGE ===
Pickup: 2016-01-01 00:00:17 to 2016-06-30 23:59:39
Dropoff: 2016-01-01 00:03:31 to 2016-07-01 23:02:03

=== DATETIME ORDER ===
Dropoff <= Pickup: 0

=== DURATION CONSISTENCY ===
Calculated duration = trip_duration: 1458644
Mismatch: 0

=== DURATION DIFFERENCE ===
count    1458644.0
mean           0.0
std            0.0
min            0.0
25%            0.0
50%            0.0
75%            0.0
max            0.0
dtype: float64
```

### Observations

#### Datetime Range

The training dataset covers:

```text
Pickup:
2016-01-01 00:00:17
        ↓
2016-06-30 23:59:39

Dropoff:
2016-01-01 00:03:31
        ↓
2016-07-01 23:02:03
```

The dropoff range extending into July is expected because trips that begin on June 30 can finish after midnight on July 1.

---

#### Datetime Ordering

The following validation was performed:

```text
dropoff_datetime <= pickup_datetime
```

Result:

```text
0 records
```

Therefore, every training record satisfies:

```text
pickup_datetime < dropoff_datetime
```

---

#### Trip Duration Consistency

The duration calculated from the datetime fields was compared against the existing `trip_duration` value:

```text
calculated_duration =
    dropoff_datetime - pickup_datetime
```

Result:

```text
Total training records : 1,458,644
Exact matches           : 1,458,644
Mismatches              : 0
```

The absolute difference between the calculated duration and `trip_duration` was also:

```text
Mean   : 0
Std    : 0
Min    : 0
Median : 0
Max    : 0
```

This confirms that:

```text
trip_duration
      =
(dropoff_datetime - pickup_datetime)
```

for **every training record**.

---

### Datetime Data Contract

Based on the complete dataset validation, the following rules are defined:

```text
1. pickup_datetime must be a valid datetime.

2. dropoff_datetime must be a valid datetime.

3. pickup_datetime must be earlier than dropoff_datetime.

4. trip_duration must be greater than 0.

5. trip_duration must equal:
   dropoff_datetime - pickup_datetime
   measured in seconds.
```

### Validation Relationship

```text
pickup_datetime
       │
       │
       ▼
   Taxi Trip
       │
       ▼
dropoff_datetime
       │
       │
       ▼
Calculated Duration
       │
       │
       ▼
Must exactly match
trip_duration
```

### Important Observation

This validation also provides an explanation for the extremely large `trip_duration` values observed during Step 1.

Since `trip_duration` exactly matches the datetime difference for all training records, those unusually large target values are not caused by a calculation mismatch.

They should therefore be investigated as **potential data-quality or outlier records** during the later cleaning stage rather than being silently corrected at this stage.

### Important Principle

The datetime consistency rule is based on the complete training dataset and provides a deterministic validation relationship:

```text
Expected Duration =
dropoff_datetime - pickup_datetime
```

Any future training record where the calculated duration does not match `trip_duration` will be considered a **datetime/target consistency violation** and should be flagged for investigation.

No records are removed or modified as part of Step 2.6.

**Status: Step 2.6 Complete ✅**

### 2.7 – Define Target Validation Rules

The target variable for this project is:

```text
trip_duration
```

The complete raw training dataset was inspected to establish target validity rules and understand the distribution of trip durations.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['trip_duration']); d=train['trip_duration']; print('=== TARGET VALIDATION ==='); print('Missing:',d.isna().sum()); print('Zero:',(d==0).sum()); print('Negative:',(d<0).sum()); print('\n=== DURATION RANGES ==='); print('<= 5 min:',(d<=300).sum()); print('5–15 min:',((d>300)&(d<=900)).sum()); print('15–30 min:',((d>900)&(d<=1800)).sum()); print('30–60 min:',((d>1800)&(d<=3600)).sum()); print('1–2 hours:',((d>3600)&(d<=7200)).sum()); print('> 2 hours:',(d>7200).sum()); print('> 24 hours:',(d>86400).sum()); print('\n=== TOP 20 LONGEST TRIPS ==='); print(d.nlargest(20).to_string(index=False))"
```

#### Output

```text
=== TARGET VALIDATION ===
Missing: 0
Zero: 0
Negative: 0

=== DURATION RANGES ===
<= 5 min: 221916
5–15 min: 746560
15–30 min: 377050
30–60 min: 100801
1–2 hours: 10064
> 2 hours: 2253
> 24 hours: 4

=== TOP 20 LONGEST TRIPS ===
3526282
2227612
2049578
1939736
  86392
  86391
  86390
  86387
  86385
  86379
  86378
  86378
  86377
  86377
  86369
  86369
  86369
  86369
  86367
  86367
```

### Observations

The complete training dataset contains **no missing, zero, or negative target values**.

```text
Missing values : 0
Zero values    : 0
Negative values: 0
```

Therefore, the basic target validity rule is clearly supported:

```text
trip_duration > 0
```

### Target Distribution

The majority of trips fall within the shorter duration ranges:

| Duration Range | Number of Records |
|---|---:|
| <= 5 minutes | 221,916 |
| 5–15 minutes | 746,560 |
| 15–30 minutes | 377,050 |
| 30–60 minutes | 100,801 |
| 1–2 hours | 10,064 |
| > 2 hours | 2,253 |
| > 24 hours | 4 |

Only **4 records** have a duration greater than 24 hours.

### Extreme Target Values

The four longest trips are:

```text
3,526,282 seconds
2,227,612 seconds
2,049,578 seconds
1,939,736 seconds
```

There is a significant gap between these four observations and the next-longest values:

```text
Next longest values:
86,392
86,391
86,390
86,387
...
```

This indicates that the four extremely large values are highly unusual observations.

However, they are **not automatically classified as invalid at this stage**.

Step 2.6 established that `trip_duration` exactly matches:

```text
dropoff_datetime - pickup_datetime
```

for all 1,458,644 training records.

Therefore, the extreme values must be investigated together with their corresponding datetime values before deciding whether they should be removed.

---

### Target Data Contract

The following deterministic validation rules are defined:

```text
1. trip_duration must be present.

2. trip_duration must be numeric.

3. trip_duration must be greater than 0.

4. trip_duration must be consistent with:
   dropoff_datetime - pickup_datetime.
```

### Upper Bound

No fixed upper bound for `trip_duration` is defined yet.

Although only four records exceed 24 hours, an upper limit should not be introduced arbitrarily.

The extreme records will be investigated during the data cleaning/validation stage before deciding whether they are:

- Genuine but unusual observations
- Data-quality issues
- Records requiring removal
- Records requiring special treatment

### Important Principle

```text
Extreme value
      ↓
Not automatically invalid
      ↓
Investigate
      ↓
Understand the reason
      ↓
Make a documented cleaning decision
```

This prevents valid but unusual observations from being silently removed.

### Target Validation Summary

| Rule | Status |
|---|---|
| Target must exist | Defined |
| Target must be numeric | Defined |
| `trip_duration > 0` | Defined |
| Target must match datetime difference | Defined |
| Fixed maximum duration | Pending investigation |

No target records are removed or modified as part of Step 2.7.

**Status: Step 2.7 Complete ✅**

### 2.8 – Define Missing-Value Rules

Missing-value validation is required to ensure that records contain all information necessary for data processing, feature engineering, and model training.

The complete raw TRAIN and TEST datasets were checked for missing values.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv'); test=pd.read_csv('data/raw/test.csv'); print('=== TRAIN MISSING VALUES ==='); print(train.isnull().sum()); print('\n=== TEST MISSING VALUES ==='); print(test.isnull().sum()); print('\n=== TRAIN TOTAL MISSING ===', train.isnull().sum().sum()); print('=== TEST TOTAL MISSING  ===', test.isnull().sum().sum())"
```

#### Output

```text
=== TRAIN MISSING VALUES ===
id                    0
vendor_id             0
pickup_datetime       0
dropoff_datetime      0
passenger_count       0
pickup_longitude      0
pickup_latitude       0
dropoff_longitude     0
dropoff_latitude      0
store_and_fwd_flag    0
trip_duration         0
dtype: int64

=== TEST MISSING VALUES ===
id                    0
vendor_id             0
pickup_datetime       0
passenger_count       0
pickup_longitude      0
pickup_latitude       0
dropoff_longitude     0
dropoff_latitude      0
store_and_fwd_flag    0
dtype: int64

=== TRAIN TOTAL MISSING === 0
=== TEST TOTAL MISSING  === 0
```

### Observations

No missing values were found in either dataset.

```text
TRAIN total missing values → 0
TEST total missing values  → 0
```

Therefore, the current raw datasets are complete with respect to all required columns.

---

### Missing-Value Data Contract

All required columns must contain valid, non-missing values.

```text
Required column
      ↓
Missing?
      │
      ├── No  → Valid
      │
      └── Yes → Invalid / Flag for investigation
```

The following fields are considered required:

#### TRAIN

```text
id
vendor_id
pickup_datetime
dropoff_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
trip_duration
```

#### TEST

```text
id
vendor_id
pickup_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
```

### Missing-Value Handling Rule

No automatic imputation strategy is defined at this stage because the current raw dataset contains **zero missing values**.

If missing values are encountered during future pipeline execution:

- The affected records should be identified and counted.
- Required-field violations should be flagged.
- Records should not be silently deleted.
- Any imputation or removal decision must be explicitly justified and documented.

### Important Principle

```text
Missing value detected
        ↓
Identify affected column
        ↓
Determine whether the column is required
        ↓
Flag the violation
        ↓
Investigate the cause
        ↓
Document the cleaning decision
```

The validation contract therefore treats missing values as a **data-quality violation**, rather than silently filling or deleting them.

**Status: Step 2.8 Complete ✅**

### 2.9 – Define Duplicate Handling Rules

Duplicate records can lead to incorrect statistics, biased model training, and misleading validation results.

Two types of duplicates were checked in the complete raw training dataset:

1. Completely duplicated rows
2. Duplicate `id` values

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv'); print('=== ROW DUPLICATES ==='); print('Duplicate rows:', train.duplicated().sum()); print('\n=== ID DUPLICATES ==='); print('Duplicate IDs:', train['id'].duplicated().sum()); print('Unique IDs:', train['id'].nunique()); print('Total rows:', len(train)); print('\n=== ID DUPLICATES INCLUDING FIRST OCCURRENCE ==='); dup_ids=train[train['id'].duplicated(keep=False)]['id']; print('Rows belonging to duplicated IDs:', len(dup_ids)); print('Number of duplicated ID values:', dup_ids.nunique())"
```

#### Output

```text
=== ROW DUPLICATES ===
Duplicate rows: 0

=== ID DUPLICATES ===
Duplicate IDs: 0
Unique IDs: 1458644
Total rows: 1458644

=== ID DUPLICATES INCLUDING FIRST OCCURRENCE ===
Rows belonging to duplicated IDs: 0
Number of duplicated ID values: 0
```

### Observations

No duplicate records were found in the complete training dataset.

```text
Total rows            : 1,458,644
Unique IDs             : 1,458,644
Duplicate rows         : 0
Duplicate IDs          : 0
```

Therefore:

- Every training record has a unique `id`.
- No completely duplicated rows were found.
- No records share the same `id`.

---

### Duplicate Data Contract

The following rules are defined:

```text
1. Each record must have a unique `id`.

2. Completely duplicated rows are not allowed.

3. Duplicate IDs must be flagged as data-quality violations.

4. Duplicate records must not be silently removed.
```

### Duplicate Handling Principle

If duplicates are detected in future pipeline execution:

```text
Duplicate detected
       ↓
Identify duplicate type
       │
       ├── Exact duplicate row
       │
       └── Same ID with different values
       ↓
Investigate
       ↓
Determine correct handling
       ↓
Document the decision
```

The current dataset contains no duplicates, so no records require removal at this stage.

**Status: Step 2.9 Complete ✅**

### 2.10 – Define Invalid-Record Rules

The validation rules defined in Steps 2.3–2.9 were consolidated and applied to the complete raw training dataset.

The purpose of this validation is to identify records that violate the defined data contract before any cleaning or removal decision is made.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',parse_dates=['pickup_datetime','dropoff_datetime']); duration_calc=(train['dropoff_datetime']-train['pickup_datetime']).dt.total_seconds(); print('=== INVALID RECORD CHECK ==='); print('Missing required values:',train.isnull().any(axis=1).sum()); print('Invalid vendor_id:',(~train['vendor_id'].isin([1,2])).sum()); print('Invalid store_and_fwd_flag:',(~train['store_and_fwd_flag'].isin(['N','Y'])).sum()); print('Invalid passenger_count:',(train['passenger_count']<=0).sum()); print('Invalid pickup latitude:',(~train['pickup_latitude'].between(-90,90)).sum()); print('Invalid pickup longitude:',(~train['pickup_longitude'].between(-180,180)).sum()); print('Invalid dropoff latitude:',(~train['dropoff_latitude'].between(-90,90)).sum()); print('Invalid dropoff longitude:',(~train['dropoff_longitude'].between(-180,180)).sum()); print('Invalid trip_duration:',(train['trip_duration']<=0).sum()); print('Invalid datetime order:',(train['pickup_datetime']>=train['dropoff_datetime']).sum()); print('Duration mismatch:',(train['trip_duration']!=duration_calc).sum()); print('Duplicate IDs:',train['id'].duplicated().sum()); print('Duplicate rows:',train.duplicated().sum())"
```

#### Output

```text
=== INVALID RECORD CHECK ===
Missing required values: 0
Invalid vendor_id: 0
Invalid store_and_fwd_flag: 0
Invalid passenger_count: 60
Invalid pickup latitude: 0
Invalid pickup longitude: 0
Invalid dropoff latitude: 0
Invalid dropoff longitude: 0
Invalid trip_duration: 0
Invalid datetime order: 0
Duration mismatch: 0
Duplicate IDs: 0
Duplicate rows: 0
```

### Observations

The complete training dataset contains **1,458,644 records**.

The validation identified only **60 records** that violate the currently defined data contract.

All 60 violations are related to:

```text
passenger_count <= 0
```

All other validation rules currently have **zero violations**.

### Validation Summary

| Validation Rule | Invalid Records |
|---|---:|
| Missing required values | 0 |
| Invalid `vendor_id` | 0 |
| Invalid `store_and_fwd_flag` | 0 |
| `passenger_count <= 0` | **60** |
| Invalid pickup latitude | 0 |
| Invalid pickup longitude | 0 |
| Invalid dropoff latitude | 0 |
| Invalid dropoff longitude | 0 |
| `trip_duration <= 0` | 0 |
| Invalid datetime order | 0 |
| Duration mismatch | 0 |
| Duplicate IDs | 0 |
| Duplicate rows | 0 |

### Invalid-Record Contract

A record is considered invalid when it violates one or more of the following rules:

```text
1. Required value is missing.

2. vendor_id is not one of:
   {1, 2}

3. store_and_fwd_flag is not one of:
   {N, Y}

4. passenger_count <= 0

5. Latitude is outside:
   [-90, 90]

6. Longitude is outside:
   [-180, 180]

7. trip_duration <= 0

8. pickup_datetime >= dropoff_datetime

9. trip_duration does not match:
   dropoff_datetime - pickup_datetime

10. Duplicate ID exists.

11. Complete duplicate row exists.
```

### Current Dataset Status

```text
Total training records
        ↓
1,458,644
        ↓
Contract validation
        ↓
60 violations
        ↓
All 60 are passenger_count <= 0
```

The current validation confirms that the dataset is largely compliant with the defined data contract.

### Important Principle

The identification of invalid records does **not automatically mean that the records should be deleted**.

The 60 records with `passenger_count <= 0` will be investigated during the data cleaning stage before deciding whether to:

- Remove them
- Correct them
- Flag them
- Apply another documented treatment

Similarly, extreme values identified earlier, such as unusually large `trip_duration` values, are not included as invalid records merely because they are extreme. They require separate investigation.

Therefore:

```text
Validation
    ↓
Identify violations
    ↓
Investigate
    ↓
Document decision
    ↓
Clean / Transform
```

**Status: Step 2.10 Complete ✅**

### 2.11 – Create Data Validation Contract

The validation rules established during Steps 2.1–2.10 were consolidated into a formal data validation contract.

The contract acts as a single reference for the expected structure and validity rules of the training and test datasets.

### Contract Location

```text
configs/
└── data_contract.yaml
```

The contract defines:

```text
Schema
   ↓
Required Columns
   ↓
Expected Data Types
   ↓
Categorical Rules
   ↓
Numeric Rules
   ↓
Datetime Rules
   ↓
Target Rules
   ↓
Missing-Value Rules
   ↓
Duplicate Rules
```

### Contract Scope

The contract currently defines:

- Expected columns for TRAIN and TEST
- Required columns
- Expected logical data types
- Allowed `vendor_id` values: `{1, 2}`
- Allowed `store_and_fwd_flag` values: `{N, Y}`
- `passenger_count > 0`
- Valid latitude range: `-90 to +90`
- Valid longitude range: `-180 to +180`
- `trip_duration > 0`
- `pickup_datetime < dropoff_datetime`
- `trip_duration` must match the datetime-derived duration
- Required fields must not contain missing values
- `id` must be unique
- Complete duplicate rows are not allowed

### Contract Design Principle

The data contract defines **what constitutes a valid record**.

It does not define automatic cleaning actions.

For example:

```text
Validation Rule
      ↓
Violation detected
      ↓
Flag / Investigate
      ↓
Document decision
      ↓
Cleaning action
```

This prevents data from being silently deleted or modified without understanding the reason.

### Rules Intentionally Left Pending

Two areas are intentionally not given strict rules yet:

#### 1. NYC-Specific Geographic Boundary

Only the fundamental global latitude/longitude ranges are currently enforced.

A tighter NYC-specific boundary will be defined only after investigating the geographic outliers observed during Step 1.

#### 2. Maximum Trip Duration

No fixed upper limit is currently defined.

Although four records have durations greater than 24 hours, Step 2.6 confirmed that their `trip_duration` values exactly match the difference between `dropoff_datetime` and `pickup_datetime`.

These records therefore require further investigation before any upper-bound rule is introduced.

### Expected Deliverable

```text
configs/
└── data_contract.yaml
```

The YAML file is now the initial **single source of truth for Phase 3 data validation rules**.

**Status: Step 2.11 Complete ✅**

### 2.12 – Validate the Contract Against the Dataset

The formal data validation contract created in Step 2.11 was implemented and executed against the complete raw TRAIN and TEST datasets.

The validation script is located at:

```text
src/
└── validation/
    └── validate_contract.py
```

The script reads the rules from:

```text
configs/
└── data_contract.yaml
```

and validates the datasets without modifying the original raw data.

#### Command

```powershell
python src/validation/validate_contract.py
```

#### Output

```text
======================================================================
DATA CONTRACT VALIDATION REPORT
======================================================================
dataset                                            rule  invalid_records status
  TRAIN                        Required columns present                0   PASS
   TEST                        Required columns present                0   PASS
  TRAIN            No missing values in required fields                0   PASS
   TEST            No missing values in required fields                0   PASS
  TRAIN          vendor_id contains only allowed values                0   PASS
  TRAIN store_and_fwd_flag contains only allowed values                0   PASS
   TEST          vendor_id contains only allowed values                0   PASS
   TEST store_and_fwd_flag contains only allowed values                0   PASS
  TRAIN                             passenger_count > 0               60   FAIL
  TRAIN                pickup_latitude within [-90, 90]                0   PASS
  TRAIN             pickup_longitude within [-180, 180]                0   PASS
  TRAIN               dropoff_latitude within [-90, 90]                0   PASS
  TRAIN            dropoff_longitude within [-180, 180]                0   PASS
  TRAIN                               trip_duration > 0                0   PASS
   TEST                             passenger_count > 0               23   FAIL
   TEST                pickup_latitude within [-90, 90]                0   PASS
   TEST             pickup_longitude within [-180, 180]                0   PASS
   TEST               dropoff_latitude within [-90, 90]                0   PASS
   TEST            dropoff_longitude within [-180, 180]                0   PASS
  TRAIN              pickup_datetime < dropoff_datetime                0   PASS
  TRAIN       trip_duration matches datetime difference                0   PASS
  TRAIN                                  IDs are unique                0   PASS
   TEST                                  IDs are unique                0   PASS
  TRAIN                               No duplicate rows                0   PASS
   TEST                               No duplicate rows                0   PASS

======================================================================
RESULT: FAIL
Failed rules: 2
======================================================================
```

### Validation Observations

The majority of the defined data-contract rules passed successfully.

The validation identified two failing rules:

```text
TRAIN → passenger_count > 0 → 60 violations
TEST  → passenger_count > 0 → 23 violations
```

Therefore:

```text
TRAIN invalid passenger_count records → 60
TEST invalid passenger_count records  → 23
```

All other currently implemented validation rules passed.

### Validation Summary

| Validation Area | TRAIN | TEST |
|---|---:|---:|
| Required columns | PASS | PASS |
| Missing values | PASS | PASS |
| Categorical values | PASS | PASS |
| Latitude range | PASS | PASS |
| Longitude range | PASS | PASS |
| `passenger_count > 0` | **60 violations** | **23 violations** |
| `trip_duration > 0` | PASS | N/A |
| Datetime ordering | PASS | N/A |
| Duration consistency | PASS | N/A |
| Unique IDs | PASS | PASS |
| Duplicate rows | PASS | PASS |

### Important Finding

The validation of the TEST dataset revealed **23 records with `passenger_count <= 0`**.

This is important because Step 1 and the earlier TRAIN validation had already identified 60 such records in TRAIN, but we had not yet explicitly validated this rule against TEST.

The contract validation therefore successfully detected a data-quality issue across both datasets.

### No Automatic Cleaning

The validation script only identifies violations.

It does **not**:

- Delete invalid records
- Modify values
- Impute missing values
- Change the raw datasets

The 60 TRAIN records and 23 TEST records with invalid `passenger_count` values will be investigated during the **data cleaning stage in Step 3**.

The raw datasets remain unchanged.

### Current Contract Status

```text
Data Contract
      ↓
Validated against raw datasets
      ↓
Most rules → PASS
      ↓
23 TEST + 60 TRAIN passenger_count violations
      ↓
Requires investigation during cleaning
```

The `FAIL` result therefore represents **data-quality violations identified by the contract**, not a failure of the validation implementation.

### Step 2 Completion Status

The data contract has been successfully:

- Defined
- Documented
- Implemented
- Executed against the raw datasets
- Used to identify actual data-quality violations

The remaining violations are now known and will be addressed during the data cleaning stage rather than being silently removed.

**Status: Step 2.12 Complete ✅**

**Step 2 – Define the Data Contract & Validation Rules: COMPLETE ✅**
