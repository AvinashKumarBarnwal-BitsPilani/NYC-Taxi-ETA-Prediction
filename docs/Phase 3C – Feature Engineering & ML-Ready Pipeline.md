# Phase 3C - Feature Engineering & Reproducible Pipeline

## Table of Contents

- [Step 5 - Feature Engineering ✅](#step-51--understand-feature-engineering-input)
  - [5.1 Understand Feature Engineering Input](#step-51--understand-feature-engineering-input)
  - [5.2 Identify Prediction-Time Features](#step-52--identify-prediction-time-features)
  - [5.3 Create Datetime Features](#step-53--create-datetime-features)
  - [5.4a Inspect Geographic Inputs](#step-54a--inspect-geographic-inputs)
  - [5.4b Create Haversine Distance](#step-54b--create-haversine-distance)
  - [5.4c Analyze Geographic Feature](#step-54c--analyze-geographic-feature)
  - [5.5 Evaluate Existing Features](#step-55--evaluate-existing-features)
  - [5.6 Evaluate/Create Rush-Hour Feature](#step-56--evaluatecreate-rush-hour-feature)
  - [5.7 Check Feature Quality](#step-57--check-feature-quality)
  - [5.8 Check Feature Leakage](#step-58--check-feature-leakage)
  - [5.9 Feature Engineering Summary](#step-59--feature-engineering-summary)

- [Step 6 - Train / Validation Split & Preprocessing ✅](#step-6--train--validation-split--preprocessing)
  - [6.1 Define Train/Validation Split Strategy](#step-61--define-trainvalidation-split-strategy)
  - [6.2 Implement Train/Validation Split](#step-62--implement-trainvalidation-split)
  - [6.3 Feature Scaling & Normalization](#step-63--feature-scaling--normalization)
  - [6.4 Encoding Categorical Features](#step-64--encoding-categorical-features)
  - [6.5 Verify Preprocessed Datasets](#step-65--verify-preprocessed-datasets)

- [Step 7 - Build the Data Engineering Code Pipeline](#step-7--build-the-data-engineering-code-pipeline)
  - [7.1 Design Modular Pipeline Architecture](#step-71--design-modular-pipeline-architecture)
  - [7.2 Implement End-to-End Pipeline](#step-72--implement-end-to-end-pipeline)
  - [7.3 Pipeline Testing & Validation](#step-73--pipeline-testing--validation)
    - [7.3a Setup Automated Testing with pytest](#step-73a--setup-automated-testing-with-pytest)
    - [7.3b Test Production Modules](#step-73b--test-production-modules)
    - [7.3c Test Data & Pipeline Invariants](#step-73c--test-data--pipeline-invariants)
    - [7.3d Test Preprocessing & Leakage Controls](#step-73d--test-preprocessing--leakage-controls)
    - [7.3e End-to-End Pipeline Test](#step-73e--end-to-end-pipeline-test)

- [Step 8 - Build the DVC Pipeline](#step-8--build-the-dvc-pipeline)
  - [8.1 DVC Stage Definition](#step-81--dvc-stage-definition)
  - [8.2 Define Dependencies](#step-82--define-dependencies)
  - [8.3 Define Outputs](#step-83--define-outputs)
  - [8.4 Define Parameters](#step-84--define-parameters)
  - [8.5 Execute with dvc repro](#step-85--execute-with-dvc-repro)
  - [8.6 Reproducibility Verification](#step-86--reproducibility-verification)

- [Step 9 - Testing & Reproducibility](#step-9--testing--reproducibility)
  - [9.1 Run Complete Automated Test Suite](#91--run-complete-automated-test-suite)
  - [9.2 Validate DVC Pipeline Reproduction](#92--validate-dvc-pipeline-reproduction)
  - [9.3 Test Pipeline Failure Scenarios](#93--test-pipeline-failure-scenarios)
  - [9.4 Verify Data & Pipeline Reproducibility](#94--verify-data--pipeline-reproducibility)
  - [9.5 Final Phase 3 Engineering Validation](#95--final-phase-3-engineering-validation)

- [Step 10 - Phase 3 Integration & Handover](#step-10--phase-3-integration--handover)
  - [10.1 Final Phase 3 Integration Check](#step-101)
  - [10.2 Verify ML-Ready Data Handover](#step-102)
  - [10.3 Final Project Structure & Dependency Check](#step-103)
  - [10.4 Documentation & Evidence Review](#step-104)
  - [10.5 Phase 4 Training Interface Definition](#step-105)
  - [10.6 Phase 3 Completion & Handover](#step-106)

- [Step 11 - Post-Completion Engineering Update – Persisted Preprocessor](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-11---post-completion-engineering-update--persisted-preprocessor)



# 📊 Phase 3C – Feature Engineering & ML-Ready Pipeline

### Step 5.1 – Understand Feature Engineering Input

### Input Dataset

Feature engineering starts from the cleaned datasets generated during Phase 3B:

```
data/interim/
├── train_clean.csv
└── test_clean.csv
```

### Dataset Shape

| Dataset | Records | Columns |
|---|---:|---:|
| TRAIN | 1,458,542 | 11 |
| TEST | 625,092 | 9 |

### TRAIN Columns

```
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

### TEST Columns

```
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

### Prediction-Time Observation

`dropoff_datetime` and `trip_duration` are available only in TRAIN.

```
TRAIN
├── dropoff_datetime
└── trip_duration  ← Target

TEST
└── Neither is available
```

Therefore, `dropoff_datetime` must **not** be used for feature engineering because it would not be available when making a real prediction.

Similarly, `trip_duration` is the target variable and must never be used as an input feature.

Feature engineering will use information available at prediction time, particularly:

```
pickup_datetime
vendor_id
passenger_count
pickup coordinates
dropoff coordinates
store_and_fwd_flag
```

### Data Type Note

When the cleaned CSV files are reloaded with `pandas.read_csv()`, datetime columns are initially represented as `object`.

The feature-engineering pipeline will explicitly convert `pickup_datetime` to a datetime type before extracting datetime-based features.

### Conclusion

The cleaned TRAIN and TEST datasets are suitable inputs for feature engineering.

> **Only information available at prediction time should be used as a model feature.**

### Implementation

**Python Script:** [5.1-Understand-Feature-Engineering-Input.py](../scripts/5.%20Feature-Engineering/5.1-Understand-Feature-Engineering-Input.py)

---

## Step 5.2 – Identify Prediction-Time Features

### Objective

Before creating new features, identify which information is genuinely available at **prediction time** and can therefore be safely used by the ML model.

The key principle is:

> **Only use information that would be available when the trip starts.**

This step helps prevent **target leakage** and ensures that the model can be used realistically in production.

---

### 5.2.1 Train vs Test Schema

The cleaned TRAIN and TEST datasets were inspected to identify columns shared between both datasets and columns present only in TRAIN.

#### TRAIN

```
['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration']
```

#### TEST

```
['id', 'vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']
```

#### Columns present only in TRAIN

```
['dropoff_datetime', 'trip_duration']
```

These columns are unavailable in the TEST dataset and are also not available at prediction time.

---

### 5.2.2 Feature Classification

Each existing column was classified based on its role in the prediction workflow.

| Column | Classification | Decision | Reason |
|---|---|---|---|
| `id` | Identifier | Exclude from model | Identifies the trip but does not represent meaningful trip characteristics |
| `vendor_id` | Candidate feature | Keep for evaluation | May capture differences between taxi vendors |
| `pickup_datetime` | Feature source | Keep | Available when the trip starts and can generate useful time-based features |
| `dropoff_datetime` | Leakage / unavailable | Exclude | Only known after the trip ends |
| `passenger_count` | Candidate feature | Keep for evaluation | Available at trip start and may influence trip characteristics |
| `pickup_longitude` | Feature source | Keep | Represents pickup location |
| `pickup_latitude` | Feature source | Keep | Represents pickup location |
| `dropoff_longitude` | Feature source | Keep | Represents destination location |
| `dropoff_latitude` | Feature source | Keep | Represents destination location |
| `store_and_fwd_flag` | Candidate feature | Keep for evaluation | Available as part of the trip record and may contain predictive information |
| `trip_duration` | Target | Exclude from features | This is the value being predicted |

---

### 5.2.3 Feature Sources vs Final Features

Some raw columns will not necessarily be used directly by the final model.

Instead, they can act as **feature sources** from which more meaningful features are derived.

For example:

```
pickup_datetime
    │
    ├── pickup_hour
    ├── pickup_day_of_week
    ├── pickup_month
    ├── pickup_year
    └── is_weekend
```

Similarly:

```
Pickup Coordinates + Dropoff Coordinates
    │
    ▼
Geographic Features
    │
    └── Distance
```

Therefore, feature engineering will transform raw prediction-time information into useful model features.

---

### 5.2.4 Identifier Handling

The `id` column will be retained for **record identification and tracking**, but will not be provided to the ML model.

```
id → Keep in dataset
id → Exclude from model features
```

An identifier does not inherently describe the trip and could cause the model to learn accidental dataset-specific patterns.

---

### 5.2.5 Leakage Handling

Two TRAIN columns are unavailable at prediction time:

```
dropoff_datetime → Information known after the trip → EXCLUDE

trip_duration    → Target variable               → EXCLUDE
```

The model must never use either column as an input feature.

---

### 5.2.6 Preliminary Feature Set

At the end of Step 5.2, the following columns have been identified as prediction-time inputs or candidate feature sources:

```
vendor_id
pickup_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
```

The following are excluded from model features:

```
id
dropoff_datetime
trip_duration
```

The candidate features have **not yet been proven to be useful**. Their predictive usefulness will be evaluated during the subsequent feature engineering steps.

---

### 5.2.7 Prediction-Time Mental Model

```
                 Prediction Time
                     │
                     ▼
        ┌──────────────────────────────┐
        │ Information Available        │
        ├──────────────────────────────┤
        │ pickup_datetime              │
        │ pickup location              │
        │ dropoff location             │
        │ passenger_count              │
        │ vendor_id                    │
        │ store_and_fwd_flag           │
        └──────────────────────────────┘
                     │
                     ▼
              Feature Engineering
                     │
                     ▼
                ML Model Input
                     │
                     ▼
           Predicted trip_duration
```

Information that becomes available only after the trip has completed must not flow into the model.

---

### 5.2.8 Step 5.2 Conclusion

The prediction-time feature boundary has been established.

#### Candidate Inputs

```
vendor_id
pickup_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
```

#### Excluded

```
id
dropoff_datetime
```

#### Target

```
trip_duration
```

This establishes a leakage-safe foundation for the next stage of feature engineering.

### Implementation

**Python Script:** [5.2–Identify Prediction-Time-Features.py](../scripts/5.%20Feature-Engineering/5.2–Identify%20Prediction-Time-Features.py)

**Status: Step 5.2 Complete ✅**

---

## Step 5.3 – Create Datetime Features

### Objective

Convert `pickup_datetime` into meaningful temporal features that may help the model learn time-dependent patterns in taxi trip duration.

The principle is:

> **Extract meaningful information from the timestamp without unnecessarily creating redundant features.**

---

### 5.3.1 Datetime Validation

`pickup_datetime` is initially loaded as `object` from the CSV files.

It was successfully converted using `pd.to_datetime()`.

```
TRAIN invalid datetime conversions: 0
TEST invalid datetime conversions : 0
```

Both TRAIN and TEST cover:

```
2016-01-01 → 2016-06-30
```

Therefore, the datetime values are valid and require no additional cleaning.

---

### 5.3.2 Candidate Datetime Features

The following features were evaluated:

| Feature | Decision | Reason |
|---|---|---|
| `pickup_hour` | Keep | Trip duration varies significantly by time of day |
| `pickup_day_of_week` | Keep | Trip patterns differ across weekdays |
| `pickup_month` | Keep | Shows variation across the six months in the dataset |
| `pickup_year` | Exclude | All records belong to 2016, making it a constant feature |
| `is_weekend` | Keep as candidate | Provides an explicit weekday/weekend distinction |
| `pickup_day` | Exclude for now | Limited direct predictive meaning; avoid unnecessary features |

---

### 5.3.3 Observed Relationship with Trip Duration

Group-level analysis showed meaningful variation in trip duration.

#### By Hour

Mean trip duration varied from approximately:

```
05:00 → 822.59 seconds
15:00 → 1118.88 seconds
```

This supports keeping:

```
pickup_hour
```

#### By Day of Week

Mean trip duration varied from approximately:

```
Monday    → 897.30 seconds
Thursday  → 1006.57 seconds
```

This supports keeping:

```
pickup_day_of_week
```

#### By Month

Mean trip duration varied from approximately:

```
January → 904.26 seconds
June    → 1013.02 seconds
```

This supports keeping:

```
pickup_month
```

#### Weekday vs Weekend

```
Weekday → 968.67 seconds
Weekend → 912.91 seconds
```

This indicates a meaningful difference between weekday and weekend trips.

Therefore:

```
is_weekend → Keep as candidate
```

These analyses are exploratory only. `trip_duration` is used to analyze candidate features, not to create them, so the feature-generation process itself does not introduce target leakage.

---

### 5.3.4 Final Datetime Feature Set

The initial datetime feature set is:

```
pickup_datetime
    │
    ├── pickup_hour          ✅
    ├── pickup_day_of_week   ✅
    ├── pickup_month         ✅
    ├── pickup_year          ❌
    ├── is_weekend           ✅
    └── pickup_day           ❌
```

#### Selected features

```
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
```

#### Excluded features

```
pickup_year
pickup_day
```

The raw `pickup_datetime` remains the source column during feature engineering but is not intended to be used directly as a final model feature.

---

### 5.3.5 Cyclic Encoding

Time-related features such as `pickup_hour` and `pickup_day_of_week` are inherently cyclic.

For example:

```
23:00 → 00:00
Sunday → Monday
```

are adjacent in real time but appear numerically far apart.

Cyclic transformations such as:

```
sin(hour)
cos(hour)
```

may be considered later depending on the model used in Phase 4.

For the current feature-engineering stage, the original interpretable features are retained.

---

### 5.3.6 Step 5.3 Conclusion

Datetime feature engineering has been defined based on:

- Data validity
- Prediction-time availability
- Domain intuition
- Observed target relationships
- Avoiding unnecessary features

#### Final datetime features

```
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
```

#### Excluded

```
pickup_year
pickup_day
```

The datetime feature-generation logic has been implemented separately from the exploratory target analysis.

### Implementation

**Python Scripts:**
- [5.3a-Create-Datetime-Features.py](../scripts/5.%20Feature-Engineering/5.3a-Create-Datetime-Features.py) – Feature generation
- [5.3b-Analyze-Datetime-Features.py](../scripts/5.%20Feature-Engineering/5.3b-Analyze-Datetime-Features.py) – Feature analysis

**Status: Step 5.3 Complete ✅**

---

# 5.4 Geographic Features

 ### 5.4a Inspect Geographic Inputs      
 ### 5.4b Create Haversine Distance      
 ### 5.4c Analyze Geographic Feature


## Step 5.4a – Inspect Geographic Inputs

### Objective

Before creating geographic features, validate the cleaned pickup and dropoff coordinate columns.

The four geographic input columns are:

```
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
```

The goal is to confirm that these inputs are suitable for geographic feature engineering.

---

### 5.4a.1 Data Type Validation

All four geographic columns are stored as `float64` in both datasets.

```
pickup_longitude     → float64
pickup_latitude      → float64
dropoff_longitude    → float64
dropoff_latitude     → float64
```

This is appropriate for geographic calculations.

---

### 5.4a.2 Missing Value Check

No missing geographic values were found.

```
TRAIN:
pickup_longitude     → 0
pickup_latitude      → 0
dropoff_longitude    → 0
dropoff_latitude     → 0

TEST:
pickup_longitude     → 0
pickup_latitude      → 0
dropoff_longitude    → 0
dropoff_latitude     → 0
```

Therefore, no additional missing-value handling is required for these columns.

---

### 5.4a.3 Geographic Range Check

The cleaned coordinate ranges are:

#### TRAIN

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| `pickup_longitude` | -74.726715 | -72.421227 |
| `pickup_latitude` | 40.099789 | 41.696796 |
| `dropoff_longitude` | -74.775429 | -72.421227 |
| `dropoff_latitude` | 40.153744 | 41.693352 |

#### TEST

| Coordinate | Minimum | Maximum |
|---|---:|---:|
| `pickup_longitude` | -74.677666 | -73.202621 |
| `pickup_latitude` | 40.298828 | 41.670166 |
| `dropoff_longitude` | -74.676003 | -73.068398 |
| `dropoff_latitude` | 40.270176 | 41.499668 |

The coordinates are within the expected cleaned geographic region.

---

### 5.4a.4 Sample Coordinate Inspection

Sample records show plausible pickup and dropoff coordinates:

```
pickup_longitude  pickup_latitude  dropoff_longitude  dropoff_latitude
-73.982155        40.767937        -73.964630          40.765602
-73.980415        40.738564        -73.999481          40.731152
-73.979027        40.763939        -74.005333          40.710087
```

No obvious coordinate formatting or value issues were observed.

---

### 5.4a.5 Identical Pickup and Dropoff Locations

The analysis found:

```
TRAIN records with identical pickup/dropoff coordinates: 5,871
```

Identical coordinates do not automatically indicate corrupted data. A taxi trip can legitimately have the same recorded pickup and dropoff location due to short trips, location precision, or GPS/data-recording behavior.

Therefore, these records will **not be removed at this stage**.

This follows the project principle:

> **An unusual value should not be removed without sufficient evidence that it is invalid.**

---

### 5.4a.6 Decision

The geographic inputs are suitable for the next stage of feature engineering.

```
Geographic Inputs
    │
    ├── Correct numeric type       ✅
    ├── Missing values             0
    ├── Reasonable coordinate range ✅
    ├── Sample values plausible    ✅
    └── Identical locations        Retain
                                    │
                                    ▼
                         Ready for geographic
                            feature creation
```

The primary geographic feature to investigate next is:

```
pickup coordinates
    +
dropoff coordinates
    ↓
Haversine distance
    ↓
distance_km
```

### Implementation

**Python Script:** [5.4a-Inspect-Geographic-Features.py](../scripts/5.%20Feature-Engineering/5.4a-Inspect-Geographic-Features.py)

**Status: Step 5.4a Complete ✅**

---

## Step 5.4b – Create Haversine Distance

### Objective

Create a meaningful geographic feature representing the distance between pickup and dropoff locations.

The four raw coordinate columns:

```
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
```

are transformed into:

```
distance_km
```

using the **Haversine formula**.

---

### 5.4b.1 Why Haversine Distance?

Latitude and longitude are geographic coordinates, so a simple Euclidean distance is not an appropriate physical distance measure.

The Haversine formula calculates the great-circle distance between two points on the Earth's surface.

Conceptually:

```
Pickup Coordinates
    +
Dropoff Coordinates
    ↓
Haversine Formula
    ↓
distance_km
```

The resulting feature represents the approximate geographic distance between the pickup and dropoff locations in kilometers.

---

### 5.4b.2 Implementation

The same Haversine calculation was applied independently to TRAIN and TEST.

```
TRAIN coordinates ──┐
                    ├── Haversine → distance_km
TEST coordinates ───┘
```

The calculation uses only geographic coordinates and does not use `trip_duration`.

Therefore, the feature-generation process does not introduce target leakage.

---

### 5.4b.3 Sample Results

Example TRAIN records produced plausible distances:

| Pickup → Dropoff | `distance_km` |
|---|---:|
| Record 1 | 1.499 |
| Record 2 | 1.806 |
| Record 3 | 6.385 |
| Record 4 | 1.485 |
| Record 5 | 1.189 |

TEST records were also successfully transformed.

---

### 5.4b.4 Distance Distribution

#### TRAIN

| Statistic | Value (km) |
|---|---:|
| Count | 1,458,542 |
| Mean | 3.436 |
| Std | 3.948 |
| Minimum | 0.000 |
| 25% | 1.232 |
| Median | 2.094 |
| 75% | 3.875 |
| Maximum | 116.496 |

#### TEST

| Statistic | Value (km) |
|---|---:|
| Count | 625,092 |
| Mean | 3.429 |
| Std | 3.929 |
| Minimum | 0.000 |
| 25% | 1.232 |
| Median | 2.094 |
| 75% | 3.883 |
| Maximum | 117.428 |

The TRAIN and TEST distributions are very similar, which indicates consistent geographic feature generation across both datasets.

---

### 5.4b.5 Zero-Distance Trips

The calculated feature contains:

```
TRAIN: 5,871 zero-distance trips
TEST : 2,459 zero-distance trips
```

These correspond to records where pickup and dropoff coordinates are identical.

A zero geographic distance does not automatically mean the record is invalid. Therefore, these records are retained at this stage.

---

### 5.4b.6 Decision

The following feature is successfully created:

```
distance_km
```

It will be retained as a **candidate geographic feature** for the ML pipeline.

The extreme maximum distances (~116–117 km) are not removed at this stage. Distance-based filtering will only be considered if further analysis provides sufficient evidence that these observations are invalid.

---

### 5.4b.7 Feature Engineering Flow

```
Pickup Coordinates
    │
    ├──────────────────────┬
    │                      │             
    ▼                      ▼           ▼
Latitude                Longitude
    │                      │
    └──────────────────────┼
                  │
                  ▼
          Haversine Formula
                  │
                  ▼
              distance_km
```

### Implementation

**Python Script:** [5.4b-Create-Haversine-Distance.py](../scripts/5.%20Feature-Engineering/5.4b-Create-Haversine-Distance.py)

**Status: Step 5.4b Complete ✅**

---

## Step 5.4c – Analyze Geographic Feature

### Objective

Evaluate whether `distance_km` has a meaningful relationship with `trip_duration` and inspect zero-distance and extreme-distance trips for potential anomalies.

---

### 5.4c.1 Trip Duration by Distance Range

Trip duration increases consistently as geographic distance increases.

| Distance Range | Records | Mean Duration (sec) | Median Duration (sec) |
|---|---:|---:|---:|
| 0–1 km | 245,863 | 431.52 | 268.0 |
| 1–2 km | 451,218 | 640.46 | 476.0 |
| 2–5 km | 498,844 | 1,007.86 | 815.0 |
| 5–10 km | 173,505 | 1,562.74 | 1,329.0 |
| 10–20 km | 65,266 | 2,242.72 | 1,916.0 |
| 20+ km | 23,846 | 3,114.54 | 2,737.0 |

The relationship is clear:

```
distance_km ↑
    ↓
trip_duration generally ↑
```

This provides strong domain and empirical evidence that `distance_km` is a useful candidate feature.

---

### 5.4c.2 Zero-Distance Trips

There are:

```
TRAIN: 5,871 zero-distance trips
```

Their trip-duration statistics are:

```
Mean   : 536.28 sec
Median : 351.00 sec
Minimum: 1 sec
Maximum: 86,352 sec
```

Zero geographic distance does not automatically indicate an invalid record because identical GPS coordinates can occur due to location precision or recording behavior.

Therefore, these records are **retained at this stage**.

However, the very high maximum duration is notable and will not be ignored. It can be considered during later outlier/error analysis if required.

---

### 5.4c.3 Distance vs Trip Duration Correlation

The Pearson correlation is:

```
distance_km ↔ trip_duration
Correlation = 0.1649
```

This indicates a **positive but relatively weak linear correlation**.

The relatively low correlation does not invalidate the feature because the relationship between distance and trip duration may not be purely linear and can be affected by factors such as traffic, time of day, and route characteristics.

The distance-range analysis provides stronger practical evidence of the relationship.

---

### 5.4c.4 Extreme-Distance Trips

The largest observed distance was:

```
TRAIN maximum distance ≈ 116.50 km
```

The top-distance records include trips between NYC-area coordinates and locations substantially farther away.

Most of these records also have relatively long trip durations, which is broadly consistent with the distance feature.

However, one notable record has:

```
distance_km   ≈ 112.55 km
trip_duration = 427 sec
```

This corresponds to an extremely long geographic distance combined with a very short recorded duration and should be considered a potential anomaly.

No automatic removal is performed at this stage because distance alone is not sufficient evidence that a record is invalid.

---

### 5.4c.5 Decision

The analysis supports retaining:

```
distance_km
```

as a candidate model feature.

#### Decision summary

```
distance_km
    │
    ├── Strong increasing pattern with trip duration → ✅ Keep
    │
    ├── Zero-distance records → Retain for now
    │
    ├── Extreme distances → Flag, don't blindly remove
    │
    └── Correlation = 0.1649 → Positive but weak linear relationship
```

The distance feature will therefore be included in the candidate feature set for subsequent preprocessing and model experimentation.

---

### 5.4c.6 Geographic Feature Engineering Conclusion

The geographic feature engineering process now has:

```
Raw Coordinates
    ↓
Coordinate Validation
    ↓
Haversine Distance
    ↓
distance_km
    ↓
Relationship Analysis
    ↓
Candidate Feature ✅
```

`distance_km` is considered a meaningful prediction-time feature because it is calculated entirely from pickup and dropoff coordinates available when the trip begins and shows a clear relationship with trip duration.

### Implementation

**Python Script:** [5.4c-Analyze-Geographic-Feature.py](../scripts/5.%20Feature-Engineering/5.4c-Analyze-Geographic-Feature.py)

**Status: Step 5.4c Complete ✅**

---

## Step 5.5 – Evaluate Existing Features

## Objective

Evaluate the existing candidate features that are available at prediction time but have not yet been transformed into derived features:

```
vendor_id
passenger_count
store_and_fwd_flag
```

The goal is to determine whether these features contain potentially useful information for predicting `trip_duration`.

---

## 5.5.1 Vendor ID

Two vendors are present:

```
Vendor 1 → 678,279 records
Vendor 2 → 780,263 records
```

Trip duration differs between vendors:

| Vendor | Records | Mean Duration (sec) | Median Duration (sec) |
|---|---:|---:|---:|
| 1 | 678,279 | 831.03 | 658.0 |
| 2 | 780,263 | 1,058.57 | 666.0 |

Vendor 2 has a substantially higher mean duration, while the medians are relatively close.

This indicates that `vendor_id` may contain useful predictive information.

**Decision: KEEP as a candidate feature.**

---

## 5.5.2 Passenger Count

The distribution is:

```
Passenger Count    Records
1                 1,033,514
2                 10,306
3                 359,894
4                 28,403
5                 78,087
6                 48,333
7                 8,191
```

The majority of trips have 1 passenger.

For the commonly represented values from 1–6 passengers, mean trip duration generally increases:

```
1 passenger → 922.96 sec
2 passengers → 995.71 sec
3 passengers → 1027.59 sec
4 passengers → 1053.54 sec
5 passengers → 1070.24 sec
6 passengers → 1061.36 sec
```

This suggests that `passenger_count` may contain useful predictive information.

Values 7, 8, and 9 are extremely rare, but were previously investigated during Phase 3B and retained because rarity alone was not sufficient evidence of invalidity.

**Decision: KEEP as a candidate feature.**

---

## 5.5.3 Store and Forward Flag

The distribution is highly imbalanced:

```
N → 1,450,500 records
Y →     8,042 records
```

Trip duration differs between the two categories:

| Flag | Records | Mean Duration (sec) | Median Duration (sec) |
|---|---:|---:|---:|
| N | 1,450,500 | 952.04 | 662.0 |
| Y | 8,042 | 1,081.16 | 813.0 |

Although `Y` is rare, it shows a noticeable difference in both mean and median trip duration.

The feature is therefore retained rather than removed solely because of its low frequency.

**Decision: KEEP as a candidate feature.**

---

## 5.5.4 Data Types

The current data types are:

```
vendor_id             int64
passenger_count       int64
store_and_fwd_flag    object
```

This will be handled later during **Step 6 – Preprocessing**.

In particular:

```
vendor_id             → categorical feature
passenger_count       → numerical/discrete feature
store_and_fwd_flag    → categorical feature
```

The exact encoding strategy will be defined during preprocessing.

---

## 5.5.5 Final Decision

All three existing candidate features are retained:

```
vendor_id             → KEEP
passenger_count       → KEEP
store_and_fwd_flag    → KEEP
```

No feature is removed based on this analysis.

The current candidate feature set is therefore:

```
Datetime Features
├── pickup_hour
├── pickup_day_of_week
├── pickup_month
└── is_weekend
Geographic Features
└── distance_km
Existing Features
├── vendor_id
├── passenger_count
└── store_and_fwd_flag
```

The final usefulness of these features will ultimately be validated through model experimentation in later phases.

### Implementation

**Python Script:** [5.5-Evaluate-Existing-Features.py](../scripts/5.%20Feature-Engineering/5.5-Evaluate-Existing-Features.py)

**Status: Step 5.5 Complete ✅**

---

## Step 5.6 – Evaluate/Create Rush-Hour Feature

## Objective

Evaluate whether a derived binary `rush_hour` feature provides useful information beyond the existing `pickup_hour` feature.

The candidate definition was:

```
Morning Rush : 07:00–09:59
Evening Rush : 16:00–18:59
```

Therefore:

```
rush_hour = 1
```

for hours:

```
7, 8, 9, 16, 17, 18
```

and `0` otherwise.

---

## 5.6.1 Trip Duration by Hour

Trip duration varies substantially across individual hours.

Examples:

```
05:00 → 822.59 sec
06:00 → 730.48 sec
14:00 → 1075.83 sec
15:00 → 1118.88 sec
16:00 → 1078.89 sec
17:00 → 1030.61 sec
18:00 → 981.61 sec
```

This confirms that `pickup_hour` already captures significant temporal variation.

---

## 5.6.2 Rush Hour vs Non-Rush Hour

| Category | Records | Mean Duration (sec) | Median Duration (sec) |
|---|---:|---:|---:|
| Non-rush | 1,036,853 | 946.24 | 660.0 |
| Rush | 421,689 | 968.79 | 668.0 |

The difference is:

```
Mean     → 22.55 seconds
Median   → 8 seconds
```

The difference exists, but it is relatively small compared with the variation observed across individual hours.

---

## 5.6.3 Feature Redundancy

The `rush_hour` feature is directly derived from:

```
pickup_hour
```

Therefore, both features contain overlapping information.

```
pickup_hour     │     ├── 07, 08, 09 → rush_hour = 1
     ├── 16, 17, 18 → rush_hour = 1
     └── other hours → rush_hour = 0
```

Since `pickup_hour` already provides much more granular information, `rush_hour` may not add significant additional predictive information.

---

## 5.6.4 Decision

```
rush_hour → ❌ Do not include as a final feature for now
```

Reason:

- `pickup_hour` already captures the underlying temporal information.
- `rush_hour` is highly derived from `pickup_hour`.
- The observed mean/median difference between rush and non-rush periods is relatively small.
- Adding the feature would increase feature complexity without strong evidence of additional value.

The feature can be reconsidered later if model experimentation shows that it provides measurable improvement.

---

## 5.6.5 Final Feature Decision

The datetime feature set remains:

```
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
```

`rush_hour` will **not** be added to the final candidate feature set at this stage.

### Implementation

**Python Script:** [5.6-Evaluate-Rush-Hour-Feature.py](../scripts/5.%20Feature-Engineering/5.6-Evaluate-Rush-Hour-Feature.py)

**Status: Step 5.6 Complete ✅**

---

## Step 5.7 – Check Feature Quality

## Objective

Perform a final quality check on the current candidate features before moving to preprocessing.

The checks cover:

- Feature availability in TRAIN and TEST
- Missing values
- Infinite values
- Expected value ranges
- Categorical value consistency
- Data types

---

## 5.7.1 Feature Consistency

All candidate features are present in both TRAIN and TEST.

```
Missing from TRAIN: []
Missing from TEST : []
```

The current candidate feature set is:

```
vendor_id
passenger_count
store_and_fwd_flag
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
```

**Decision: PASS**

---

## 5.7.2 Missing Values

No missing values were found in any candidate feature.

```
TRAIN → 0 missing values
TEST  → 0 missing values
```

**Decision: PASS**

No additional missing-value handling is required at the feature-engineering stage.

---

## 5.7.3 Infinite Values

No infinite values were found in the numeric features.

```
TRAIN → 0
TEST  → 0
```

**Decision: PASS**

---

## 5.7.4 Range Validation

The engineered features fall within their expected ranges.

| Feature | TRAIN | TEST | Expected |
|---|---|---|---|
| `pickup_hour` | 0–23 | 0–23 | 0–23 |
| `pickup_day_of_week` | 0–6 | 0–6 | 0–6 |
| `pickup_month` | 1–12 | 1–12 | 1–12 |
| `is_weekend` | 0, 1 | 0, 1 | 0/1 |
| `passenger_count` | 1–9 | 1–9 | > 0 |
| `distance_km` | 0–116.50 | 0–117.43 | ≥ 0 |

No invalid ranges were identified.

**Decision: PASS**

---

## 5.7.5 Categorical Value Consistency

TRAIN and TEST contain the same categorical values.

```
vendor_id: TRAIN → [1, 2]
TEST  → [1, 2]
store_and_fwd_flag: TRAIN → ['N', 'Y']
TEST  → ['N', 'Y']
```

This is important because the preprocessing pipeline must be able to handle the same categorical categories during training and inference.

**Decision: PASS**

---

## 5.7.6 Feature Data Types

Current data types are consistent between TRAIN and TEST:

```
vendor_id             → int64
passenger_count       → int64
store_and_fwd_flag    → object
pickup_hour            → int32
pickup_day_of_week     → int32
pickup_month           → int32
is_weekend             → int64
distance_km            → float64
```

These types are appropriate for the current feature-engineering stage.

Final categorical encoding and numerical preprocessing will be handled in **Step 6**.

---

## 5.7.7 Final Decision

All current candidate features passed the feature-quality checks.

```
Feature Quality
├── TRAIN/TEST consistency  ✅
├── Missing values          ✅
├── Infinite values         ✅
├── Value ranges            ✅
├── Category consistency    ✅
└── Data types              ✅
```

No feature-engineering correction is required at this stage.

### Current candidate feature set

```
Datetime
├── pickup_hour
├── pickup_day_of_week
├── pickup_month
└── is_weekend
Geographic
└── distance_km
Existing
├── vendor_id
├── passenger_count
└── store_and_fwd_flag
```

The feature set is ready for the final leakage review.

### Implementation

**Python Script:** [5.7-Check-Feature-Quality.py](../scripts/5.%20Feature-Engineering/5.7-Check-Feature-Quality.py)

**Status: Step 5.7 Complete ✅**

---

## Step 5.8 – Check Feature Leakage

## Objective

Perform a final leakage audit to verify that all candidate features are based only on information available at prediction time.

The audit checks:

- Candidate feature availability in TRAIN and TEST
- Target and post-trip column exclusion
- Feature-to-source mapping
- Prediction-time availability of feature sources

---

## 5.8.1 Candidate Feature Availability

All candidate features are available in both TRAIN and TEST.

```
Missing from TRAIN: []
Missing from TEST : []
```

The final candidate feature set is:

```
vendor_id
passenger_count
store_and_fwd_flag
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
```

**Decision: PASS**

---

## 5.8.2 Target and Post-Trip Columns

The target and post-trip columns are present only in TRAIN:

```
trip_duration: TRAIN → Present
TEST  → Absent

dropoff_datetime: TRAIN → Present
TEST  → Absent
```

Neither is included in the candidate feature set.

```
trip_duration    → Target
dropoff_datetime → Post-trip information
```

Therefore, neither can leak information into the model.

---

## 5.8.3 Feature Source Mapping

Each candidate feature was traced back to its original prediction-time source:

| Feature | Source |
|---|---|
| `vendor_id` | `vendor_id` |
| `passenger_count` | `passenger_count` |
| `store_and_fwd_flag` | `store_and_fwd_flag` |
| `pickup_hour` | `pickup_datetime` |
| `pickup_day_of_week` | `pickup_datetime` |
| `pickup_month` | `pickup_datetime` |
| `is_weekend` | `pickup_datetime` |
| `distance_km` | Pickup and dropoff coordinates |

All source columns are available when the prediction is made.

---

## 5.8.4 Prediction-Time Availability

The audit confirmed:

```
All feature sources are available at prediction time.
```

The feature dependency can therefore be represented as:

```
Prediction-Time Information
    ├── Datetime Features
    │   ├── pickup_hour
    │   ├── pickup_day_of_week
    │   ├── pickup_month
    │   └── is_weekend
    ├── Geographic Features
    │   └── distance_km
    └── Existing Features
        ├── vendor_id
        ├── passenger_count
        └── store_and_fwd_flag
```

---

## 5.8.5 Final Leakage Decision

The final audit produced:

```
PASS - No feature leakage identified.
```

Therefore, the current candidate feature set is considered **leakage-safe**.

### Final Candidate Features

```
Datetime
├── pickup_hour
├── pickup_day_of_week
├── pickup_month
└── is_weekend
Geographic
└── distance_km
Existing
├── vendor_id
├── passenger_count
└── store_and_fwd_flag
```

### Explicitly Excluded

```
id                 → Identifier
dropoff_datetime   → Post-trip information
trip_duration      → Target
rush_hour          → Not retained due to limited additional value
pickup_year        → Constant
pickup_day         → Not retained
```

**Step 5.8 is complete.**

### Implementation

**Python Script:** [5.8-Check-Feature-Leakage.py](../scripts/5.%20Feature-Engineering/5.8-Check-Feature-Leakage.py)

**Status: Step 5.8 Complete ✅**

---

## Step 5.9 – Feature Engineering Summary

## Objective

Consolidate the decisions from Step 5 and define the final feature set that will move forward to **Step 6 – Train / Validation Split & Preprocessing**.

---

## 5.9.1 Feature Engineering Process

The feature-engineering process followed this sequence:

```
Cleaned Data
    ↓
Identify Prediction-Time Information
    ↓
Datetime Feature Engineering
    ↓
Geographic Feature Engineering
    ↓
Evaluate Existing Features
    ↓
Evaluate Rush-Hour Feature
    ↓
Feature Quality Checks
    ↓
Leakage Audit
    ↓
Final Candidate Feature Set
```

---

## 5.9.2 Final Candidate Features

### Datetime Features

| Feature | Decision | Reason |
|---|---|---|
| `pickup_hour` | KEEP | Significant variation in trip duration across hours |
| `pickup_day_of_week` | KEEP | Trip duration varies across days |
| `pickup_month` | KEEP | Variation observed across the six-month dataset |
| `is_weekend` | KEEP | Weekday and weekend durations differ |

### Geographic Feature

| Feature | Decision | Reason |
|---|---|---|
| `distance_km` | KEEP | Clear increase in trip duration with increasing distance |

`distance_km` was calculated using the Haversine formula from pickup and dropoff coordinates.

### Existing Features

| Feature | Decision | Reason |
|---|---|---|
| `vendor_id` | KEEP | Meaningful difference in trip-duration statistics between vendors |
| `passenger_count` | KEEP | Duration generally increases across commonly represented passenger counts |
| `store_and_fwd_flag` | KEEP | Different duration statistics between `N` and `Y` |

---

## 5.9.3 Excluded Features

| Feature | Decision | Reason |
|---|---|---|
| `id` | EXCLUDE | Identifier, not a meaningful model feature |
| `dropoff_datetime` | EXCLUDE | Post-trip information and unavailable at prediction time |
| `trip_duration` | EXCLUDE | Target variable |
| `pickup_year` | EXCLUDE | Constant value (`2016`) |
| `pickup_day` | EXCLUDE | Not retained to avoid unnecessary feature complexity |
| `rush_hour` | EXCLUDE | Largely redundant with `pickup_hour` and showed limited additional signal |

---

## 5.9.4 Final Feature Set

The final candidate feature set contains **8 features**:

```
vendor_id
passenger_count
store_and_fwd_flag
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
```

---

## 5.9.5 Feature Quality

The final candidate features passed the quality checks:

```
TRAIN/TEST feature consistency → PASS
Missing values                → 0
Infinite values               → 0
Expected ranges               → PASS
Categorical consistency       → PASS
Data types                    → Consistent
```

---

## 5.9.6 Leakage Check

A final leakage audit confirmed:

```
All feature sources are available at prediction time.
No target or post-trip columns are included.
```

Final status:

```
PASS - No feature leakage identified.
```

---

## 5.9.7 Final Feature Engineering Outcome

```
Cleaned Data
    ↓
Prediction-Time Inputs
    ↓
┌──────────┴──────────┐
│                     │
▼                     ▼
Datetime Features     Geographic Feature
│ pickup_hour         │    distance_km
│ pickup_day_of_week  │
│ pickup_month        │
│ is_weekend          │
└─────────────────────┘
           │
           ▼
    Existing Features
    ├── vendor_id
    ├── passenger_count
    └── store_and_fwd_flag
           │
           ▼
      8 Candidate Features
           │
           ▼
        Leakage Audit
           │
           ▼
            PASS
```

---

## 5.9.8 Step 5 Completion

**Step 5 – Feature Engineering is complete.**

The project now has a defined, validated, and leakage-safe candidate feature set ready for the next stage:

```
Step 5 – Feature Engineering        ✅ COMPLETE
Step 6 – Train / Validation Split   & Preprocessing   → NEXT
```

The next phase will determine how these features are split, encoded, scaled where appropriate, and transformed into a model-ready representation.

### Implementation

**Python Script:** [5.9-Summary-Feature-Engineering.py](../scripts/5.%20Feature-Engineering/5.9-Summary-Feature-Engineering.py)

**Status: Step 5.9 Complete ✅**

---

# Step 6 – Train / Validation Split & Preprocessing

## Step 6.1 – Define Train/Validation Split Strategy

## Objective

Define an appropriate strategy for splitting the cleaned TRAIN dataset into training and validation sets.

The key consideration is that this is a **time-dependent prediction problem**. In a realistic deployment scenario, the model would be trained on historical trips and then used to predict future trips.

---

## 6.1.1 Dataset Temporal Coverage

The cleaned TRAIN dataset covers:

```
Start: 2016-01-01 00:00:17
End  : 2016-06-30 23:59:39
```

The dataset contains all **182 calendar days** in this period.

```
Expected calendar days: 182
Observed calendar days: 182
Missing calendar days : 0
```

Therefore, the dataset has continuous temporal coverage.

---

## 6.1.2 Monthly Distribution

The number of records per month is:

| Month | Records |
|---|---:|
| January | 229,686 |
| February | 238,276 |
| March | 256,172 |
| April | 251,630 |
| May | 248,468 |
| June | 234,310 |

The dataset has substantial data throughout the six-month period, without an obvious missing month.

---

## 6.1.3 Split Strategy Decision

Two common approaches were considered:

### Random Split

```
All data
    ↓
Randomly divide
    ├── Training
    └── Validation
```

This mixes observations from different points in time between the two datasets.

### Time-Based Split

```
Earlier trips
    ↓
Training

Later trips
    ↓
Validation
```

This better represents the real-world prediction scenario:

> **Train on historical data → predict future trips.**

For this project, a **chronological/time-based split** will therefore be used.

---

## 6.1.4 Why Time-Based Split?

The target is `trip_duration`, and predictions are based on information available at pickup time.

In production, we would not train a model using future trips and then evaluate it on historical trips.

A chronological split provides a more realistic evaluation:

```
Past
│
├─────────────── Training ───────────────┤
│
│
└────────────────── Future ───────────────┤
                                          │
                                      Validation
```

This also helps evaluate how well the model generalizes to a later period.

---

## 6.1.5 Decision

```
Train/Validation Strategy
        │
        ▼
Chronological Split
        │
        ├── Earlier period → Training
        │
        └── Later period   → Validation
```

A specific date cutoff and validation proportion will be defined during **Step 6.2 – Implement Train/Validation Split**.

**Decision: Time-based split selected.**

### Implementation

**Python Script:** [6.1-Analyze-Train-Validation-Split-Strategy.py](../scripts/6.%20Train-Validation-Split/6.1-Analyze-Train-Validation-Split-Strategy.py)

**Status: Step 6.1 Complete ✅**

---

## Step 6.2 – Implement Train/Validation Split

### Objective

Implement the chronological train/validation split defined in Step 6.1.

The cleaned TRAIN dataset was sorted by `pickup_datetime` and divided into:

```
80% → Training
20% → Validation
```

The split is chronological rather than random so that validation represents a later period than the training data.

---

### 6.2.1 Split Results

The cleaned dataset contains:

```
Total records     : 1,458,542
Training records  : 1,166,833
Validation records:   291,709
```

This corresponds to:

```
Training   → 80%
Validation → 20%
```

---

### 6.2.2 Temporal Separation

The training period is:

```
2016-01-01 00:00:17        ↓
2016-05-24 01:55:00
```

The validation period is:

```
2016-05-24 01:55:48        ↓
2016-06-30 23:59:39
```

The chronological check returned:

```
Training maximum < Validation minimum: True
```

Therefore, validation contains only observations occurring after the training data.

**Decision: PASS**

---

### 6.2.3 Feature and Target Separation

The final 8 features from Step 5.9 were used:

```
vendor_id
passenger_count
store_and_fwd_flag
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
```

The target was kept separately as:

```
trip_duration
```

Resulting shapes:

```
X_train: (1,166,833, 8)
y_train: (1,166,833,)
X_val  : (291,709, 8)
y_val  : (291,709,)
```

The feature count matches the final feature-engineering output.

---

### 6.2.4 Target Distribution

| Dataset | Mean (sec) | Median (sec) | Std (sec) |
|---|---:|---:|---:|
| Training | 939.62 | 563.0 | 148.25 |
| Validation | 1,005.28 | 691.0 | 257.71 |

Validation has somewhat higher trip-duration statistics than training.

This is acceptable and is an expected possibility when using a chronological split. We do not modify the validation set to force its distribution to match training.

The large standard deviation reflects the highly skewed nature of `trip_duration`, including the long trips intentionally retained during Phase 3B.

---

### 6.2.5 Output Files

The split datasets were successfully generated:

```
data/split/
├── X_train.csv
├── X_val.csv
├── y_train.csv
└── y_val.csv
```

These files will be used as the inputs for the preprocessing stage.

---

### 6.2.6 Final Decision

```
Chronological split       → PASS
80/20 split               → PASS
Temporal separation       → PASS
Feature/target separation → PASS
Feature count             → 8
Output files              → Created
```

**Step 6.2 is complete.**

The project now has a clean separation between training and validation data, and preprocessing can safely be fitted **only on `X_train`**.

### Implementation

**Python Script:** [6.2-Implement-Train-Validation-Split.py](../scripts/6.%20Train-Validation-Split/6.2-Implement-Train-Validation-Split.py)

**Status: Step 6.2 Complete ✅**

---

## Step 6.3 – Feature Scaling & Normalization

## Objective

Identify which candidate features require numerical scaling before Machine Learning preprocessing.

The analysis showed that the current feature set contains categorical, discrete/temporal, binary, and continuous numerical features. Therefore, scaling is not applied blindly to every numerical-looking column.

---

## 6.3.1 Feature Classification

| Feature | Type | Scaling |
|---|---|---|
| `vendor_id` | Categorical | No |
| `passenger_count` | Discrete count | No |
| `store_and_fwd_flag` | Categorical | No |
| `pickup_hour` | Temporal/discrete | No |
| `pickup_day_of_week` | Temporal/discrete | No |
| `pickup_month` | Temporal/discrete | No |
| `is_weekend` | Binary | No |
| `distance_km` | Continuous numerical | Yes |

`vendor_id` is stored as an integer but represents a category rather than a continuous quantity. It will therefore be handled during categorical encoding.

---

## 6.3.2 Scaling Candidate

`distance_km` is the primary scaling candidate.

Its training-set range is:

```
Minimum: 0.0 km
Maximum: 116.496 km
Mean:    3.414 km
Std:     3.923 km
```

Compared with the other numerical features, it has a substantially larger continuous range.

---

## 6.3.3 Scaling Strategy

`StandardScaler` will be used for the continuous numerical feature.

The scaler must be fitted **only on the training data**:

```text
                    X_train
                       │
                       ▼
              ┌────────────────────┐
              │ Fit StandardScaler │
              │ using X_train only │
              └────────┬───────────┘
                       │
                 Fitted Scaler
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Transform X_train   Transform X_val
              │                 │
              ▼                 ▼
       X_train_scaled       X_val_scaled
 ```

Validation data will only be transformed using the scaler learned from training data.

This prevents validation information from influencing the preprocessing parameters.

---

## 6.3.4 Why Not Scale the Other Features?

The remaining numerical-looking features represent discrete, temporal, binary, or categorical information rather than continuous measurements.

For example:

```
pickup_hour        → 0–23
pickup_day_of_week → 0–6
pickup_month       → 1–12
is_weekend         → 0/1
```

These values have meaningful discrete interpretations, so they will not be blindly standardized.

`passenger_count` is also retained in its original form as a discrete count. Its treatment can be revisited if required by the model family selected in Phase 4.

---

## 6.3.5 Decision

```
Continuous numerical
└── distance_km → StandardScaler

Categorical
├── vendor_id
└── store_and_fwd_flag
    ↓
    Step 6.4

Discrete / temporal / binary
├── passenger_count
├── pickup_hour
├── pickup_day_of_week
├── pickup_month
└── is_weekend
    ↓
    No scaling
```

**Step 6.3 analysis is complete.**

The actual scaler implementation will be incorporated into the preprocessing pipeline, with the scaler fitted **only on training data**.

### Implementation

**Python Script:** [6.3-Analyze-Feature-Scaling.py](../scripts/6.%20Train-Validation-Split/6.3-Analyze-Feature-Scaling.py)

**Status: Step 6.3 Complete ✅**

---

## Step 6.4 – Encoding Categorical Features

## Objective

Convert categorical features into a numerical representation suitable for Machine Learning while ensuring that validation data does not influence the encoding process.

---

## 6.4.1 Categorical Features

The following features were identified as categorical:

```
vendor_id
store_and_fwd_flag
```

Their observed categories are:

```
vendor_id          → 1, 2
store_and_fwd_flag → N, Y
```

Although `vendor_id` is stored as `int64`, it represents a categorical value rather than a continuous numerical quantity.

---

## 6.4.2 Category Consistency

Both categorical features contain the same categories in TRAIN and VALIDATION.

| Feature | TRAIN | VALIDATION | Status |
|---|---|---|---|
| `vendor_id` | 1, 2 | 1, 2 | PASS |
| `store_and_fwd_flag` | N, Y | N, Y | PASS |

No unseen validation categories were found.

---

## 6.4.3 Missing Values

No missing values were found:

```
TRAIN:
vendor_id             0
store_and_fwd_flag    0

VALIDATION:
vendor_id             0
store_and_fwd_flag    0
```

---

## 6.4.4 Encoding Strategy

**One-Hot Encoding** was selected for both categorical features.

Conceptually:

```
vendor_id
    │
    ▼
One-Hot Encoding
    │
    ├── vendor_id_1
    └── vendor_id_2
```

and:

```
store_and_fwd_flag
    │
    ▼
One-Hot Encoding
    │
    ├── store_and_fwd_flag_N
    └── store_and_fwd_flag_Y
```

This avoids introducing an artificial numerical ordering between categories.

---

## 6.4.5 Unknown Category Handling

The encoder will use:

```
handle_unknown="ignore"
```

This ensures that if a previously unseen category appears during future inference, the preprocessing pipeline does not fail.

---

## 6.4.6 Leakage Prevention

The encoder will be **fitted only on training data**:

```
X_train categorical features
    │
    ▼
Fit Encoder
    │
    ├── Transform X_train
    └── Transform X_val
```

Validation data is only transformed using the encoder learned from TRAIN.

---

## 6.4.7 Decision

```
Categorical features
    │
    ├── vendor_id
    └── store_and_fwd_flag
            │
            ▼
        OneHotEncoder
            │
            ├── handle_unknown="ignore"
            └── fit on TRAIN only
```

**Step 6.4 is complete.**

The categorical encoding strategy is defined and ready to be incorporated into the final preprocessing pipeline.

### Implementation

**Python Script:** [6.4-Analyze-Categorical-Features.py](../scripts/6.%20Train-Validation-Split/6.4-Analyze-Categorical-Features.py)

**Status: Step 6.4 Complete ✅**

---

## Step 6.5 – Verify Preprocessed Datasets

## Objective

Verify that the preprocessing strategy defined in Steps 6.3 and 6.4 produces valid, complete, and compatible ML-ready feature datasets.

The preprocessing stage must preserve all features approved during feature engineering while applying the appropriate transformation to each feature type.

The final preprocessing strategy is:

```text
Input Features
│
├── Continuous Numerical
│   └── distance_km
│       ↓
│   StandardScaler
│
├── Categorical
│   ├── vendor_id
│   └── store_and_fwd_flag
│       ↓
│   OneHotEncoder(handle_unknown="ignore")
│
└── Discrete / Temporal / Binary
    ├── passenger_count
    ├── pickup_hour
    ├── pickup_day_of_week
    ├── pickup_month
    └── is_weekend
        ↓
    Passthrough
```

The preprocessor was fitted **only on the training dataset** and then used to transform both TRAIN and VALIDATION.

---

### 6.5.1 — Preprocessing Strategy

The final feature treatment is:

| Feature | Type | Preprocessing |
|---|---|---|
| distance_km | Continuous numerical | StandardScaler |
| vendor_id | Categorical | OneHotEncoder |
| store_and_fwd_flag | Categorical | OneHotEncoder |
| passenger_count | Discrete count | Passthrough |
| pickup_hour | Temporal/discrete | Passthrough |
| pickup_day_of_week | Temporal/discrete | Passthrough |
| pickup_month | Temporal/discrete | Passthrough |
| is_weekend | Binary | Passthrough |

This preserves all 8 features finalized during Step 5.9.

No feature approved during feature engineering is silently removed during preprocessing.

**Status: 6.5.1 – COMPLETED ✅**

---

### 6.5.2 — Leakage-Safe Preprocessing

The preprocessing pipeline follows the required anti-leakage sequence:

```text
X_train
   │
   ▼
Fit Preprocessor
   │
   ├──► Transform X_train
   │
   └──► Transform X_val
```

The scaler and categorical encoder are fitted only on `X_train`.

Validation data is never used to calculate scaling statistics or determine categorical encoding state.

This ensures that validation information does not influence the preprocessing parameters used during model training.

**Status: 6.5.2 – COMPLETED ✅**

---

### 6.5.3 — Processed Dataset Shapes

The original train/validation feature datasets contain:
```text
X_train: (1,166,833, 8)
X_val  : (291,709, 8)
```

After preprocessing, the number of rows remains unchanged.

The feature count increases because categorical features are one-hot encoded while the remaining approved features are preserved using passthrough:
```text
X_train: (1,166,833, 10)
X_val  : (291,709, 10)
```

**The transformation is:**

```text
8 original features
        │
        ├── distance_km
        │      ↓
        │  StandardScaler
        │      ↓
        │  1 feature
        │
        ├── vendor_id
        │      ↓
        │  OneHotEncoder
        │      ↓
        │  2 features
        │
        ├── store_and_fwd_flag
        │      ↓
        │  OneHotEncoder
        │      ↓
        │  2 features
        │
        └── 5 remaining approved features
               ↓
           passthrough
               ↓
           5 features

Total = 1 + 2 + 2 + 5 = 10 ML-ready features
```

**Status: 6.5.3 – COMPLETED ✅**

---

### 6.5.4 — Final Processed Features

The final processed feature representation contains **10 ML-ready features**:

```text
numerical__distance_km

passthrough_numerical__passenger_count
passthrough_numerical__pickup_hour
passthrough_numerical__pickup_day_of_week
passthrough_numerical__pickup_month
passthrough_numerical__is_weekend

categorical__vendor_id_1
categorical__vendor_id_2

categorical__store_and_fwd_flag_N
categorical__store_and_fwd_flag_Y
```

The exact feature names generated by the fitted `ColumnTransformer` should be treated as the authoritative preprocessing output.

The persisted preprocessing artifact and the processed datasets must use the same feature ordering and feature-name contract.

**Status: 6.5.4 – COMPLETED ✅**

---

### 6.5.5 — Scaling Verification

`distance_km` is standardized using `StandardScaler`.

The scaler is fitted only on the training data.

Expected training-set statistics after scaling:
```text
Mean ≈ 0.0
Std  ≈ 1.0
```

The validation dataset is transformed using the scaler learned from the training dataset.

Validation data must not be used to fit the scaler.

**Status: 6.5.5 – COMPLETED ✅**

---

### 6.5.6 — Categorical Encoding Verification

The categorical features are:
```text
vendor_id
store_and_fwd_flag
```

They are converted using `OneHotEncoder` with:

handle_unknown="ignore"


**Expected representation:**

```text
vendor_id
    ↓
One-Hot Encoding
    ├── vendor_id_1
    └── vendor_id_2

store_and_fwd_flag
    ↓
One-Hot Encoding
    ├── store_and_fwd_flag_N
    └── store_and_fwd_flag_Y
```

The encoder is fitted only on `X_train`.

Validation data is transformed using the encoder learned from training data.

If an unseen category is encountered during future inference, `handle_unknown="ignore"` prevents the preprocessing pipeline from failing.

**Status: 6.5.6 – COMPLETED ✅**

---

### 6.5.7 — Passthrough Feature Verification

The following approved features do not require scaling or categorical encoding in the current preprocessing design:
```text
passenger_count
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
```

These features must therefore be preserved through the preprocessing stage using passthrough.

They must not be silently dropped from the ML-ready dataset.

**Expected representation:**
```text
passenger_count
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
```

Their values must remain unchanged between the feature-engineered input and the corresponding processed representation, apart from any dataframe/array type conversion introduced by the preprocessing implementation.

**Status: 6.5.7 – COMPLETED ✅**

---

### 6.5.8 — Data Quality Verification

The processed TRAIN and VALIDATION datasets must be checked for:
- Missing values
- Infinite values
- Unexpected feature count
- Unexpected feature names
- Unexpected feature ordering

**Expected result:**

```text
Missing values:
TRAIN      → 0
VALIDATION → 0

Infinite values:
TRAIN      → 0
VALIDATION → 0

Feature count:
TRAIN      → 10
VALIDATION → 10

Feature columns:
TRAIN      → Same representation
VALIDATION → Same representation
```

**Status: 6.5.8 – COMPLETED ✅**

---

### 6.5.9 — Train / Validation Compatibility

The processed TRAIN and VALIDATION datasets must have:
```text
Same feature columns → True
Same feature count   → True
Same feature order   → True
```

**Expected shape:**
```text
X_train_processed → 1,166,833 × 10
X_val_processed   →   291,709 × 10
```

This ensures that both datasets provide a compatible representation for Machine Learning model training and evaluation.

**Status: 6.5.9 – COMPLETED ✅**

---

### 6.5.10 — Leakage Verification

The complete preprocessing process must follow:
```text
X_train
    │
    ▼
Fit StandardScaler
Fit OneHotEncoder
    │
    ├───────────────┐
    ▼               ▼
Transform X_train  Transform X_val
```

**Verification requirements:**
```text
Scaler fitted on TRAIN only              → PASS
Encoder fitted on TRAIN only             → PASS
Validation transformed using TRAIN state → PASS
No validation information used for fit   → PASS
```

**Status: 6.5.10 – COMPLETED ✅**

---

### 6.5.11 — Final Verification

The final preprocessing verification must confirm:

```text
All 8 approved input features preserved through preprocessing → PASS
10 ML-ready output features produced                          → PASS
Scaling of distance_km             → PASS
Categorical encoding               → PASS
Passthrough features preserved     → PASS
Missing values                     → PASS
Infinite values                    → PASS
Train/validation matching          → PASS
Feature ordering                   → PASS
Preprocessing leakage              → PASS
```

**Final status:**
```text
PASS - Preprocessed datasets are valid, complete, leakage-safe,
and compatible for Machine Learning.
```

**Status: 6.5.11 – COMPLETED ✅**

---

### 6.5.12 — Step 6 Completion

Step 6 – Train / Validation Split & Preprocessing is now complete.

The project has successfully established:

```text
Cleaned Data
    ↓
Feature Engineering
    ↓
Chronological Train/Validation Split
    ↓
Feature Classification
    ↓
Scaling Strategy
    ↓
Categorical Encoding
    ↓
Passthrough of Approved Discrete/Temporal/Binary Features
    ↓
Leakage-Safe Preprocessing
    ↓
10 ML-Ready Features
```

The preprocessing stage now preserves the complete feature set defined during feature engineering while applying the appropriate transformation to each feature type.

The next major task is to ensure that the production preprocessing implementation under `src/` exactly matches this documented preprocessing contract.

**Implementation**

Python Script: `6.5-Verify-Preprocessed-Datasets.py`

**Status: Step 6.5 – Complete ✅**

---

# Step 7 – Build the Data Engineering Code Pipeline

**Goal:** Move the data-engineering logic from exploratory scripts into reusable production-style Python modules.

---

## Step 7.1 – Design Modular Pipeline Architecture

### Objective

Define how the finalized Phase 3 data engineering logic will be organized into the existing `src/` structure.

The goal is to separate **learning/analysis scripts** from **reusable production logic** without unnecessarily restructuring the project.

The architecture is designed so that the finalized components can later be connected into an end-to-end pipeline in Step 7.2.

---

### Project Philosophy

The project follows a clear separation of responsibilities:

```text
scripts/    → Learning + Exploration + Historical Evidence
src/        → Reusable Production Logic
tests/      → Automated Verification
pipelines/  → Connect the reusable components
DVC         → Reproduce the complete workflow
```

Existing scripts under `scripts/` will not be moved, deleted, or replaced.

Instead, finalized logic from the relevant scripts will be recreated/refactored as clean, reusable production modules under `src/`.

This preserves the original scripts as learning artifacts, historical evidence, and implementation references, while `src/` becomes the reusable production layer.

### Proposed Modular Architecture

```text
src/
├── ingestion/
│   └── cleaning.py
├── validation/
│   └── validate_contract.py
├── features/
│   └── feature_engineering.py
├── pipelines/
│   ├── train_validation_split.py
│   └── preprocessing.py
├── training/
├── inference/
├── monitoring/
├── api/
└── utils/
    └── logger.py
```

The `training/`, `inference/`, `monitoring/`, and `api/` directories are reserved for later phases and are not implemented as part of Step 7.1.

### Production Modules Created

The following reusable modules have now been created and individually verified.

| Production Module | Responsibility | Status |
|---|---|---|
| `src/utils/logger.py` | Centralized logging for production modules | ✅ Verified |
| `src/ingestion/cleaning.py` | Load and clean raw TRAIN/TEST data | ✅ Verified |
| `src/validation/validate_contract.py` | Validate datasets against the data contract | ✅ Verified |
| `src/features/feature_engineering.py` | Create finalized datetime and geographic features | ✅ Verified |
| `src/pipelines/train_validation_split.py` | Create chronological TRAIN/VALIDATION split | ✅ Verified |
| `src/pipelines/preprocessing.py` | Scale and encode ML features | ✅ Verified |

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/utils/logger.py` | Provide centralized logging for reusable production modules |
| `src/ingestion/cleaning.py` | Load raw data and perform finalized data-cleaning operations |
| `src/validation/validate_contract.py` | Validate datasets against the project data contract |
| `src/features/feature_engineering.py` | Create finalized datetime and geographic features |
| `src/pipelines/train_validation_split.py` | Create a chronological 80/20 TRAIN/VALIDATION split |
| `src/pipelines/preprocessing.py` | Fit preprocessing on TRAIN and transform TRAIN/VALIDATION |
| `tests/` | Automatically verify reusable production modules |
| `DVC` | Track and reproduce the complete data pipeline |

### Production Logging

All reusable modules under `src/` use the centralized logger instead of `print()` statements.

```text
src/
├── utils/
│   └── logger.py
├── ingestion/
│   └── cleaning.py
├── validation/
│   └── validate_contract.py
├── features/
│   └── feature_engineering.py
└── pipelines/
    ├── train_validation_split.py
    └── preprocessing.py
```

The common pattern is:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

This provides consistent runtime visibility and allows failures to be logged with useful context and traceback information.

The exploratory scripts under `scripts/` may continue to use `print()` because they serve a different purpose.

### Data Engineering Flow

The reusable components form the following logical flow:

```text
Raw Data
    ↓
Data Contract Validation
    ↓
Data Cleaning
    ↓
Feature Engineering
    ├── Datetime Features
    │   ├── pickup_hour
    │   ├── pickup_day_of_week
    │   ├── pickup_month
    │   └── is_weekend
    └── Geographic Feature
        └── distance_km
    ↓
Train / Validation Split
    └── Chronological 80/20 Split
    ↓
Preprocessing
    ├── StandardScaler
    │   └── distance_km
    └── OneHotEncoder
        ├── vendor_id
        └── store_and_fwd_flag
    ↓
ML-Ready Dataset
```

### Important Preprocessing Rule

Preprocessing follows the anti-leakage rule established during Step 6:

```text
TRAIN → Fit Preprocessor → Transform TRAIN → Transform VALIDATION
```

The preprocessor is never fitted using the complete TRAIN + VALIDATION dataset.

The finalized preprocessing configuration is:

```text
Numerical:
    distance_km → StandardScaler

Categorical:
    vendor_id
    store_and_fwd_flag → OneHotEncoder(handle_unknown="ignore")
```

The resulting processed feature set contains:

```text
numerical__distance_km
categorical__vendor_id_1
categorical__vendor_id_2
categorical__store_and_fwd_flag_N
categorical__store_and_fwd_flag_Y
```

### Chronological Train / Validation Strategy

The train/validation split is chronological rather than random.

The finalized implementation uses:

```text
80% → TRAIN
20% → VALIDATION
```

The current dataset produces:

```text
TRAIN:      1,166,833 records
VALIDATION:   291,709 records
```

The chronological boundary was verified:

```text
TRAIN:      2016-01-01 00:00:17 → 2016-05-24 01:55:00
VALIDATION: 2016-05-24 01:55:48 → 2016-06-30 23:59:39
```

Therefore:

```text
TRAIN maximum < VALIDATION minimum
```

This preserves the intended future-prediction simulation.

### Relationship Between `scripts/` and `src/`

The existing scripts document how decisions were explored, implemented, analyzed, and validated.

The production modules are recreated from the finalized logic in those scripts.

#### Data Cleaning

```text
scripts/clean.py
        ↓ finalized cleaning logic
src/ingestion/cleaning.py
```

#### Data Contract Validation

```text
scripts/validation/validate_contract.py
        ↓ finalized validation logic
src/validation/validate_contract.py
```

#### Feature Engineering

```text
scripts/5. Feature-Engineering/5.3a-Create-Datetime-Features.py
scripts/5. Feature-Engineering/5.4b-Create-Haversine-Distance.py
        ↓ finalized feature logic
src/features/feature_engineering.py
```

#### Train / Validation Split

```text
scripts/6. Train-Validation-Split/6.2-Implement-Train-Validation-Split.py
        ↓ finalized split logic
src/pipelines/train_validation_split.py
```

#### Preprocessing

```text
scripts/6. Train-Validation-Split/6.3-Analyze-Feature-Scaling.py
scripts/6. Train-Validation-Split/6.4-Analyze-Categorical-Features.py
scripts/6. Train-Validation-Split/6.5-Verify-Preprocessed-Datasets.py
        ↓ finalized preprocessing decisions
src/pipelines/preprocessing.py
```

The original scripts remain available as historical evidence, learning artifacts, and references.

They are not moved into `src/`.

### Reusability Principle

The production modules are designed as reusable components rather than one large script.

For example:

```text
feature_engineering.py
        ↓
engineer_features()
        ↓
returns engineered DataFrame
```

and:

```text
train_validation_split.py
        ↓
split_train_validation()
        ↓
returns X_train, X_val, y_train, y_val
```

and:

```text
preprocessing.py
        ↓
preprocess_train_validation()
        ↓
returns processed TRAIN, processed VALIDATION, and fitted preprocessor
```

This allows Step 7.2 to connect these components without duplicating their internal logic.

### Current Production Component Flow

The components created so far can be viewed as:

```text
Data Contract Validation
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train / Validation Split
        ↓
Preprocessing
        ↓
ML-Ready Data
```

Step 7.2 will connect these reusable components into a single executable end-to-end workflow.

### Design Principles

- Do not duplicate production logic unnecessarily.
- Keep exploration and analysis scripts under `scripts/`.
- Keep reusable logic under `src/`.
- Keep automated verification under `tests/`.
- Keep pipeline orchestration under `src/pipelines/`.
- Use centralized logging instead of `print()` in production modules.
- Do not introduce unnecessary modules or abstractions.
- Reuse finalized decisions from Phase 3 rather than rebuilding exploratory analysis.
- Fit preprocessing only on TRAIN data.
- Keep the original scripts unchanged as historical evidence and learning artifacts.
- Build production modules by refactoring finalized existing logic rather than blindly copying exploratory scripts.

### Verification Status

Each production component created during Step 7.1 was executed and verified independently.

| Module | Verification | Status |
|---|---|---|
| `src/utils/logger.py` | Logging verified | ✅ |
| `src/ingestion/cleaning.py` | Cleaning output verified | ✅ |
| `src/validation/validate_contract.py` | Contract validation verified | ✅ |
| `src/features/feature_engineering.py` | Datetime + Haversine features verified | ✅ |
| `src/pipelines/train_validation_split.py` | Chronological 80/20 split verified | ✅ |
| `src/pipelines/preprocessing.py` | Scaling + encoding + leakage checks | ✅ |

The reusable components therefore have been individually verified before being connected into the end-to-end pipeline.

### Expected Outcome

At the end of Step 7.1, the project has a clear separation:

```text
scripts/        Learning + Exploration + Historical Evidence
        ↓
src/            Reusable Production Logic
        ↓
tests/          Automated Verification
        ↓
src/pipelines/  End-to-End Execution
        ↓
DVC             Reproducibility
```

The project is now ready for Step 7.2 – Implement End-to-End Pipeline, where these independently verified components will be connected into a reproducible workflow.

**Status: Step 7.1 Complete ✅**

---

## Step 7.2 – Implement End-to-End Pipeline

### Objective

Implement a single end-to-end orchestration pipeline that connects the reusable production modules created in Step 7.1.

The objective is to ensure that the complete Phase 3 data-engineering workflow can be executed through one entry point without duplicating the business logic already implemented in `src/`.

The pipeline is implemented in:

```text
src/
└── pipelines/
    └── run_pipeline.py
```

The pipeline uses the reusable modules created during Step 7.1 rather than directly executing the historical scripts under `scripts/`.

### Production Pipeline Flow

The final pipeline follows this sequence:

```text
Raw TRAIN / TEST
       │
       ▼
Pre-Clean Contract Validation
       │
       │ Diagnostic
       ▼
Data Cleaning
       │
       ▼
Post-Clean Contract Validation
       │
       │ Mandatory Gate
       ├── FAIL → Stop Pipeline
       └── PASS
             │
             ▼
      Feature Engineering
        TRAIN only
             │
             ▼
   Chronological Train /
      Validation Split
             │
             ▼
   TRAIN-only Preprocessing
             │
             ▼
   Preprocessed Dataset
       Verification
             │
             ▼
       ML-Ready Data
```

### Pipeline Entry Point

The complete workflow is executed through:

```text
src/pipelines/run_pipeline.py
```

Run from the project root using:

```bash
python -m src.pipelines.run_pipeline
```

Using module execution ensures that the `src` package is resolved correctly and allows the production modules to import each other consistently.

### Stage 1 – Pre-Clean Data Contract Validation

The pipeline first validates the raw TRAIN and TEST datasets against the project data contract.

This validation is diagnostic rather than blocking. The raw datasets are allowed to contain known data-quality violations because these violations are expected to be handled by the cleaning stage.

For the current dataset, the pre-clean validation identified:

```text
TRAIN: passenger_count > 0 → 60 invalid records
TEST:  passenger_count > 0 → 23 invalid records
```

These violations were logged but did not stop the pipeline.

The pipeline therefore follows:

```text
Raw Data
    ↓
Contract Validation
    ↓
Known raw-data violations
    ↓
Continue to Cleaning
```

This prevents the contract validation stage from blocking the cleaning process that is specifically responsible for handling these raw-data quality issues.

### Stage 2 – Data Cleaning

The reusable cleaning module, `src/ingestion/cleaning.py`, is executed through its reusable functions.

The pipeline receives:

```text
train_clean
test_clean
train_audit
test_audit
```

The cleaning stage also produces a detailed audit describing why records were removed.

#### TRAIN Cleaning Audit

```text
Original Records       : 1,458,644
Invalid Passenger      : 60
Invalid Coordinates    : 38
Extreme Outliers       : 4
Total Removed          : 102
Final Records          : 1,458,542
```

The removal count is fully reconciled:

```text
60 + 38 + 4 = 102
```

#### TEST Cleaning Audit

```text
Original Records       : 625,134
Invalid Passenger      : 23
Invalid Coordinates    : 19
Total Removed          : 42
Final Records          : 625,092
```

The removal count is fully reconciled:

```text
23 + 19 = 42
```

The cleaning audit is logged through the centralized project logger, providing traceability without using `print()` statements.

### Stage 3 – Post-Clean Data Contract Validation

After cleaning, the pipeline performs a second data-contract validation.

Unlike the pre-clean validation, this validation is a mandatory quality gate.

```text
Cleaning
    ↓
Post-Clean Contract Validation
    ↓
PASS → Continue
FAIL → Stop Pipeline
```

The cleaned TRAIN and TEST datasets were successfully validated.

Final result:

```text
RESULT: PASS
All implemented data-contract rules passed.
```

This ensures that downstream feature engineering and ML preparation only operate on datasets satisfying the implemented data contract.

### Stage 4 – Feature Engineering

The reusable feature-engineering module, `src/features/feature_engineering.py`, is used to create the finalized Phase 3C features.

The production feature set includes:

```text
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
```

The features are derived only from prediction-time information.

#### Training Pipeline Decision

During implementation, we identified that the TEST engineered DataFrame was not required for the training/validation pipeline.

Therefore, the final training pipeline performs feature engineering on TRAIN only:

```text
Clean TRAIN
    ↓
Feature Engineering
    ↓
Train / Validation Split
```

TEST is still cleaned and contract validated, but is not unnecessarily feature-engineered during the training pipeline.

The same reusable feature-engineering logic will be used later when TEST/inference data is required for prediction.

This avoids unnecessary computation while preserving the reusable feature-engineering module.

### Stage 5 – Chronological Train / Validation Split

The reusable module, `src/pipelines/train_validation_split.py`, creates the TRAIN/VALIDATION datasets.

Because the project is time-dependent, the split is chronological rather than random.

Final split:

```text
Total records      : 1,458,542
Training records   : 1,166,833
Validation records :   291,709
```

Equivalent proportions:

```text
Training   : 80%
Validation : 20%
```

#### Temporal Ranges

Training:

```text
2016-01-01 00:00:17
        ↓
2016-05-24 01:55:00
```

Validation:

```text
2016-05-24 01:55:48
        ↓
2016-06-30 23:59:39
```

The training period ends before the validation period begins.

This preserves the temporal nature of the problem and prevents future observations from being used to construct the historical training set.

### Stage 6 – Preprocessing

The reusable preprocessing module, `src/pipelines/preprocessing.py`, performs the finalized preprocessing strategy.

#### Numerical Feature

`distance_km` is standardized using `StandardScaler`.

#### Categorical Features

`vendor_id` and `store_and_fwd_flag` are encoded using `OneHotEncoder`.

Unknown categories are handled using:

```text
handle_unknown = "ignore"
```

#### TRAIN-Only Fitting

The preprocessing pipeline is fitted only on TRAIN:

```text
TRAIN
    ↓
Fit scaler
Fit encoder
    ↓
Transform TRAIN
    ↓
Transform VALIDATION
```

Validation data is never used to fit the preprocessing transformations.

This prevents preprocessing-related data leakage.

### Stage 7 – Preprocessed Dataset Verification

The preprocessing module verifies the resulting datasets before they are persisted.

Final shapes:

```text
Original X_train : (1,166,833, 8)
Original X_val   : (291,709, 8)

Processed X_train: (1,166,833, 5)
Processed X_val  : (291,709, 5)
```

The verification confirms:

- No missing values
- No infinite values
- Compatible TRAIN / VALIDATION feature columns
- Same feature count
- Numerical scaling completed
- Categorical encoding completed
- TRAIN-only preprocessing maintained

Final verification status:

```text
PASS - Preprocessed datasets are valid and compatible.
```

### Output Artifacts

The pipeline produces the following datasets:

#### Train / Validation Split

```text
data/split/
├── X_train.csv
├── X_val.csv
├── y_train.csv
└── y_val.csv
```

#### Processed Datasets

```text
data/processed/
├── X_train_processed.csv
└── X_val_processed.csv
```

These datasets are ready for consumption by the ML training stage in Phase 4.

### Centralized Logging

The end-to-end pipeline uses the project's centralized logger rather than `print()` statements.

Example:

```text
2026-08-15 ... - INFO - __main__ - PHASE 3 - END-TO-END DATA ENGINEERING PIPELINE STARTED
```

Individual reusable modules also report their progress through the same logging system.

This provides visibility into:

- Pipeline start
- Validation
- Cleaning
- Cleaning audit
- Feature engineering
- Train/validation split
- Preprocessing
- Verification
- Output persistence
- Pipeline completion/failure

If an unexpected exception occurs, the pipeline logs the exception and stops rather than silently continuing.

### Final End-to-End Verification

The final execution successfully completed the complete Phase 3 workflow.

The pipeline confirmed:

```text
Pre-clean validation
        ↓
Cleaning
        ↓
Post-clean validation → PASS
        ↓
Feature engineering → TRAIN only
        ↓
Chronological split → 80/20
        ↓
TRAIN-only preprocessing
        ↓
Processed dataset verification → PASS
        ↓
Output persistence
        ↓
Pipeline completed successfully
```

The final log confirms:

```text
PHASE 3 - END-TO-END DATA ENGINEERING PIPELINE COMPLETED
```

### Key Design Decisions

1. **Pre-clean validation is diagnostic.** Raw-data contract violations do not automatically stop the pipeline because known violations are handled by the cleaning stage.
2. **Post-clean validation is a mandatory gate.** If the cleaned datasets fail the implemented data contract, the pipeline stops before downstream ML preparation.
3. **TEST is not unnecessarily feature-engineered.** The training pipeline only feature-engineers TRAIN because TEST is not required for the train/validation split or preprocessing stage. The reusable feature-engineering module remains available for future inference/test processing.
4. **Reusable modules are orchestrated rather than duplicated.** `run_pipeline.py` connects the existing `src/` modules instead of recreating their internal logic.
5. **Logging is centralized.** The production pipeline uses the project's logger rather than `print()` statements.

### Step 7.2 Outcome

The Phase 3 data-engineering workflow is now executable through a single production-oriented entry point:

```bash
python -m src.pipelines.run_pipeline
```

The pipeline successfully connects:

```text
Validation
    ↓
Cleaning
    ↓
Validation Gate
    ↓
Feature Engineering
    ↓
Train / Validation Split
    ↓
Preprocessing
    ↓
Verification
    ↓
ML-Ready Data
```

The implementation is reproducible, observable, modular, and ready for automated testing in Step 7.3 – Pipeline Testing & Validation.

**Status: Step 7.2 Complete ✅**

---

## Step 7.3 – Pipeline Testing & Validation

### Objective

Establish automated tests that validate the reusable production modules, the pipeline's data and processing invariants, leakage controls, and the complete end-to-end workflow.

The tests will provide repeatable evidence that the Phase 3 data-engineering pipeline behaves correctly as it evolves.

### Step 7.3a – Setup Automated Testing with pytest

#### Objective

Establish `pytest` as the automated testing framework for the project's reusable production modules.

#### Implementation

Added `pytest` as a development/testing dependency:

```text
requirements.txt
    ↓
pytest
```

#### Freeze Project Dependencies

After installing the required testing dependency, regenerate the locked dependency file from the active virtual environment:

```powershell
python -m pip freeze > requirements-lock.txt
```

Verify that `pytest` was captured in the lock file:

```powershell
Select-String "pytest" requirements-lock.txt
```

The exact installed version is captured in `requirements-lock.txt` using the existing project dependency-freezing workflow.

#### Verification

Pytest was successfully installed and verified:

```text
Python 3.12.9
pytest 9.1.1
```

The test runner successfully started from the project virtual environment:

```bash
python -m pytest -v
```

At this stage:

```text
collected 0 items
no tests ran
```

This is expected because automated tests are created in the following steps.

Pytest import was also verified successfully.

#### Outcome

The automated testing environment is ready for testing the reusable production modules in `src/`.

**Step 7.3a → PASS ✅**

**Status: Step 7.3a Complete ✅**

### Step 7.3b – Test Production Modules

#### Objective

Verify that the reusable production modules created under `src/` can be imported successfully and expose their expected public functions.

#### Test File

```text
tests/
└── test_production_modules.py
```

#### Modules Tested

| Production Module | Status |
|---|---|
| `src/ingestion/cleaning.py` | PASS |
| `src/validation/validate_contract.py` | PASS |
| `src/features/feature_engineering.py` | PASS |
| `src/pipelines/train_validation_split.py` | PASS |
| `src/pipelines/preprocessing.py` | PASS |

#### Test Execution

```bash
python -m pytest tests/test_production_modules.py -v
```

#### Result

```text
collected 5 items
5 passed in 10.68s
```

All five production modules successfully imported and exposed their expected public functions.

#### What This Test Covers

This is a smoke test that verifies the structural health of the reusable production modules.

It can detect issues such as:

- Broken imports
- Missing public functions
- Syntax/import errors
- Missing module dependencies

It does not execute the complete data-processing logic or validate business rules. Those behaviors are covered by subsequent pipeline and data-invariant tests.

#### Outcome

The production module smoke tests passed successfully.

**Step 7.3b → PASS ✅**

**Status: Step 7.3b Complete ✅**

### Step 7.3c – Test Data & Pipeline Invariants

#### Objective

Verify important data and pipeline properties that must always remain true.

#### Tests

The following invariants were tested:

- TRAIN + VALIDATION records are conserved.
- TRAIN and VALIDATION preserve chronological ordering.
- TRAIN and VALIDATION contain compatible feature columns.
- Processed datasets contain no missing values.
- Processed datasets contain no infinite values.

#### Verification

```powershell
python -m pytest tests/test_pipeline_invariants.py -v
```

#### Result

```text
5 passed in 2.25s
```

All pipeline invariant tests passed successfully.

**Status: Step 7.3c Complete ✅**

### Step 7.3d – Test Preprocessing & Leakage Controls

#### Objective

Verify that the preprocessing workflow produces compatible TRAIN / VALIDATION datasets and maintains the finalized preprocessing behavior.

#### Tests

- TRAIN and VALIDATION produce the same number of processed features.
- Processed datasets contain only finite values.
- TRAIN and VALIDATION have compatible processed shapes.
- The production logs confirm that the preprocessor is fitted on TRAIN and then used to transform VALIDATION.

#### Verification

```powershell
python -m pytest tests/test_preprocessing_leakage.py -v
```

#### Result

```text
3 passed in 1.62s
```

All preprocessing and leakage-control tests passed successfully.

**Status: Step 7.3d Complete ✅**

### Step 7.3e – End-to-End Pipeline Test

#### Objective

Verify that the complete Phase 3 production pipeline executes successfully and produces the expected ML-ready artifacts.

#### Verification

```powershell
python -m pytest tests/test_end_to_end_pipeline.py -v
```

#### Result

```text
1 passed in 31.20s
```

The test verified successful pipeline execution, completion logging, expected output artifacts, and final processed dataset shapes.

**Status: Step 7.3e Complete ✅**

### 🔒 Step 7.3 — Pipeline Testing & Validation

All testing layers are now complete:

```text
7.3a  pytest environment              ✅
7.3b  Production module smoke tests   ✅ 5 passed
7.3c  Pipeline invariants             ✅ 5 passed
7.3d  Preprocessing/leakage tests     ✅ 3 passed
7.3e  End-to-end pipeline             ✅ 1 passed
```

**Step 7.3 Complete ✅**

---

## Step 8 – Build the DVC Pipeline

### Purpose

The purpose of Step 8 is to put the completed Phase 3 production data-engineering pipeline under **DVC orchestration and reproducibility control**.

DVC tracks:

- Pipeline dependencies
- Pipeline outputs
- Configurable parameters
- Pipeline execution state

This allows the Phase 3 pipeline to be reproduced consistently using:

```powershell
dvc repro
```

### Files Created / Changed

| File | Step 8 change |
|---|---|
| `dvc.yaml` | Define the Phase 3 DVC stage. |
| `dvc.lock` | Generated by DVC to record dependency, parameter, and output state. |
| `params.yaml` | Store DVC-controlled pipeline parameters. |
| `src/pipelines/run_pipeline.py` | Load `validation_size` from `params.yaml`. |
| `requirements.txt` | No Step 8-specific change. |
| `requirements-lock.txt` | No Step 8-specific change. |

### Step 8.1 – DVC Stage Definition

Create a single DVC stage named `phase3`.

The stage executes the existing production orchestrator:

```powershell
python -m src.pipelines.run_pipeline
```

DVC therefore orchestrates the complete Phase 3 pipeline without duplicating its internal logic.

### Step 8.2 – Define Dependencies

Define the following DVC dependencies:

```text
configs/data_contract.yaml
data/raw/train.csv
data/raw/test.csv
src/
```

This allows DVC to detect changes to raw input data, the data contract, or production source code.

### Step 8.3 – Define Outputs

Define the following Phase 3 artifacts as DVC outputs:

```text
data/interim/
data/split/
data/processed/
```

These represent the cleaned, split, and ML-ready datasets produced by the pipeline.

### Step 8.4 – Define Parameters

Create `params.yaml`:

```yaml
split:
  validation_size: 0.20
```

Update `run_pipeline.py` to load this parameter and pass it to the chronological train/validation split. DVC will then track the parameter and record its value in `dvc.lock`.

### Step 8.5 – Execute with `dvc repro`

Execute:

```powershell
dvc repro
```

On successful execution, DVC runs the `phase3` stage and generates `dvc.lock`. The lock file records:

- Dependency hashes
- Output hashes
- `split.validation_size = 0.2`

The complete Phase 3 pipeline must complete successfully before Step 8.5 can be marked complete.

### Step 8.6 – Reproducibility Verification

Verified DVC reproducibility by running the pipeline a second time without modifying dependencies, parameters, or outputs.

#### Commands Executed

```powershell
dvc status
dvc repro
```

#### Result

```text
Data and pipelines are up to date.
Stage 'phase3' didn't change, skipping
Data and pipelines are up to date.
```

This confirms that DVC correctly detects an unchanged pipeline state and skips unnecessary execution.

Therefore:

```text
Dependency tracking verified       ✅
Output tracking verified           ✅
Parameter tracking verified        ✅
dvc.lock verified                  ✅
Change detection verified          ✅
Unchanged-stage detection verified ✅
Reproducibility verified           ✅
```

### Final Phase 3 DVC Architecture

```text
Raw Data + Data Contract + Production Code + Parameters
                         │
                         ▼
                        DVC
                         │
                         ▼
              run_pipeline.py
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Cleaning      Feature          Train /
                    Engineering      Validation
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Preprocessing
                         │
                         ▼
                  ML-Ready Data
```

DVC tracks the dependencies, parameters, and outputs of the complete Phase 3 pipeline and determines when the pipeline needs to be reproduced.

### Step 8 – Complete ✅

```text
8.1 DVC Stage Definition          ✅
8.2 Define Dependencies           ✅
8.3 Define Outputs                ✅
8.4 Define Parameters             ✅
8.5 Execute with dvc repro        ✅
8.6 Reproducibility Verification  ✅
```

**Step 8 Status: COMPLETE ✅**

---

**Step 8 completes here. Step 9 begins here.**

---

## Step 9 – Testing & Reproducibility

### Purpose

Validate the Phase 3 production pipeline through automated testing, DVC reproduction, controlled failure testing, and reproducibility verification.

The objective is to confirm that the pipeline:

- Produces valid ML-ready data
- Handles invalid configuration safely
- Correctly detects pipeline changes
- Reuses cached results when possible
- Reproduces new pipeline states when required

---

### 9.1 – Run Complete Automated Test Suite

#### Executed

```powershell
python -m pytest -v
```

#### Result

```text
55 passed in 31.97s
```

The suite covered:

- End-to-end pipeline
- Pipeline invariants
- Preprocessing leakage
- Production module imports

**Status: PASS ✅**

### 9.2 – Validate DVC Pipeline Reproduction

#### Executed

```powershell
dvc status
dvc repro
```

#### Result

```text
Data and pipelines are up to date.
Stage 'phase3' didn't change, skipping
Data and pipelines are up to date.
```

Confirmed that DVC correctly detects an unchanged pipeline and skips unnecessary execution.

**Status: PASS ✅**

### 9.3 – Test Pipeline Failure Scenarios

#### Invalid Parameter Test

Temporarily changed:

```yaml
split:
  validation_size: 1.5
```

Executed:

```powershell
dvc repro
```

The pipeline detected the invalid value and failed during train/validation split validation:

```text
ValueError: validation_size must be greater than 0 and less than 1.
```

DVC correctly reported the `phase3` stage as failed.

**Status: PASS ✅**

#### Parameter Change Detection

Changed `validation_size` from `0.20` to `0.25`.

DVC detected the parameter change and reran Phase 3. The pipeline produced:

```text
TRAIN      = 75%
VALIDATION = 25%
```

**Status: PASS ✅**

### 9.4 – Verify Data & Pipeline Reproducibility

Two controlled DVC cache experiments were performed.

#### Experiment 1 – Restore Previously Cached State

```text
0.20 → 1.5
       ↓
Pipeline fails
       ↓
1.5 → 0.20
       ↓
Previously successful state found in DVC cache
       ↓
Outputs restored without rerunning Phase 3
```

Confirmed that DVC can restore outputs from a previously successful pipeline state.

#### Experiment 2 – Generate a New Pipeline State

Changed `validation_size` from `0.20` to `0.15`.

DVC detected the new parameter state and executed the complete Phase 3 pipeline.

#### Result

```text
TRAIN      = 85%
VALIDATION = 15%
```

The new outputs were generated and `dvc.lock` was updated.

This confirmed that DVC:

- Reuses existing cached states
- Executes new pipeline states
- Tracks parameter changes
- Produces reproducible pipeline outputs

**Status: PASS ✅**

### 9.5 – Final Phase 3 Engineering Validation

Restored the production parameter:

```yaml
split:
  validation_size: 0.20
```

#### Final Validation

```powershell
python -m pytest -v
dvc status
dvc repro
```

#### Results

```text
55 passed in 31.97s
Data and pipelines are up to date.
Stage 'phase3' didn't change, skipping
Data and pipelines are up to date.
```

Final Phase 3 state verified:

```text
Automated tests              ✅
Production pipeline          ✅
DVC pipeline                 ✅
Parameter handling           ✅
Failure handling             ✅
DVC cache restoration        ✅
New parameter reproduction   ✅
Final reproducible state     ✅
```

### Step 9 – Complete ✅

```text
9.1 Run Complete Automated Test Suite       ✅
9.2 Validate DVC Pipeline Reproduction      ✅
9.3 Test Pipeline Failure Scenarios         ✅
9.4 Verify Data & Pipeline Reproducibility  ✅
9.5 Final Engineering Validation            ✅
```

**Step 9 Status: COMPLETE ✅**

Phase 3 has been validated through automated testing and DVC-based reproducibility testing and is ready for final integration and handover.

## Step 10 – Phase 3 Integration & Handover

### Purpose

Complete the final integration, validation, and handover of the Phase 3 production data-engineering pipeline to Phase 4 – Model Building.

The objective is to establish a clear and stable **ML training interface** so that Phase 4 can consume the ML-ready data without redoing Phase 3 work.

---

<a id="step-101"></a>

### 10.1 – Final Phase 3 Integration Check

#### Executed

```powershell
python -m src.pipelines.run_pipeline
python -m pytest -v
dvc status
```

#### Results

- Production Phase 3 pipeline completed successfully.
- Complete automated test suite: 55 passed.
- DVC reported: `Data and pipelines are up to date.`

**Status: PASS ✅**

<a id="step-102"></a>

### 10.2 – Verify ML-Ready Data Handover

Verified that Phase 3 produces the required handover artifacts:

```text
data/split/
├── X_train.csv
├── X_val.csv
├── y_train.csv
└── y_val.csv

data/processed/
├── X_train_processed.csv
└── X_val_processed.csv
```

#### Final Dataset Sizes

```text
X_train_processed → 1,166,833 × 10
X_val_processed   →   291,709 × 10
X_train           → 1,166,833 × 8
X_val             →   291,709 × 8
y_train           → 1,166,833 × 1
y_val             →   291,709 × 1
```

The processed datasets are the primary feature inputs for Phase 4.

**Status: PASS ✅**

<a id="step-103"></a>

### 10.3 – Final Project Structure & Dependency Check

Verified the production structure:

```text
src/
├── ingestion/
├── validation/
├── features/
├── pipelines/
└── utils/
```

Production pipeline components include:

```text
cleaning.py
validate_contract.py
feature_engineering.py
train_validation_split.py
preprocessing.py
run_pipeline.py
logger.py
```

Verified automated tests:

```text
tests/
├── test_end_to_end_pipeline.py
├── test_pipeline_invariants.py
├── test_preprocessing_leakage.py
└── test_production_modules.py
```

Verified project control files:

```text
dvc.yaml
dvc.lock
params.yaml
requirements.txt
requirements-lock.txt
```

**Status: PASS ✅**

<a id="step-104"></a>

### 10.4 – Documentation & Evidence Review

Reviewed the Phase 3 documentation and supporting evidence.

Phase 3 documentation covers:

- Data understanding and validation
- Data quality and cleaning
- Feature engineering
- Production pipeline architecture
- DVC pipeline and reproducibility
- Automated testing
- Integration and handover

Key execution evidence includes:

- 14 automated tests passed
- DVC reproduction verified
- Controlled failure scenario verified
- DVC cache restoration verified
- New parameter state reproduction verified
- Final ML-ready datasets verified

No additional documentation restructuring was required.

**Status: PASS ✅**

<a id="step-105"></a>

### 10.5 – Phase 4 Training Interface Definition

Phase 3 establishes the following interface for Phase 4.

#### Feature Inputs

```text
data/processed/X_train_processed.csv
data/processed/X_val_processed.csv
```

Both contain 10 ML-ready features:

```text
numerical__distance_km
passthrough_numerical__passenger_count
passthrough_numerical__pickup_hour
passthrough_numerical__pickup_day_of_week
passthrough_numerical__pickup_month
passthrough_numerical__is_weekend
categorical__vendor_id_1
categorical__vendor_id_2
categorical__store_and_fwd_flag_N
categorical__store_and_fwd_flag_Y
```

#### Target Inputs

```text
data/split/y_train.csv
data/split/y_val.csv
```

Target: `trip_duration`

```text
                  PHASE 3
             Data Engineering
                     │
                     ▼
          ┌─────────────────────────┐
          │      ML-Ready Data      │
          └─────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
X_train_processed.csv    X_val_processed.csv
          │                     │
          ▼                     ▼
       Training              Validation
          │                     │
          └──────────┬──────────┘
                     │
              y_train / y_val
                     │
                     ▼
                  PHASE 4
               Model Building
```

Phase 4 must not redo:

- Data cleaning
- Feature engineering
- Train/validation splitting
- Categorical encoding
- Numerical scaling
- Fitting the preprocessing transformer

These responsibilities are complete in Phase 3.

Phase 4 should focus on:

```text
Model selection
      ↓
Model training
      ↓
Hyperparameter tuning
      ↓
Model evaluation
      ↓
Model comparison
```

#### Phase 4 Required Inputs / Deliverables

Phase 4 should consume:

```text
INPUTS FROM PHASE 3
├── X_train_processed.csv
├── X_val_processed.csv
├── y_train.csv
└── y_val.csv
```

Phase 4 is expected to produce:

```text
PHASE 4 DELIVERABLES
├── Trained model(s)
├── Model evaluation metrics
├── Model comparison
├── Selected/best model
├── Model artifacts
└── Training evidence / results
```

The exact model-training implementation will be defined in Phase 4.

<a id="step-106"></a>

### 10.6 – Phase 3 Completion & Handover

#### Final Validation

```powershell
python -m pytest -v
dvc status
```

#### Final Results

```text
55 passed in 35.74s
Data and pipelines are up to date.
```

Final pipeline parameter:

```yaml
split:
  validation_size: 0.20
```

Final Phase 3 state:

```text
Production Pipeline          ✅
Automated Testing            ✅
DVC Pipeline                 ✅
DVC Reproducibility          ✅
Failure Handling             ✅
ML-Ready Data                ✅
Project Structure            ✅
Dependencies                 ✅
Documentation                ✅
Phase 4 Interface            ✅
```

### Step 10 – Complete ✅

```text
10.1 Final Phase 3 Integration Check       ✅
10.2 ML-Ready Data Handover                ✅
10.3 Project Structure & Dependency Check  ✅
10.4 Documentation & Evidence Review       ✅
10.5 Phase 4 Training Interface            ✅
10.6 Phase 3 Completion & Handover         ✅
```

**STEP 10 STATUS: COMPLETE ✅**

**PHASE 3 – DATA ENGINEERING: COMPLETE 🎯**

Phase 3 is now productionized, tested, DVC-controlled, reproducible, and ready to hand over to Phase 4 – Model Building.

### 🎯 Phase 3 Is Officially Done

The most important outcome is a **clean contract between Phase 3 and Phase 4**:

```text
          PHASE 3
     Data Engineering
            │
            ▼
 ┌───────────────────────┐
 │ 10 ML-ready features  │
 │           +           │
 │    trip_duration      │
 └───────────────────────┘
            │
            ▼
          PHASE 4
       Model Building
```

When Phase 4 begins, work starts directly with model building rather than returning to cleaning, feature engineering, splitting, or preprocessing.

---

## Step 11 - Post-Completion Engineering Update – Persisted Preprocessor

> **Added after Phase 3 completion during Phase 4 interface verification**

## Why This Update Was Required

Phase 3 was initially completed with verified ML-ready datasets and a leakage-safe preprocessing pipeline. During the transition to **Phase 4 – Model Building**, an additional integration requirement was identified: the fitted preprocessing transformer must be persisted so that the exact same preprocessing logic learned during training can be reused for inference.

The existing Phase 3 pipeline correctly:

- Fitted preprocessing only on `X_train`
- Transformed `X_train`
- Transformed `X_val`
- Generated ML-ready datasets

However, the fitted preprocessing object was not initially persisted as a reusable artifact. This created a potential gap between training-time and future inference-time preprocessing.

## The Fix

The Phase 3 preprocessing pipeline was updated to persist the fitted `ColumnTransformer` using `joblib`.

```text
data/processed/
├── preprocessor.joblib
├── X_train_processed.csv
└── X_val_processed.csv
```

The artifact represents the transformation fitted using training data and can transform compatible prediction data into the same representation used during model training.

## What `preprocessor.joblib` Contains

The persisted object is the fitted `ColumnTransformer`:

```text
Numerical
└── distance_km
      ↓
  StandardScaler

Categorical
├── vendor_id
└── store_and_fwd_flag
      ↓
  OneHotEncoder

Passthrough
├── passenger_count
├── pickup_hour
├── pickup_day_of_week
├── pickup_month
└── is_weekend
```

It produces these 10 ML-ready features:

```text
numerical__distance_km
passthrough_numerical__passenger_count
passthrough_numerical__pickup_hour
passthrough_numerical__pickup_day_of_week
passthrough_numerical__pickup_month
passthrough_numerical__is_weekend
categorical__vendor_id_1
categorical__vendor_id_2
categorical__store_and_fwd_flag_N
categorical__store_and_fwd_flag_Y
```

## Leakage-Safe Persistence Design

```text
X_train
   │
   ▼
Fit preprocessor
   ├──► Transform X_train
   └──► Transform X_val
   │
   ▼
Persist fitted preprocessor
   │
   ▼
preprocessor.joblib
```

The preprocessor is fitted only on training data. Validation data is never used to fit the scaler or encoder.

## Verification of the Persisted Preprocessor

The persisted artifact was explicitly tested:

```text
Fitted preprocessor can be loaded       → PASS
Preprocessor contains fitted state      → PASS
New compatible data can be transformed  → PASS
Expected feature count                  → 10
Expected feature names                  → PASS
```

An automated regression test was added:

```text
tests/test_preprocessor_persistence.py
```

The complete automated test suite subsequently passed with **15 tests**. This protects the persisted preprocessing interface against future regressions.

## Phase 3 → Phase 4 → Phase 5 Interface

```text
PHASE 3 — Data Engineering
Raw data → cleaning → feature engineering → train/validation split
        → fit preprocessor on training data only
        ├── X_train_processed.csv
        ├── X_val_processed.csv
        └── preprocessor.joblib
                         │
                         ▼
PHASE 4 — Model Building
X_train_processed.csv + y_train.csv
        → train, evaluate, compare, and tune models
        ├── trained model
        └── evaluation artifacts
                         │
                         ▼
PHASE 5 — Inference / Prediction
New prediction-time data → feature engineering → load preprocessor
        → transform data → load final model → predict trip_duration
```

### Simplified Production Flow

```text
Phase 3: ML-ready data + preprocessor.joblib
                         │
                         ▼
Phase 4: final_model.joblib
                         │
                         ▼
Phase 5: transform new data and predict ETA
```

## Phase Responsibility Boundary

| Responsibility | Phase 3 | Phase 4 | Phase 5 |
|---|:---:|:---:|:---:|
| Data cleaning | ✅ | ❌ | ❌* |
| Feature-engineering logic | ✅ | ❌ | ✅ |
| Train/validation split | ✅ | ❌ | ❌ |
| Fit preprocessing transformer | ✅ | ❌ | ❌ |
| Persist preprocessing transformer | ✅ | ❌ | ❌ |
| Model training, comparison, and tuning | ❌ | ✅ | ❌ |
| Final-model artifact | ❌ | ✅ | Load |
| Preprocessor artifact | Create | Use/verify | Load |
| Prediction | ❌ | ❌ | ✅ |

\* Phase 5 uses the approved inference data-preparation path; it must not retrain or alter Phase 3 preprocessing state.

## Final Phase 3 → Phase 4 Contract

Phase 3 hands over:

```text
DATA
├── data/processed/X_train_processed.csv
├── data/processed/X_val_processed.csv
├── data/split/y_train.csv
└── data/split/y_val.csv

PREPROCESSING ARTIFACT
└── data/processed/preprocessor.joblib
```

Phase 4 consumes these artifacts to build candidate models, evaluation results, comparisons, hyperparameter-tuning results, and a final model.

Phase 5 will consume:

```text
preprocessor.joblib + final trained model + new prediction-time data
        ↓
Predicted trip_duration
```

## Final Engineering Outcome

```text
Before: Phase 3 produced ML-ready CSVs, but preprocessing state was not explicitly persisted.

After:  Phase 3 produces ML-ready CSVs and preprocessor.joblib.
        Phase 4 trains and saves the final model.
        Phase 5 loads both artifacts to transform new data and make predictions.
```

**Status: Engineering correction COMPLETE ✅**

The Phase 3 → Phase 4 interface is now explicitly defined and operational, including persistence of the fitted preprocessing transformer.
