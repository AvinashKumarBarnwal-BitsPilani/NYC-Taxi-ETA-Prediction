# Phase 3B - Data Quality Analysis & Cleaning

## Table of Contents

- [Step 3 - Data Quality Analysis ✅](#31--analyze-missing-values)
  - [3.1 Analyze Missing Values](#31--analyze-missing-values)
  - [3.2 Analyze Duplicate Records](#32--analyze-duplicate-records)
  - [3.3 Analyze Invalid Values](#33--analyze-invalid-values)
  - [3.4 Analyze Outliers](#34--analyze-outliers)
  - [3.5 Analyze Impossible Timestamps](#35--analyze-impossible-timestamps)
  - [3.6 Analyze Invalid Geographical Coordinates](#36--analyze-invalid-geographical-coordinates)
  - [3.7 Analyze Unexpected Categorical Values](#37--analyze-unexpected-categorical-values)
  - [3.8 Analyze Target Distribution](#38--analyze-target-distribution)
  - [3.9 Analyze Extremely Short Trips](#39--analyze-extremely-short-trips)
  - [3.10 Analyze Extremely Long Trips](#310--analyze-extremely-long-trips)
  - [3.11 Analyze Unrealistic Passenger Counts](#311--analyze-unrealistic-passenger-counts)
  - [3.12 Data Quality Report](#312--data-quality-report)
  - [Visual Data Quality Analysis](#visual-data-quality-analysis)

- [Step 4 - Data Cleaning ✅](#41--handle-missing-values)
  - [4.1 Handle Missing Values](#41--handle-missing-values)
  - [4.2 Handle Duplicates](#42--handle-duplicates)
  - [4.3 Handle Invalid Records](#43--handle-invalid-records)
  - [4.4 Handle Impossible Timestamps](#44--handle-impossible-timestamps)
  - [4.5 Handle Invalid Coordinates](#45--handle-invalid-coordinates)
  - [4.6 Handle Invalid Passenger Counts](#46--handle-invalid-passenger-counts)
  - [4.7 Handle Invalid Target Values](#47--handle-invalid-target-values)
  - [4.8 Handle Justified Outliers](#48--handle-justified-outliers)
  - [4.9 Ensure Consistent Data Types](#49--ensure-consistent-data-types)
  - [4.10 Verify Cleaned Dataset](#410--verify-cleaned-dataset)


### 3.1 – Analyze Missing Values

The complete raw TRAIN and TEST datasets were analyzed to identify missing values and quantify their impact.

The analysis was performed on **100% of the raw data**.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv'); test=pd.read_csv('data/raw/test.csv'); print('=== TRAIN MISSING VALUE ANALYSIS ==='); train_missing=train.isnull().sum(); train_pct=(train_missing/len(train)*100).round(4); print(pd.DataFrame({'Missing_Count':train_missing,'Missing_Percent':train_pct}).to_string()); print('\nRecords with >=1 missing value:',train.isnull().any(axis=1).sum()); print('\n=== TEST MISSING VALUE ANALYSIS ==='); test_missing=test.isnull().sum(); test_pct=(test_missing/len(test)*100).round(4); print(pd.DataFrame({'Missing_Count':test_missing,'Missing_Percent':test_pct}).to_string()); print('\nRecords with >=1 missing value:',test.isnull().any(axis=1).sum())"
```

#### Output

```text
=== TRAIN MISSING VALUE ANALYSIS ===
                    Missing_Count  Missing_Percent
id                              0              0.0
vendor_id                       0              0.0
pickup_datetime                 0              0.0
dropoff_datetime                0              0.0
passenger_count                 0              0.0
pickup_longitude                0              0.0
pickup_latitude                 0              0.0
dropoff_longitude               0              0.0
dropoff_latitude                0              0.0
store_and_fwd_flag              0              0.0
trip_duration                   0              0.0

Records with >=1 missing value: 0

=== TEST MISSING VALUE ANALYSIS ===
                    Missing_Count  Missing_Percent
id                              0              0.0
vendor_id                       0              0.0
pickup_datetime                 0              0.0
passenger_count                 0              0.0
pickup_longitude                0              0.0
pickup_latitude                 0              0.0
dropoff_longitude               0              0.0
dropoff_latitude                0              0.0
store_and_fwd_flag              0              0.0

Records with >=1 missing value: 0
```

### Observations

No missing values were found in either dataset.

| Dataset | Total Records with Missing Values | Missing Percentage |
|---|---:|---:|
| TRAIN | 0 | 0% |
| TEST | 0 | 0% |

Every column in both datasets contains complete data.

### Data Quality Assessment

```text
Problem:
Missing values

TRAIN:
0 records affected

TEST:
0 records affected

Impact:
No missing-value related data-quality issue identified.

Decision:
No missing-value cleaning or imputation is required.
```

Since there are no missing values, no records will be removed or modified due to missingness.

The existing data contract rule remains applicable:

```text
Required fields must not contain missing values.
```

### Step 3.1 Conclusion

The complete raw TRAIN and TEST datasets contain **no missing values**.

Therefore:

- No missing-value imputation is required.
- No records need to be removed because of missing values.
- No additional missing-value handling logic is required during Step 4 cleaning.

**Status: Step 3.1 Complete ✅**

### 3.2 – Analyze Duplicate Records

The complete raw TRAIN and TEST datasets were analyzed for duplicate records.

Two types of duplication were checked:

1. Exact duplicate rows
2. Duplicate `id` values

The analysis was performed on **100% of the raw data**.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv'); test=pd.read_csv('data/raw/test.csv'); print('=== TRAIN DUPLICATE ANALYSIS ==='); print('Total records:',len(train)); print('Exact duplicate rows:',train.duplicated().sum()); print('Duplicate IDs:',train['id'].duplicated().sum()); print('Unique IDs:',train['id'].nunique()); print('Records belonging to duplicated IDs:',train[train['id'].duplicated(keep=False)]['id'].count()); print('\n=== TEST DUPLICATE ANALYSIS ==='); print('Total records:',len(test)); print('Exact duplicate rows:',test.duplicated().sum()); print('Duplicate IDs:',test['id'].duplicated().sum()); print('Unique IDs:',test['id'].nunique()); print('Records belonging to duplicated IDs:',test[test['id'].duplicated(keep=False)]['id'].count())"
```

#### Output

```text
=== TRAIN DUPLICATE ANALYSIS ===
Total records: 1458644
Exact duplicate rows: 0
Duplicate IDs: 0
Unique IDs: 1458644
Records belonging to duplicated IDs: 0

=== TEST DUPLICATE ANALYSIS ===
Total records: 625134
Exact duplicate rows: 0
Duplicate IDs: 0
Records belonging to duplicated IDs: 0
```

### Observations

No duplicate records were identified in either dataset.

| Dataset | Total Records | Exact Duplicate Rows | Duplicate IDs | Unique IDs |
|---|---:|---:|---:|---:|
| TRAIN | 1,458,644 | 0 | 0 | 1,458,644 |
| TEST | 625,134 | 0 | 0 | 625,134 |

Every record has a unique `id`, and no completely duplicated rows were found.

### Data Quality Assessment

```text
Problem:
Duplicate records

TRAIN:
0 records affected

TEST:
0 records affected

Impact:
No duplicate-related data-quality issue identified.

Decision:
No duplicate removal or deduplication is required.
```

The duplicate validation rules defined in the data contract are therefore satisfied by both datasets.

### Step 3.2 Conclusion

The complete raw TRAIN and TEST datasets contain:

- No exact duplicate rows.
- No duplicate `id` values.
- No records belonging to duplicated IDs.

Therefore, no deduplication is required during Step 4 cleaning.

The existing data contract rules remain applicable:

```text
Each record must have a unique id.
Complete duplicate rows are not allowed.
```

**Status: Step 3.2 Complete ✅**

### 3.3 – Analyze Invalid Values

The complete raw TRAIN and TEST datasets were analyzed to identify values that violate the validation rules defined in the data contract.

The analysis was performed on **100% of the raw data**.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv'); test=pd.read_csv('data/raw/test.csv'); print('=== TRAIN INVALID VALUE ANALYSIS ==='); print('Invalid vendor_id:',(~train['vendor_id'].isin([1,2])).sum()); print('Invalid store_and_fwd_flag:',(~train['store_and_fwd_flag'].isin(['N','Y'])).sum()); print('Invalid passenger_count:',(train['passenger_count']<=0).sum()); print('Invalid pickup latitude:',(~train['pickup_latitude'].between(-90,90)).sum()); print('Invalid pickup longitude:',(~train['pickup_longitude'].between(-180,180)).sum()); print('Invalid dropoff latitude:',(~train['dropoff_latitude'].between(-90,90)).sum()); print('Invalid dropoff longitude:',(~train['dropoff_longitude'].between(-180,180)).sum()); print('Invalid trip_duration:',(train['trip_duration']<=0).sum()); print('\n=== TEST INVALID VALUE ANALYSIS ==='); print('Invalid vendor_id:',(~test['vendor_id'].isin([1,2])).sum()); print('Invalid store_and_fwd_flag:',(~test['store_and_fwd_flag'].isin(['N','Y'])).sum()); print('Invalid passenger_count:',(test['passenger_count']<=0).sum()); print('Invalid pickup latitude:',(~test['pickup_latitude'].between(-90,90)).sum()); print('Invalid pickup longitude:',(~test['pickup_longitude'].between(-180,180)).sum()); print('Invalid dropoff latitude:',(~test['dropoff_latitude'].between(-90,90)).sum()); print('Invalid dropoff longitude:',(~test['dropoff_longitude'].between(-180,180)).sum())"
```

#### Output

```text
=== TRAIN INVALID VALUE ANALYSIS ===
Invalid vendor_id: 0
Invalid store_and_fwd_flag: 0
Invalid passenger_count: 60
Invalid pickup latitude: 0
Invalid pickup longitude: 0
Invalid dropoff latitude: 0
Invalid dropoff longitude: 0
Invalid trip_duration: 0

=== TEST INVALID VALUE ANALYSIS ===
Invalid vendor_id: 0
Invalid store_and_fwd_flag: 0
Invalid passenger_count: 23
Invalid pickup latitude: 0
Invalid pickup longitude: 0
Invalid dropoff latitude: 0
Invalid dropoff longitude: 0
```

### Observations

The analysis confirms that almost all values comply with the current data contract.

The only invalid-value violation identified is:

```text
passenger_count <= 0
```

| Validation Rule | TRAIN | TEST |
|---|---:|---:|
| Invalid `vendor_id` | 0 | 0 |
| Invalid `store_and_fwd_flag` | 0 | 0 |
| Invalid `passenger_count` | **60** | **23** |
| Invalid pickup latitude | 0 | 0 |
| Invalid pickup longitude | 0 | 0 |
| Invalid dropoff latitude | 0 | 0 |
| Invalid dropoff longitude | 0 | 0 |
| Invalid `trip_duration` | 0 | N/A |

### Data Quality Assessment

```text
Problem:
Invalid passenger_count values

TRAIN:
60 records affected

TEST:
23 records affected

Total:
83 records affected
```

These records violate the data contract rule:

```text
passenger_count > 0
```

### Important Distinction

This analysis checks **contract violations**, not statistical or geographical anomalies.

For example:

```text
Invalid value
      ↓
Directly violates defined data contract
```

Whereas:

```text
Unusual geographic coordinate
      ↓
Requires geographic investigation
      ↓
Step 3.6
```

And:

```text
Extremely long trip_duration
      ↓
Requires outlier / extreme-value investigation
      ↓
Steps 3.4 / 3.10
```

Therefore, unusual but contract-valid values are not classified as invalid at this stage.

### Decision

No data is modified or removed during Step 3.3.

The **83 records with `passenger_count <= 0`** will be investigated further before making a cleaning decision in Step 4.

The current findings are:

```text
TRAIN → 60 invalid passenger counts
TEST  → 23 invalid passenger counts
```

### Step 3.3 Conclusion

The current data-quality analysis identified **83 invalid passenger-count records** across TRAIN and TEST.

All other values checked against the current data contract are valid.

**Status: Step 3.3 Complete ✅**

### 3.4 – Analyze Outliers

The complete raw TRAIN dataset was analyzed for statistically unusual observations using the **Interquartile Range (IQR)** method.

The purpose of this analysis is to identify potential outliers for further investigation.

> **Important:** A statistical outlier is not automatically an invalid record. No records are removed or modified as part of this analysis.

### IQR Method

For each numerical feature:

```text
Q1  = 25th percentile
Q3  = 75th percentile
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these bounds are classified as **statistical outliers** for analysis purposes.

### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']); cols=train.columns; print('=== IQR OUTLIER ANALYSIS ==='); print(f'{'Feature':<25}{'Q1':>12}{'Q3':>12}{'IQR':>12}{'Lower':>15}{'Upper':>15}{'Outliers':>12}{'Outlier %':>12}'); print('-'*123); [print(f'{c:<25}{(q1:=train[c].quantile(0.25)):>12.2f}{(q3:=train[c].quantile(0.75)):>12.2f}{(iqr:=q3-q1):>12.2f}{(lower:=q1-1.5*iqr):>15.2f}{(upper:=q3+1.5*iqr):>15.2f}{(n:=((train[c]<lower)|(train[c]>upper)).sum()):>12,}{(n/len(train)*100):>11.2f}%') for c in cols]"
```

### Output

```text
=== IQR OUTLIER ANALYSIS ===
Feature                            Q1          Q3         IQR          Lower          Upper    Outliers   Outlier %
---------------------------------------------------------------------------------------------------------------------------
passenger_count                  1.00        2.00        1.00          -0.50           3.50     154,830      10.61%
pickup_longitude               -73.99      -73.97        0.02         -74.03         -73.93      84,322       5.78%
pickup_latitude                 40.74       40.77        0.03          40.69          40.81      52,743       3.62%
dropoff_longitude              -73.99      -73.96        0.03         -74.03         -73.92      77,969       5.35%
dropoff_latitude                40.74       40.77        0.03          40.68          40.82      71,990       4.94%
trip_duration                  397.00     1075.00      678.00        -620.00        2092.00      74,220       5.09%
```

### Observations

The IQR method identified statistically unusual observations across all analyzed numerical features.

| Feature | IQR Upper Bound | Outliers | Outlier % |
|---|---:|---:|---:|
| `passenger_count` | 3.50 | 154,830 | 10.61% |
| `pickup_longitude` | -73.93 | 84,322 | 5.78% |
| `pickup_latitude` | 40.81 | 52,743 | 3.62% |
| `dropoff_longitude` | -73.92 | 77,969 | 5.35% |
| `dropoff_latitude` | 40.82 | 71,990 | 4.94% |
| `trip_duration` | 2,092 sec | 74,220 | 5.09% |

### Important Findings

The IQR analysis demonstrates that a statistical outlier does not necessarily represent a bad record.

For example:

```text
passenger_count
      ↓
IQR identifies 154,830 outliers
      ↓
But only 60 records violate our data contract
      ↓
Therefore, IQR outlier ≠ invalid value
```

Similarly, the geographic features have thousands of IQR-based outliers. These observations may simply represent trips outside the central geographic concentration of the dataset.

The `trip_duration` feature is also highly right-skewed. The IQR method identifies **74,220 records (5.09%)** as statistical outliers, but this does not mean these trips are invalid.

### Data Quality Assessment

The IQR analysis is therefore being used as an **investigation mechanism**, not as an automatic cleaning rule.

```text
Statistical Outlier
        ↓
Investigate
        ↓
Determine whether it is:
        │
        ├── Valid but unusual → Keep
        │
        └── Invalid / suspicious → Consider cleaning
```

Further investigation will be performed separately for:

- Geographic anomalies → **Step 3.6**
- Target distribution → **Step 3.8**
- Extremely short trips → **Step 3.9**
- Extremely long trips → **Step 3.10**
- Passenger-count anomalies → **Step 3.11**

### Decision

No records are removed or modified based solely on the IQR analysis.

The IQR results will be used as supporting evidence during the subsequent data-quality investigations.

### Step 3.4 Conclusion

The dataset contains several statistically unusual observations across numerical features.

However, these observations are **not automatically classified as invalid**.

The analysis confirms the importance of separating:

```text
Statistical Outlier
        ≠
Invalid Record
```

Further domain-specific investigation is required before any outlier is removed or transformed.

**Status: Step 3.4 Complete ✅**

### 3.5 – Analyze Impossible Timestamps

The complete raw TRAIN dataset was analyzed to identify impossible or internally inconsistent timestamp records.

The analysis focused on:

- Missing `pickup_datetime`
- Missing `dropoff_datetime`
- `dropoff_datetime <= pickup_datetime`
- Non-positive calculated trip duration
- Mismatch between calculated duration and `trip_duration`

The analysis was performed on **100% of the raw training data**.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['pickup_datetime','dropoff_datetime','trip_duration'],parse_dates=['pickup_datetime','dropoff_datetime']); calculated=(train['dropoff_datetime']-train['pickup_datetime']).dt.total_seconds(); print('=== IMPOSSIBLE TIMESTAMP ANALYSIS ==='); print('Missing pickup_datetime:',train['pickup_datetime'].isna().sum()); print('Missing dropoff_datetime:',train['dropoff_datetime'].isna().sum()); print('Dropoff <= Pickup:',(train['dropoff_datetime']<=train['pickup_datetime']).sum()); print('Calculated duration <= 0:',(calculated<=0).sum()); print('Duration mismatch:',(train['trip_duration']!=calculated).sum()); print('\n=== CALCULATED DURATION SUMMARY ==='); print(calculated.describe().to_string()); print('\n=== LARGEST DATETIME DIFFERENCES ==='); print(calculated.nlargest(10).to_string(index=False))"
```

#### Output

```text
=== IMPOSSIBLE TIMESTAMP ANALYSIS ===
Missing pickup_datetime: 0
Missing dropoff_datetime: 0
Dropoff <= Pickup: 0
Calculated duration <= 0: 0
Duration mismatch: 0

=== CALCULATED DURATION SUMMARY ===
count    1.458644e+06
mean     9.594923e+02
std      5.237432e+03
min      1.000000e+00
25%      3.970000e+02
50%      6.620000e+02
75%      1.075000e+03
max      3.526282e+06

=== LARGEST DATETIME DIFFERENCES ===
3526282.0
2227612.0
2049578.0
1939736.0
  86392.0
  86391.0
  86390.0
  86387.0
  86385.0
  86379.0
```

### Observations

No impossible or internally inconsistent timestamps were identified.

| Timestamp Validation | Invalid Records |
|---|---:|
| Missing `pickup_datetime` | 0 |
| Missing `dropoff_datetime` | 0 |
| `dropoff_datetime <= pickup_datetime` | 0 |
| Calculated duration <= 0 | 0 |
| `trip_duration` mismatch | 0 |

Therefore, all training records satisfy the basic chronological and duration-consistency rules.

### Datetime Consistency

The following relationship holds for all training records:

```text
trip_duration
      =
dropoff_datetime - pickup_datetime
```

There were:

```text
Duration mismatches → 0
```

This confirms that the target variable is internally consistent with the recorded timestamps.

### Extreme Timestamp Differences

The largest calculated datetime differences were:

```text
3,526,282 seconds
2,227,612 seconds
2,049,578 seconds
1,939,736 seconds
86,392 seconds
86,391 seconds
86,390 seconds
86,387 seconds
86,385 seconds
86,379 seconds
```

These observations represent **extremely long trips**, but they are not classified as impossible timestamps because:

```text
pickup_datetime < dropoff_datetime
        AND
calculated duration = trip_duration
```

The extremely long trips will therefore be investigated separately under:

```text
Step 3.10 – Analyze Extremely Long Trips
```

### Data Quality Assessment

```text
Problem:
Impossible or inconsistent timestamps

Records affected:
0

Impact:
No timestamp-order or duration-consistency issue identified.

Decision:
No timestamp cleaning is required.
```

The current data contract rule remains valid:

```text
pickup_datetime < dropoff_datetime

trip_duration =
dropoff_datetime - pickup_datetime
```

### Step 3.5 Conclusion

The complete raw TRAIN dataset contains **no impossible or internally inconsistent timestamps**.

All records have:

- Valid pickup timestamps
- Valid dropoff timestamps
- Pickup occurring before dropoff
- Positive calculated duration
- `trip_duration` exactly matching the datetime difference

The extremely long duration observations are retained for further domain-specific investigation in **Step 3.10**.

**Status: Step 3.5 Complete ✅**

### 3.6 – Analyze Invalid Geographical Coordinates

The complete raw TRAIN and TEST datasets were analyzed to identify geographically unusual coordinates.

The analysis was performed in two stages:

1. Validate coordinates against the global latitude/longitude ranges.
2. Identify records outside a broad NYC-area geographic bounding box.

The global coordinate rules were already established in the data contract:

```text
Latitude  → [-90, +90]
Longitude → [-180, +180]
```

A broad NYC-area bounding box was used for investigation:

```text
Longitude: -75 to -72
Latitude :  40 to 42
```

> This broad bounding box is used as an **analysis criterion**, not yet as a final cleaning rule.

### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']); test=pd.read_csv('data/raw/test.csv',usecols=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']); cols=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']; train_out=~(train['pickup_longitude'].between(-75,-72)&train['dropoff_longitude'].between(-75,-72)&train['pickup_latitude'].between(40,42)&train['dropoff_latitude'].between(40,42)); test_out=~(test['pickup_longitude'].between(-75,-72)&test['dropoff_longitude'].between(-75,-72)&test['pickup_latitude'].between(40,42)&test['dropoff_latitude'].between(40,42)); print('=== BROAD NYC BOUNDING BOX ==='); print('Longitude: -75 to -72'); print('Latitude : 40 to 42'); print('\nTRAIN records outside broad box:',train_out.sum()); print('TRAIN percentage:',round(train_out.mean()*100,4)); print('TEST records outside broad box:',test_out.sum()); print('TEST percentage:',round(test_out.mean()*100,4)); print('\n=== INDIVIDUAL COORDINATE VIOLATIONS ==='); print('TRAIN pickup longitude outside:',(~train['pickup_longitude'].between(-75,-72)).sum()); print('TRAIN pickup latitude outside:',(~train['pickup_latitude'].between(40,42)).sum()); print('TRAIN dropoff longitude outside:',(~train['dropoff_longitude'].between(-75,-72)).sum()); print('TRAIN dropoff latitude outside:',(~train['dropoff_latitude'].between(40,42)).sum()); print('TEST pickup longitude outside:',(~test['pickup_longitude'].between(-75,-72)).sum()); print('TEST pickup latitude outside:',(~test['pickup_latitude'].between(40,42)).sum()); print('TEST dropoff longitude outside:',(~test['dropoff_longitude'].between(-75,-72)).sum()); print('TEST dropoff latitude outside:',(~test['dropoff_latitude'].between(40,42)).sum())"
```

### Output

```text
=== BROAD NYC BOUNDING BOX ===
Longitude: -75 to -72
Latitude : 40 to 42

TRAIN records outside broad box: 38
TRAIN percentage: 0.0026
TEST records outside broad box: 19
TEST percentage: 0.003

=== INDIVIDUAL COORDINATE VIOLATIONS ===
TRAIN pickup longitude outside: 26
TRAIN pickup latitude outside: 26
TRAIN dropoff longitude outside: 30
TRAIN dropoff latitude outside: 29
TEST pickup longitude outside: 12
TEST pickup latitude outside: 12
TEST dropoff longitude outside: 12
TEST dropoff latitude outside: 15
```

### Observations

The vast majority of records fall within the broad NYC-area bounding box.

| Dataset | Records Outside Broad NYC Box | Percentage |
|---|---:|---:|
| TRAIN | 38 | 0.0026% |
| TEST | 19 | 0.0030% |

The number of affected records is extremely small compared with the overall dataset size.

### Individual Coordinate Analysis

| Coordinate | TRAIN Outside | TEST Outside |
|---|---:|---:|
| Pickup longitude | 26 | 12 |
| Pickup latitude | 26 | 12 |
| Dropoff longitude | 30 | 12 |
| Dropoff latitude | 29 | 15 |

These counts represent individual coordinate violations. They should not be added together because a single record may violate more than one coordinate condition.

### Data Quality Assessment

The analysis identifies a small number of geographically unusual records:

```text
TRAIN → 38 records
TEST  → 19 records
```

The coordinate values are mathematically valid because they fall within the global latitude/longitude ranges.

However, they are outside the broad geographic region expected for an NYC taxi trip.

Therefore:

```text
Global coordinate validity
        ↓
PASS

NYC geographic plausibility
        ↓
38 TRAIN + 19 TEST records flagged
```

### Important Distinction

These records are **flagged as geographically unusual**, but are not automatically classified as invalid.

The broad bounding box is being used to identify suspicious records for further investigation.

```text
Geographic anomaly
        ↓
Investigate
        ↓
Determine whether:
        │
        ├── Genuine unusual trip → Keep
        │
        └── Invalid location → Handle during cleaning
```

### Decision

No geographic records are removed at this stage.

The following records are flagged for further consideration during Step 4:

```text
TRAIN → 38 records
TEST  → 19 records
```

The final geographic cleaning rule will be established only after considering the nature of these records and their relationship with other fields such as:

```text
pickup location
dropoff location
pickup_datetime
dropoff_datetime
trip_duration
```

### Step 3.6 Conclusion

The dataset contains a very small number of geographically unusual records outside the broad NYC-area bounding box.

```text
TRAIN → 38 / 1,458,644 → 0.0026%
TEST  → 19 / 625,134   → 0.0030%
```

These records are flagged for investigation but are **not automatically removed**.

**Status: Step 3.6 Complete ✅**

### 3.7 – Analyze Unexpected Categorical Values

The complete raw TRAIN and TEST datasets were analyzed to identify unexpected or invalid categorical values.

The categorical features analyzed were:

```text
vendor_id
store_and_fwd_flag
```

The analysis was performed on **100% of the raw data**.

#### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['vendor_id','store_and_fwd_flag']); test=pd.read_csv('data/raw/test.csv',usecols=['vendor_id','store_and_fwd_flag']); print('=== TRAIN: vendor_id ==='); print(train['vendor_id'].value_counts(dropna=False).sort_index()); print('\nTRAIN unique values:',train['vendor_id'].unique()); print('\n=== TEST: vendor_id ==='); print(test['vendor_id'].value_counts(dropna=False).sort_index()); print('\nTEST unique values:',test['vendor_id'].unique()); print('\n=== TRAIN: store_and_fwd_flag ==='); print(train['store_and_fwd_flag'].value_counts(dropna=False).sort_index()); print('\nTRAIN unique values:',train['store_and_fwd_flag'].unique()); print('\n=== TEST: store_and_fwd_flag ==='); print(test['store_and_fwd_flag'].value_counts(dropna=False).sort_index()); print('\nTEST unique values:',test['store_and_fwd_flag'].unique()); print('\n=== UNEXPECTED CATEGORIES ==='); print('TRAIN vendor_id:',(~train['vendor_id'].isin([1,2])).sum()); print('TEST vendor_id:',(~test['vendor_id'].isin([1,2])).sum()); print('TRAIN store_and_fwd_flag:',(~train['store_and_fwd_flag'].isin(['N','Y'])).sum()); print('TEST store_and_fwd_flag:',(~test['store_and_fwd_flag'].isin(['N','Y'])).sum())"
```

#### Output

```text
=== TRAIN: vendor_id ===
vendor_id
1    678342
2    780302

TRAIN unique values: [2 1]

=== TEST: vendor_id ===
vendor_id
1    290760
2    334374

TEST unique values: [1 2]

=== TRAIN: store_and_fwd_flag ===
store_and_fwd_flag
N    1450599
Y       8045

TRAIN unique values: ['N' 'Y']

=== TEST: store_and_fwd_flag ===
N    621704
Y      3430

TEST unique values: ['N' 'Y']

=== UNEXPECTED CATEGORIES ===
TRAIN vendor_id: 0
TEST vendor_id: 0
TRAIN store_and_fwd_flag: 0
TEST store_and_fwd_flag: 0
```

### Observations

Both categorical features contain only the categories defined in the data contract.

| Feature | Allowed Values | TRAIN | TEST |
|---|---|---:|---:|
| `vendor_id` | `1, 2` | 1, 2 | 1, 2 |
| `store_and_fwd_flag` | `N, Y` | N, Y | N, Y |

No unexpected categorical values were identified.

### Category Distribution

#### `vendor_id`

| Dataset | Vendor 1 | Vendor 2 |
|---|---:|---:|
| TRAIN | 678,342 | 780,302 |
| TEST | 290,760 | 334,374 |

Both TRAIN and TEST contain the same two valid vendor categories.

#### `store_and_fwd_flag`

| Dataset | `N` | `Y` |
|---|---:|---:|
| TRAIN | 1,450,599 | 8,045 |
| TEST | 621,704 | 3,430 |

The `Y` category is relatively rare, but rarity does not make a category invalid.

```text
Rare category
     ≠
Unexpected category
```

The category is valid because it is explicitly allowed by the data contract.

### Data Quality Assessment

```text
Problem:
Unexpected categorical values

TRAIN:
0 unexpected vendor_id values
0 unexpected store_and_fwd_flag values

TEST:
0 unexpected vendor_id values
0 unexpected store_and_fwd_flag values
```

All categorical values conform to the defined contract.

### Decision

No categorical values require correction, removal, or replacement.

The following rules remain valid:

```text
vendor_id ∈ {1, 2}

store_and_fwd_flag ∈ {N, Y}
```

No additional categorical cleaning logic is required for Step 4 based on unexpected categories.

### Step 3.7 Conclusion

The complete raw TRAIN and TEST datasets contain **no unexpected categorical values**.

Both categorical features contain only the categories defined in the data contract.

**Status: Step 3.7 Complete ✅**

### 3.8 – Analyze Target Distribution

The complete raw TRAIN dataset was analyzed to understand the distribution of the target variable:

```text
trip_duration
```

The analysis focused on:

- Central tendency
- Percentile distribution
- Skewness
- Duration ranges
- Extreme tail behavior

The analysis was performed on **100% of the raw training data**.

> `trip_duration` is available only in TRAIN because it is the target variable. TEST does not contain the target.

### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['trip_duration']); d=train['trip_duration']; print('=== TARGET DISTRIBUTION ==='); print(d.describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99,.995,.999]).to_string()); print('\n=== SKEWNESS ==='); print('Skewness:',round(d.skew(),4)); print('\n=== DURATION RANGES ==='); ranges={'<= 1 min':d<=60,'1–5 min':(d>60)&(d<=300),'5–10 min':(d>300)&(d<=600),'10–15 min':(d>600)&(d<=900),'15–30 min':(d>900)&(d<=1800),'30–60 min':(d>1800)&(d<=3600),'1–2 hours':(d>3600)&(d<=7200),'2–24 hours':(d>7200)&(d<=86400),'> 24 hours':d>86400}; print(pd.DataFrame({'Count':{k:v.sum() for k,v in ranges.items()},'Percent':{k:round(v.mean()*100,4) for k,v in ranges.items()}}).to_string()); print('\n=== EXTREME PERCENTILES ==='); print('99th percentile :',d.quantile(.99)); print('99.5th percentile:',d.quantile(.995)); print('99.9th percentile:',d.quantile(.999)); print('Maximum:',d.max())"
```

### Output

```text
=== TARGET DISTRIBUTION ===
count    1.458644e+06
mean     9.594923e+02
std      5.237432e+03
min      1.000000e+00
1%       8.700000e+01
5%       1.800000e+02
10%      2.450000e+02
25%      3.970000e+02
50%      6.620000e+02
75%      1.075000e+03
90%      1.634000e+03
95%      2.104000e+03
99%      3.440000e+03
99.5%    4.139000e+03
99.9%    8.512836e+04
max      3.526282e+06

=== SKEWNESS ===
Skewness: 343.1639

=== DURATION RANGES ===
             Count  Percent
<= 1 min      8777   0.6017
1–5 min     213139  14.6121
5–10 min    430997  29.5478
10–15 min   315563  21.6340
15–30 min   377050  25.8494
30–60 min   100801   6.9106
1–2 hours    10064   0.6900
2–24 hours    2249   0.1542
> 24 hours       4   0.0003

=== EXTREME PERCENTILES ===
99th percentile : 3440.0
99.5th percentile: 4139.0
99.9th percentile: 85128.35700000008
Maximum: 3526282
```

### Observations

The target distribution is **strongly right-skewed**.

The key statistics are:

| Statistic | Value |
|---|---:|
| Mean | 959.49 sec |
| Median | 662 sec |
| 75th percentile | 1,075 sec |
| 90th percentile | 1,634 sec |
| 95th percentile | 2,104 sec |
| 99th percentile | 3,440 sec |
| 99.5th percentile | 4,139 sec |
| 99.9th percentile | 85,128 sec |
| Maximum | 3,526,282 sec |

The skewness is:

```text
343.1639
```

This indicates an extremely heavy right tail.

### Duration Distribution

Most trips are concentrated within the first hour:

| Duration Range | Records | Percentage |
|---|---:|---:|
| <= 1 min | 8,777 | 0.6017% |
| 1–5 min | 213,139 | 14.6121% |
| 5–10 min | 430,997 | 29.5478% |
| 10–15 min | 315,563 | 21.6340% |
| 15–30 min | 377,050 | 25.8494% |
| 30–60 min | 100,801 | 6.9106% |
| 1–2 hours | 10,064 | 0.6900% |
| 2–24 hours | 2,249 | 0.1542% |
| > 24 hours | 4 | 0.0003% |

Approximately **98.3% of trips are within 1 hour**, while only a very small fraction extend beyond this range.

### Important Tail Behavior

The extreme tail requires special attention.

```text
99th percentile
    ↓
3,440 sec (~57 min)

99.5th percentile
    ↓
4,139 sec (~69 min)

99.9th percentile
    ↓
85,128 sec (~23.6 hours)

Maximum
    ↓
3,526,282 sec (~40.8 days)
```

The large jump between the 99.5th and 99.9th percentiles indicates that a very small number of observations have extremely large durations.

### Data Quality Assessment

The target distribution is not symmetric and contains a substantial right tail.

However:

```text
Highly skewed
      ≠
Automatically invalid
```

The analysis does not by itself establish that the extreme observations are incorrect.

The extreme values will therefore be investigated separately in:

```text
Step 3.9 – Analyze Extremely Short Trips
Step 3.10 – Analyze Extremely Long Trips
```

### Decision

No target records are removed or transformed based solely on the distribution analysis.

The distribution characteristics will be considered later when designing the data-cleaning and ML pipeline.

In particular, the strong right skew may influence:

- Outlier handling
- Target transformation considerations
- Model selection
- Evaluation strategy

These decisions belong to later phases and will not be made during Step 3.8.

### Step 3.8 Conclusion

The `trip_duration` target is **extremely right-skewed**, with most trips concentrated between a few minutes and one hour and a very small but extreme long-duration tail.

The key finding is:

```text
Median ≈ 11 minutes
Mean   ≈ 16 minutes
Maximum ≈ 40.8 days
```

The extreme tail requires further investigation before any cleaning decision is made.

**Status: Step 3.8 Complete ✅**

### 3.9 – Analyze Extremely Short Trips

The target distribution identified a significant number of short trips. Further analysis was performed to determine whether extremely short trip durations are potentially suspicious when compared with the geographic distance between pickup and dropoff locations.

The analysis focused on trips with:

```text
trip_duration <= 60 seconds
```

For these records, the straight-line geographic distance between pickup and dropoff coordinates was calculated using the **Haversine distance**.

### Objective

The purpose of this analysis is to distinguish between:

```text
Legitimate short trip
        vs.
Potentially suspicious short trip
```

A very short trip with a very small geographic distance can be reasonable, while a very short trip covering several kilometers may require further investigation.

### Command

```powershell
python -c "import pandas as pd, numpy as np; cols=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']; train=pd.read_csv('data/raw/train.csv',usecols=cols); short=train[train['trip_duration']<=60].copy(); lon1=np.radians(short['pickup_longitude']); lat1=np.radians(short['pickup_latitude']); lon2=np.radians(short['dropoff_longitude']); lat2=np.radians(short['dropoff_latitude']); dlon=lon2-lon1; dlat=lat2-lat1; a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2; short['distance_km']=6371*2*np.arcsin(np.sqrt(a)); print('=== TRIPS <= 60 SECONDS ==='); print('Records:',len(short)); print('\n=== DISTANCE SUMMARY (KM) ==='); print(short['distance_km'].describe(percentiles=[.25,.50,.75,.90,.95,.99]).to_string()); print('\n=== ZERO / NEAR-ZERO DISTANCE ==='); print('Distance = 0 km:',(short['distance_km']==0).sum()); print('Distance < 0.01 km:',(short['distance_km']<0.01).sum()); print('Distance < 0.1 km:',(short['distance_km']<0.1).sum()); print('Distance > 1 km:',(short['distance_km']>1).sum()); print('Distance > 5 km:',(short['distance_km']>5).sum()); print('\n=== SHORTEST TRIPS WITH DISTANCE ==='); print(short.nsmallest(20,'trip_duration')[['trip_duration','distance_km']].to_string(index=False)); print('\n=== SHORT TRIPS WITH LARGEST DISTANCE ==='); print(short.nlargest(20,'distance_km')[['trip_duration','distance_km']].to_string(index=False))"
```

### Output

```text
=== TRIPS <= 60 SECONDS ===
Records: 8777

=== DISTANCE SUMMARY (KM) ===
count    8777.000000
mean        0.128828
std         0.357270
min         0.000000
25%         0.000848
50%         0.021218
75%         0.190375
90%         0.405439
95%         0.519950
99%         0.727499
max        19.948152

=== ZERO / NEAR-ZERO DISTANCE ===
Distance = 0 km: 1687
Distance < 0.01 km: 3844
Distance < 0.1 km: 5821
Distance > 1 km: 28
Distance > 5 km: 4

=== SHORTEST TRIPS WITH DISTANCE ===
 trip_duration  distance_km
             1     0.008963
             1     0.000643
             1     0.133023
             1     0.000424
             1     0.000000
             1     0.019891
             1     0.009980
             1     0.000000
             1     0.002707
             1     0.231459
             1     0.002130
             1     0.007081
             1     0.016163
             1     0.000000
             1     0.000000
             1     0.000000
             1     0.009799
             1     0.001286

=== SHORT TRIPS WITH LARGEST DISTANCE ===
 trip_duration(sec)  distance_km
            51    19.948152
             7    18.034405
            20     6.911542
            60     5.943602
            15     3.399963
            29     3.067880
            10     2.526014
            55     1.884039
            38     1.661603
            54     1.652689
            30     1.544279
            20     1.535704
            16     1.493178
            44     1.290788
             8     1.271745
            31     1.224190
            33     1.207630
            49     1.179381
            57     1.157775
            16     1.147109
```

### Observations

There are:

```text
8,777 trips
```

with a duration of **60 seconds or less**.

The geographic distance distribution for these trips is heavily concentrated near zero.

| Distance Criterion | Records |
|---|---:|
| Distance = 0 km | 1,687 |
| Distance < 0.01 km | 3,844 |
| Distance < 0.1 km | 5,821 |
| Distance > 1 km | 28 |
| Distance > 5 km | 4 |

The median distance is only:

```text
0.0212 km ≈ 21 metres
```

and the 75th percentile is:

```text
0.1904 km ≈ 190 metres
```

This indicates that most very short trips also have very small pickup-to-dropoff distances.

### Potentially Suspicious Records

A small number of very short trips have unusually large geographic distances.

Examples include:

| Trip Duration | Distance |
|---:|---:|
| 7 sec | 18.03 km |
| 51 sec | 19.95 km |
| 20 sec | 6.91 km |
| 60 sec | 5.94 km |
| 15 sec | 3.40 km |
| 29 sec | 3.07 km |

These observations are potentially suspicious because the recorded duration is extremely short relative to the geographic distance.

However, they should **not automatically be deleted** because the calculated distance is a straight-line distance rather than the actual road distance traveled.

### Data Quality Assessment

The analysis shows that extremely short trips consist primarily of records with very small geographic distances.

Therefore:

```text
Short duration
      +
Very small distance
      ↓
Potentially legitimate
```

while:

```text
Short duration
      +
Large geographic distance
      ↓
Potentially suspicious
      ↓
Requires further investigation
```

### Decision

No short-duration records are removed based solely on this analysis.

The following observations are flagged for potential further investigation:

```text
28 trips  → distance > 1 km
4 trips   → distance > 5 km
```

These records may be investigated alongside other attributes such as:

```text
pickup_datetime
dropoff_datetime
pickup location
dropoff location
trip_duration
```

before making any cleaning decision.

### Step 3.9 Conclusion

The analysis does **not support a blanket rule such as**:

```text
trip_duration <= 60 seconds → invalid
```

Most trips of 60 seconds or less have very small geographic distances and may represent legitimate short trips.

Only a very small number of records combine extremely short durations with unusually large geographic distances and should be treated as potentially suspicious.

No cleaning action is taken at this stage.

**Status: Step 3.9 Complete ✅**

### 3.10 – Analyze Extremely Long Trips

The target distribution showed a long right tail in `trip_duration`. Further analysis was performed to quantify extremely long trips and identify suspicious patterns in the tail.

The analysis focused on:

- Trips longer than 1 hour
- Trips longer than 2 hours
- Trips between 2 and 24 hours
- Trips close to 24 hours
- Trips longer than 24 hours
- The longest individual trips

The analysis was performed on **100% of the raw TRAIN dataset**.

### Command – Initial Long Trip Analysis

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['id','pickup_datetime','dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'],parse_dates=['pickup_datetime','dropoff_datetime']); d=train['trip_duration']; print('=== EXTREMELY LONG TRIPS ==='); thresholds={'> 1 hour':3600,'> 2 hours':7200,'> 3 hours':10800,'> 6 hours':21600,'> 12 hours':43200,'> 24 hours':86400,'> 48 hours':172800,'> 7 days':604800}; print(pd.DataFrame({'Count':{k:(d>v).sum() for k,v in thresholds.items()},'Percent':{k:round((d>v).mean()*100,5) for k,v in thresholds.items()}}).to_string()); print('\n=== TOP 20 LONGEST TRIPS ==='); print(train.nlargest(20,'trip_duration').to_string(index=False))"
```

### Initial Findings

```text
> 1 hour     12,317 records   0.84441%
> 2 hours     2,253 records   0.15446%
> 3 hours     2,112 records   0.14479%
> 6 hours     2,061 records   0.14130%
> 12 hours    1,993 records   0.13663%
> 24 hours        4 records   0.00027%
> 48 hours        4 records   0.00027%
> 7 days         4 records   0.00027%
```

The majority of long trips are therefore below 24 hours. However, the distribution becomes highly unusual near the 24-hour boundary.

### Command – Investigate the Near-24-Hour Cluster

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['id','pickup_datetime','dropoff_datetime','trip_duration']); d=train['trip_duration']; print('=== 2–24 HOUR TRIPS ==='); bands={'2–3 hours':(d>7200)&(d<=10800),'3–6 hours':(d>10800)&(d<=21600),'6–12 hours':(d>21600)&(d<=43200),'12–18 hours':(d>43200)&(d<=64800),'18–23 hours':(d>64800)&(d<82800),'23–24 hours':(d>=82800)&(d<=86400)}; print(pd.DataFrame({'Count':{k:v.sum() for k,v in bands.items()},'Percent':{k:round(v.mean()*100,5) for k,v in bands.items()}}).to_string()); print('\n=== NEAR-24-HOUR TRIPS ==='); near=train[(d>=86000)&(d<=86400)].sort_values('trip_duration',ascending=False); print('Count:',len(near)); print(near['trip_duration'].describe().to_string()); print('\nTop 30 near-24-hour durations:'); print(near['trip_duration'].head(30).to_string(index=False)); print('\n=== EXACTLY / NEARLY 24 HOURS ==='); print('Exactly 86400:',(d==86400).sum()); print('>=86300:',(d>=86300).sum()); print('>=86000:',(d>=86000).sum())"
```

### Output

```text
=== 2–24 HOUR TRIPS ===
             Count  Percent
2–3 hours      141  0.00967
3–6 hours       51  0.00350
6–12 hours      68  0.00466
12–18 hours     45  0.00309
18–23 hours    105  0.00720
23–24 hours   1839  0.12608

=== NEAR-24-HOUR TRIPS ===
Count: 883

count      883.000000
mean     86220.942242
std        102.055685
min      86002.000000
25%      86144.000000
50%      86242.000000
75%      86308.000000
max      86392.000000

=== EXACTLY / NEARLY 24 HOURS ===
Exactly 86400: 0
>=86300: 255
>=86000: 887
```

### Observations

A clear concentration of observations exists in the **23–24 hour range**.

| Duration Range | Records | Percentage |
|---|---:|---:|
| 2–3 hours | 141 | 0.00967% |
| 3–6 hours | 51 | 0.00350% |
| 6–12 hours | 68 | 0.00466% |
| 12–18 hours | 45 | 0.00309% |
| 18–23 hours | 105 | 0.00720% |
| **23–24 hours** | **1,839** | **0.12608%** |

The number of records increases sharply in the 23–24 hour range compared with the preceding duration bands.

### Near-24-Hour Pattern

There are:

```text
887 records with trip_duration >= 86,000 seconds
```

and:

```text
255 records with trip_duration >= 86,300 seconds
```

while:

```text
Exactly 86,400 seconds → 0 records
```

The 883 records between 86,000 and 86,400 seconds have:

```text
Minimum  → 86,002 sec
Median   → 86,242 sec
Maximum  → 86,392 sec
```

This indicates a strong concentration immediately below the 24-hour boundary.

```text
86,400 seconds = 24 hours
```

The pattern is therefore considered **highly suspicious for a taxi-trip dataset**.

### Extremely Long Trips Beyond 24 Hours

Only **4 records** have durations greater than 24 hours.

The four longest records are:

| ID | Duration (sec) | Approx. Duration |
|---|---:|---:|
| `id0053347` | 3,526,282 | ~40.8 days |
| `id1325766` | 2,227,612 | ~25.8 days |
| `id0369307` | 2,049,578 | ~23.7 days |
| `id1864733` | 1,939,736 | ~22.5 days |

These records have pickup and dropoff coordinates within the NYC region, yet their recorded durations span multiple weeks.

Such durations are **highly implausible for a normal taxi trip**.

### Relationship with Timestamp Consistency

The extremely long records are not timestamp-calculation errors.

Step 3.5 established:

```text
dropoff_datetime > pickup_datetime
```

and:

```text
trip_duration =
dropoff_datetime - pickup_datetime
```

for all training records.

Therefore, the issue is not an arithmetic mismatch between the timestamps and the target.

Instead, the concern is **domain plausibility**:

```text
Timestamp calculation
        ↓
Consistent
        ↓
But duration is unrealistic for a taxi trip
```

### Data Quality Assessment

Two distinct patterns were identified.

#### Pattern 1 – Near-24-Hour Cluster

```text
~887 records
trip_duration >= 86,000 seconds
```

with a strong concentration immediately below:

```text
86,400 seconds = 24 hours
```

This is a suspicious data pattern and requires cleaning consideration.

#### Pattern 2 – Multi-Week Trips

```text
4 records
trip_duration > 24 hours
```

with durations ranging from approximately:

```text
22.5 days → 40.8 days
```

These are highly implausible as normal NYC taxi trips.

### Decision

The analysis provides sufficient evidence to classify the following as **high-priority suspicious records**:

```text
Near-24-hour cluster:
trip_duration >= 86,000 seconds

Multi-day trips:
trip_duration > 24 hours
```

However, the final removal/handling logic will be implemented during:

```text
Step 4 – Data Cleaning
```

No records are deleted during the Data Quality Analysis stage.

The cleaning decision must remain:

```text
Evidence
   ↓
Documented rule
   ↓
Reproducible cleaning logic
```

rather than:

```text
Extreme value
   ↓
Automatically delete
```

### Step 3.10 Conclusion

The target contains a small but highly unusual long-duration tail.

The most significant finding is a strong cluster immediately below **24 hours**, along with four multi-week trips.

```text
23–24 hours → 1,839 records
>= 86,000 sec → 887 records
>= 86,300 sec → 255 records
> 24 hours → 4 records
```

The four multi-week records are particularly implausible for NYC taxi trips, while the near-24-hour cluster represents a suspicious systematic pattern that should be handled during the cleaning stage.

**Status: Step 3.10 Complete ✅**

### 3.11 – Analyze Unrealistic Passenger Counts

The complete raw TRAIN and TEST datasets were analyzed to understand the distribution of `passenger_count` and identify potentially unrealistic values.

The analysis focused on:

- Complete passenger-count distribution
- Percentage distribution
- Zero passenger records
- Passenger counts greater than 6
- Comparison between TRAIN and TEST

The analysis was performed on **100% of the raw TRAIN and TEST datasets**.

### Command

```powershell
python -c "import pandas as pd; train=pd.read_csv('data/raw/train.csv',usecols=['passenger_count']); test=pd.read_csv('data/raw/test.csv',usecols=['passenger_count']); print('=== TRAIN PASSENGER COUNT ==='); print(train['passenger_count'].value_counts(dropna=False).sort_index()); print('\n=== TRAIN PERCENTAGE ==='); print((train['passenger_count'].value_counts(normalize=True,dropna=False).sort_index()*100).round(4).to_string()); print('\n=== TEST PASSENGER COUNT ==='); print(test['passenger_count'].value_counts(dropna=False).sort_index()); print('\n=== TEST PERCENTAGE ==='); print((test['passenger_count'].value_counts(normalize=True,dropna=False).sort_index()*100).round(4).to_string()); print('\n=== PASSENGER COUNT VALIDATION ==='); print('TRAIN <= 0:',(train['passenger_count']<=0).sum()); print('TRAIN > 6:',(train['passenger_count']>6).sum()); print('TEST <= 0:',(test['passenger_count']<=0).sum()); print('TEST > 6:',(test['passenger_count']>6).sum())"
```

### Output

```text
=== TRAIN PASSENGER COUNT ===
passenger_count
0         60
1    1033540
2     210318
3      59896
4      28404
5      78088
6      48333
7          3
8          1
9          1
Name: count, dtype: int64

=== TRAIN PERCENTAGE ===
passenger_count
0     0.0041
1    70.8562
2    14.4187
3     4.1063
4     1.9473
5     5.3535
6     3.3136
7     0.0002
8     0.0001
9     0.0001

=== TEST PASSENGER COUNT ===
passenger_count
0        23
1    443447
2     90027
3     25686
4     12017
5     33411
6     20521
9         2
Name: count, dtype: int64

=== TEST PERCENTAGE ===
passenger_count
0     0.0037
1    70.9363
2    14.4012
3     4.1089
4     1.9223
5     5.3446
6     3.2827
9     0.0003

=== PASSENGER COUNT VALIDATION ===
TRAIN <= 0: 60
TRAIN > 6: 5
TEST <= 0: 23
TEST > 6: 2
```

### Observations

The passenger-count distribution is heavily concentrated around **1–6 passengers**.

For TRAIN:

```text
Passenger count = 1 → 1,033,540 records → 70.8562%
Passenger count = 2 →   210,318 records → 14.4187%
Passenger count = 3 →    59,896 records →  4.1063%
Passenger count = 4 →    28,404 records →  1.9473%
Passenger count = 5 →    78,088 records →  5.3535%
Passenger count = 6 →    48,333 records →  3.3136%
```

The same overall pattern is observed in TEST.

### Zero Passenger Records

The analysis identified:

| Dataset | `passenger_count = 0` | Percentage |
|---|---:|---:|
| TRAIN | 60 | 0.0041% |
| TEST | 23 | 0.0037% |

A passenger count of zero is inconsistent with a completed passenger taxi trip and therefore represents a clear data-quality issue.

These records were already identified during:

```text
Step 3.3 – Analyze Invalid Values
```

### Passenger Counts Greater Than 6

A very small number of records contain passenger counts greater than 6:

| Dataset | `> 6` Records | Percentage |
|---|---:|---:|
| TRAIN | 5 | ~0.0003% |
| TEST | 2 | ~0.0003% |

TRAIN contains:

```text
7 → 3 records
8 → 1 record
9 → 1 record
```

TEST contains:

```text
9 → 2 records
```

These values are extremely rare compared with the dominant `1–6` passenger range.

### Important Observation

The presence of rare values does not automatically mean that the records are invalid.

Therefore:

```text
Rare passenger count
        ≠
Automatically invalid
```

The clearly invalid condition identified from the current analysis is:

```text
passenger_count <= 0
```

The extremely rare values above 6 are flagged for consideration during the cleaning stage rather than automatically removed during quality analysis.

### Data Quality Assessment

The passenger-count analysis identified two categories of potentially problematic records:

```text
1. passenger_count <= 0
       ↓
       Clearly invalid
       ↓
       TRAIN: 60
       TEST : 23

2. passenger_count > 6
       ↓
       Extremely rare
       ↓
       TRAIN: 5
       TEST : 2
       ↓
       Requires domain-based cleaning decision
```

### Decision

No passenger records are removed during the Data Quality Analysis stage.

The following rule is confirmed:

```text
passenger_count > 0
```

Records with:

```text
passenger_count <= 0
```

will require correction/removal during **Step 4 – Data Cleaning**.

The values above 6 will be reviewed during the cleaning stage before deciding whether they should be retained, flagged, or removed.

### Step 3.11 Conclusion

Passenger counts are overwhelmingly concentrated between **1 and 6 passengers**, with approximately **71% of trips having one passenger**.

The analysis identified:

```text
TRAIN:
60 records with passenger_count = 0
5 records with passenger_count > 6

TEST:
23 records with passenger_count = 0
2 records with passenger_count > 6
```

The zero-passenger records are clearly invalid according to the data contract.

The values above 6 are extremely rare and will be handled based on a documented cleaning decision rather than being automatically removed.

**Status: Step 3.11 Complete ✅**

## 3.12 – Data Quality Report

### Objective

Consolidate the findings from **Step 3.1 through Step 3.11** into a single data-quality assessment.

The purpose of this report is to:

- Summarize the quality issues identified in the raw dataset.
- Quantify the number and percentage of affected records.
- Distinguish between valid, suspicious, and clearly invalid records.
- Define the issues that must be addressed during **Step 4 – Data Cleaning**.
- Ensure that every cleaning decision is supported by evidence.

### Analysis Scope

The analysis was performed using:

```text
TRAIN → 1,458,644 records
TEST  →   625,134 records
```

The 10% development dataset was used earlier for initial dataset validation and was confirmed to be representative of the full TRAIN dataset.

All major data-quality conclusions in this report are based on the **complete raw TRAIN/TEST datasets** where applicable.

---

## 1. Data Quality Summary

| Data Quality Area | TRAIN | TEST | Finding | Decision |
|---|---:|---:|---|---|
| Missing values | 0 | 0 | No missing values | Keep |
| Exact duplicate rows | 0 | 0 | No duplicate rows | Keep |
| Duplicate IDs | 0 | 0 | All IDs are unique | Keep |
| Invalid categorical values | 0 | 0 | All categories follow the contract | Keep |
| Invalid passenger count (`= 0`) | 60 | 23 | Clearly invalid | Handle in Step 4 |
| Negative passenger count | 0 | 0 | None observed | No action |
| Passenger count `> 6` | 5 | 2 | Extremely rare | Review in Step 4 |
| Invalid global coordinates | 0 | 0 | All coordinates within valid lat/lon ranges | Keep |
| Geographic anomalies | 38 | 19 | Outside broad NYC-area bounding box | Investigate in Step 4 |
| Impossible timestamps | 0 | N/A | No timestamp-order issues | Keep |
| Duration mismatch | 0 | N/A | Target exactly matches datetime difference | Keep |
| Trips `<= 60 sec` | 8,777 | N/A | Mostly associated with very small distances | No blanket removal |
| Near-24-hour trips (`>= 86,000 sec`) | 887 | N/A | Suspicious concentration | Investigate in Step 4 |
| Trips `> 24 hours` | 4 | N/A | Extremely implausible multi-day trips | Strong cleaning candidates |
| Target distribution | Highly right-skewed | N/A | Skewness = 343.1639 | Consider during ML |

---

## 2. Missing Values

### Finding

No missing values were identified in either dataset.

```text
TRAIN records with >=1 missing value → 0
TEST records with >=1 missing value  → 0
```

All required fields are populated.

### Decision

No missing-value treatment is required.

```text
Decision → KEEP
```

**Source:** Step 3.1

---

## 3. Duplicate Records

### Finding

No exact duplicate rows or duplicate IDs were identified.

```text
TRAIN:
Total rows     → 1,458,644
Duplicate rows → 0
Duplicate IDs  → 0

TEST:
Total rows     → 625,134
Duplicate rows → 0
Duplicate IDs  → 0
```

### Decision

No duplicate removal is required.

```text
Decision → KEEP
```

**Source:** Step 3.2

---

## 4. Invalid Values

### Finding

The categorical fields contain only valid values.

```text
vendor_id:
Allowed → {1, 2}
Invalid TRAIN → 0
Invalid TEST  → 0

store_and_fwd_flag:
Allowed → {N, Y}
Invalid TRAIN → 0
Invalid TEST  → 0
```

The only clearly invalid numeric condition identified was:

```text
passenger_count <= 0
```

with:

```text
TRAIN → 60 records
TEST  → 23 records
```

Further investigation in Step 3.11 confirmed that all of these records have:

```text
passenger_count = 0
```

There are no negative passenger counts.

### Decision

Zero-passenger records require handling during Step 4.

```text
Decision → HANDLE IN CLEANING
```

**Source:** Step 3.3 and Step 3.11

---

## 5. Statistical Outliers

The IQR method identified statistical outliers in several numerical features.

| Feature | Outliers | Outlier % |
|---|---:|---:|
| `passenger_count` | 154,830 | 10.61% |
| `pickup_longitude` | 84,322 | 5.78% |
| `pickup_latitude` | 52,743 | 3.62% |
| `dropoff_longitude` | 77,969 | 5.35% |
| `dropoff_latitude` | 71,990 | 4.94% |
| `trip_duration` | 74,220 | 5.09% |

### Important Interpretation

These are **statistical outliers**, not automatically invalid records.

For example:

```text
IQR outlier
     ≠
Invalid record
```

The IQR method identifies values that are statistically unusual relative to the distribution.

The domain-specific analysis in later steps is therefore required before removing any observations.

### Decision

No records are removed solely because they were identified as IQR outliers.

```text
Decision → DO NOT BLINDLY REMOVE
```

**Source:** Step 3.4

---

## 6. Impossible Timestamps

The complete TRAIN dataset was checked for timestamp consistency.

```text
Missing pickup_datetime       → 0
Missing dropoff_datetime      → 0
Dropoff <= Pickup             → 0
Calculated duration <= 0      → 0
Duration mismatch             → 0
```

The following relationship holds for every TRAIN record:

```text
trip_duration
=
dropoff_datetime - pickup_datetime
```

### Decision

No timestamp correction is required.

```text
Decision → KEEP
```

Extremely long durations are treated separately as a domain-quality issue rather than a timestamp-calculation issue.

**Source:** Step 3.5

---

## 7. Geographic Quality

All coordinates satisfy the global mathematical coordinate ranges:

```text
Latitude  → [-90, +90]
Longitude → [-180, +180]
```

However, a broad NYC-area bounding box was used to identify geographically unusual records:

```text
Longitude → -75 to -72
Latitude  →  40 to 42
```

The results were:

```text
TRAIN outside broad NYC box → 38 records → 0.0026%
TEST outside broad NYC box  → 19 records → 0.0030%
```

### Important Interpretation

These records are geographically unusual but are not automatically invalid.

```text
Global coordinate validity
        ↓
PASS

NYC geographic plausibility
        ↓
38 TRAIN + 19 TEST flagged
```

### Decision

Flag these records for further consideration during cleaning.

```text
Decision → INVESTIGATE IN STEP 4
```

No automatic deletion is performed at the Data Quality Analysis stage.

**Source:** Step 3.6

---

## 8. Unexpected Categorical Values

The categorical distributions were validated against the data contract.

### `vendor_id`

Allowed values:

```text
{1, 2}
```

Observed:

```text
TRAIN → {1, 2}
TEST  → {1, 2}
```

Unexpected values:

```text
TRAIN → 0
TEST  → 0
```

### `store_and_fwd_flag`

Allowed values:

```text
{N, Y}
```

Observed:

```text
TRAIN → {N, Y}
TEST  → {N, Y}
```

Unexpected values:

```text
TRAIN → 0
TEST  → 0
```

### Decision

No categorical cleaning is required.

```text
Decision → KEEP
```

**Source:** Step 3.7

---

## 9. Target Distribution

The target variable is:

```text
trip_duration
```

The distribution is extremely right-skewed.

Key statistics:

| Statistic | Value |
|---|---:|
| Mean | 959.49 sec |
| Median | 662 sec |
| 75th percentile | 1,075 sec |
| 90th percentile | 1,634 sec |
| 95th percentile | 2,104 sec |
| 99th percentile | 3,440 sec |
| 99.5th percentile | 4,139 sec |
| 99.9th percentile | 85,128 sec |
| Maximum | 3,526,282 sec |

Skewness:

```text
343.1639
```

### Interpretation

The mean is substantially higher than the median:

```text
Mean   → 959 sec
Median → 662 sec
```

This confirms a strong right tail.

Approximately:

```text
98%+ of trips are within 1 hour
```

while a very small number of records have extremely large durations.

### Decision

No target values are removed solely because of skewness.

The extreme tail is handled through the specific short-trip and long-trip investigations.

```text
Decision → INVESTIGATE EXTREMES
```

**Source:** Step 3.8

---

## 10. Extremely Short Trips

Trips with:

```text
trip_duration <= 60 seconds
```

were investigated using geographic distance.

Number of records:

```text
8,777
```

Their distance distribution showed:

```text
Distance = 0 km       → 1,687
Distance < 0.01 km    → 3,844
Distance < 0.1 km     → 5,821
Distance > 1 km       → 28
Distance > 5 km       → 4
```

The median geographic distance was approximately:

```text
0.021 km ≈ 21 metres
```

### Interpretation

Most very short trips also have very small geographic distances.

Therefore, a blanket rule such as:

```text
trip_duration <= 60 seconds
        ↓
INVALID
```

is not justified.

A small number of records combine very short durations with unusually large distances and may require further investigation.

### Decision

Do not automatically remove all short trips.

```text
Decision → KEEP
           +
           INVESTIGATE EXTREME COMBINATIONS IF REQUIRED
```

**Source:** Step 3.9

---

## 11. Extremely Long Trips

The long-duration tail contains several distinct patterns.

### Trips longer than 1 hour

```text
12,317 records
0.84441%
```

### Trips longer than 2 hours

```text
2,253 records
0.15446%
```

### Near-24-hour cluster

A significant concentration was identified between:

```text
23–24 hours
```

with:

```text
1,839 records
0.12608%
```

Further analysis showed:

```text
trip_duration >= 86,000 sec → 887 records
trip_duration >= 86,300 sec → 255 records
trip_duration = 86,400 sec   → 0 records
```

Since:

```text
86,400 seconds = 24 hours
```

the concentration immediately below 24 hours is highly suspicious.

### Multi-week trips

Only four records exceed 24 hours.

Their durations are approximately:

```text
22.5 days
23.7 days
25.8 days
40.8 days
```

These are highly implausible for normal NYC taxi trips.

### Decision

The near-24-hour cluster and the four multi-day records require explicit handling during Step 4.

```text
Decision → INVESTIGATE / HANDLE IN CLEANING
```

The four multi-week records are considered **strong candidates for removal**, subject to the final documented cleaning rule.

**Source:** Step 3.10

---

## 12. Unrealistic Passenger Counts

The passenger-count distribution is heavily concentrated between:

```text
1–6 passengers
```

### Zero passengers

```text
TRAIN → 60 records → 0.0041%
TEST  → 23 records → 0.0037%
```

All identified invalid records have:

```text
passenger_count = 0
```

No negative passenger counts were observed.

### Passenger counts > 6

```text
TRAIN → 5 records
TEST  → 2 records
```

These values are extremely rare.

TRAIN contains:

```text
7 → 3 records
8 → 1 record
9 → 1 record
```

TEST contains:

```text
9 → 2 records
```

### Interpretation

The zero-passenger records are clearly invalid.

The values greater than 6 are extremely rare but require a domain-based decision before removal.

### Decision

```text
passenger_count = 0
        ↓
Handle during Step 4

passenger_count > 6
        ↓
Review during Step 4
```

**Source:** Step 3.11

---

## 13. Final Data Quality Decision Matrix

The following matrix converts our findings into actionable categories.

| Issue | Severity | Action |
|---|---|---|
| Missing values | None | No action |
| Duplicate rows | None | No action |
| Duplicate IDs | None | No action |
| Invalid categorical values | None | No action |
| Negative passenger counts | None | No action |
| Zero passenger counts | Clear invalid | Handle in Step 4 |
| Passenger count > 6 | Extremely rare | Review in Step 4 |
| Global coordinate violations | None | No action |
| Geographic anomalies | Very small | Investigate in Step 4 |
| Timestamp ordering issues | None | No action |
| Duration mismatches | None | No action |
| Statistical outliers | Present | Do not blindly remove |
| Very short trips | Present | No blanket removal |
| Near-24-hour duration cluster | Suspicious | Handle in Step 4 |
| Multi-day durations | Highly suspicious | Strong cleaning candidates |
| Target skewness | Extreme | Consider during ML preparation |

---

## 14. Key Findings

The raw dataset is generally high quality with respect to:

```text
Missing values
Duplicates
IDs
Categorical consistency
Timestamp consistency
Global coordinate validity
Target calculation consistency
```

The major data-quality concerns are concentrated in a very small number of records:

```text
1. Zero passenger counts
2. Geographically unusual records
3. Extremely short trips with unusually large distances
4. Near-24-hour trip-duration cluster
5. Four multi-week trip durations
6. Extremely rare passenger counts above 6
```

The target variable also has an extremely heavy right tail:

```text
Skewness = 343.1639
```

---

## 15. Cleaning Principles for Step 4

The findings from this report will be converted into explicit, deterministic cleaning rules during:

```text
Step 4 – Data Cleaning
```

The following principles will be followed:

```text
Evidence
   ↓
Explicit Rule
   ↓
Reproducible Code
   ↓
Validation
   ↓
Clean Dataset
```

We will **not** use arbitrary rules such as:

```text
"Remove all outliers"
"Remove all trips > 1 hour"
"Remove all trips < 1 minute"
```

Instead, each cleaning rule must have:

```text
Problem
   ↓
Affected Records
   ↓
Reason
   ↓
Defined Rule
   ↓
Cleaning Action
```

---

## 16. Phase 3B Data Quality Analysis – Conclusion

The complete raw dataset has now been systematically analyzed across:

```text
Missing Values
Duplicates
Invalid Values
Statistical Outliers
Timestamps
Geographical Coordinates
Categorical Values
Target Distribution
Short Trips
Long Trips
Passenger Counts
```

The analysis confirms that the majority of the dataset is internally consistent and usable.

The identified quality issues are small in number relative to the overall dataset, with the most significant concern being the **extreme right tail of `trip_duration`**, particularly the suspicious near-24-hour cluster and four multi-week records.

No cleaning was performed during the Data Quality Analysis stage.

The findings documented here will serve as the evidence base for:

```text
Phase 3B
     ↓
Step 4 – Data Cleaning
```

### Phase 3B Data Quality Analysis Status

```text
Step 3.1  → Complete ✅
Step 3.2  → Complete ✅
Step 3.3  → Complete ✅
Step 3.4  → Complete ✅
Step 3.5  → Complete ✅
Step 3.6  → Complete ✅
Step 3.7  → Complete ✅
Step 3.8  → Complete ✅
Step 3.9  → Complete ✅
Step 3.10 → Complete ✅
Step 3.11 → Complete ✅
Step 3.12 → Complete ✅
```

## Visual Data Quality Analysis

To complement the numerical data-quality analysis, a focused visualization notebook was created using the complete raw TRAIN dataset.

The notebook contains five visualizations:

1. Trip Duration Distribution
2. Trip Duration Box Plot
3. Passenger Count Distribution
4. Geographic Scatter Plot
5. Numerical Feature Correlation Heatmap

These visualizations provide a visual baseline of the raw dataset and support the findings documented in Steps 3.4, 3.6, 3.8, 3.9, 3.10, and 3.11.

No data cleaning or transformation is performed in the visualization notebook.

**Visualization Notebook:**  
[Phase 3B – Data Quality Visualization](../notebooks/Phase_3B_data_quality_visualization.ipynb)

### Key Visual Insights

- `trip_duration` is highly right-skewed with a long extreme tail.
- Statistical outliers are clearly visible in the trip-duration box plot.
- Passenger counts are heavily concentrated around 1–6 passengers, with `0` and values above `6` being extremely rare.
- Pickup locations are strongly concentrated around the NYC area, with a small number of geographically unusual observations.
- Raw latitude/longitude features have very weak linear correlation with `trip_duration`, highlighting the future need for meaningful geographic feature engineering.

The notebook serves as the **visual evidence layer** supporting the Data Quality Report.


**Phase 3B – Step 3 Data Quality Analysis: COMPLETE ✅**

---

# Step 4 – Data Cleaning

### 4.1 – Handle Missing Values

### Objective

Ensure that required fields contain no missing values before proceeding with further data cleaning.

### Validation

The raw TRAIN and TEST datasets were checked for missing values across all columns.

| Dataset | Total Missing Values | Records with Missing Values | Decision |
|---|---:|---:|---|
| TRAIN | 0 | 0 | No action required |
| TEST | 0 | 0 | No action required |

### Cleaning Decision

No imputation, replacement, or row removal is required because neither dataset contains missing values.

The cleaning pipeline will retain all records unchanged with respect to missing-value handling.

### Conclusion

**Step 4.1 – PASS**

Both TRAIN and TEST datasets contain complete values across all available fields.

### 4.2 – Handle Duplicates

### Objective

Ensure that the dataset does not contain duplicate records or duplicate trip IDs.

### Validation

Both exact duplicate rows and duplicate `id` values were checked in the TRAIN and TEST datasets.

| Dataset | Duplicate Rows | Duplicate IDs | Decision |
|---|---:|---:|---|
| TRAIN | 0 | 0 | No action required |
| TEST | 0 | 0 | No action required |

### Cleaning Decision

No records will be removed because neither dataset contains exact duplicate rows or duplicate IDs.

The `id` field is unique across both datasets.

### Conclusion

**Step 4.2 – PASS**

Both TRAIN and TEST datasets contain unique records and unique trip IDs. No duplicate-handling operation is required.

### 4.3 – Handle Invalid Records

### Objective

Identify and handle records that violate the data contract and cannot represent a valid taxi trip.

Based on the Data Quality Analysis, the clearly invalid condition identified at this stage is:

```text
passenger_count <= 0
```

Further analysis confirmed that all affected records have:

```text
passenger_count = 0
```

No negative passenger counts were observed.

### Validation Before Cleaning

#### TRAIN

```text
Total records       : 1,458,644
Invalid records     : 60
Percentage affected : 0.0041%
Invalid value       : passenger_count = 0
```

#### TEST

```text
Total records       : 625,134
Invalid records     : 23
Percentage affected : 0.0037%
Invalid value       : passenger_count = 0
```

The affected records represent a very small fraction of the respective datasets.

### Cleaning Rule

The data contract defines the following requirement:

```text
passenger_count > 0
```

Therefore:

```text
passenger_count = 0
        ↓
Invalid taxi-trip record
        ↓
Remove during cleaning
```

### Cleaning Decision

The following records will be removed from the cleaned datasets:

| Dataset | Invalid Records | Percentage | Action |
|---|---:|---:|---|
| TRAIN | 60 | 0.0041% | Remove |
| TEST | 23 | 0.0037% | Remove |

No other passenger-count values are removed in this step.

The rare values above 6 are **not handled here**, because they require separate analysis and are covered under:

```text
Step 4.6 – Handle Invalid Passenger Counts
```

### Important Principle

The original raw datasets will **not** be modified directly.

The cleaning process will read the data from:

```text
data/raw/
```

and generate cleaned/intermediate data separately.

This preserves the original dataset and makes the cleaning process reproducible.

```text
Raw Dataset
     ↓
Apply validation rule
     ↓
Identify passenger_count = 0
     ↓
Remove invalid records
     ↓
Clean Dataset
```

### Expected Result

After applying this rule:

#### TRAIN

```text
1,458,644
    ↓
Remove 60 invalid records
    ↓
1,458,584 records
```

#### TEST

```text
625,134
    ↓
Remove 23 invalid records
    ↓
625,111 records
```

These counts will be verified after the cleaning operation is implemented.

### Conclusion

**Step 4.3 – Invalid Records: COMPLETE ✅**

The invalid-record condition has been identified, quantified, and converted into an explicit cleaning rule.

```text
Rule:
passenger_count > 0

Records to remove:
TRAIN → 60
TEST  → 23
```

The actual removal will be implemented through the reusable cleaning pipeline rather than by manually modifying the raw datasets.

### 4.4 – Handle Impossible Timestamps

### Objective

Ensure that pickup and dropoff timestamps represent a valid chronological taxi trip and that the recorded `trip_duration` is consistent with the timestamps.

---

### Validation Before Cleaning

The following conditions were checked in the TRAIN dataset:

```text
pickup_datetime < dropoff_datetime

trip_duration > 0

trip_duration =
dropoff_datetime - pickup_datetime
```

### Validation

The cleaning-stage validation produced:

```text
=== TRAIN TIMESTAMP CLEANING CHECK ===
Invalid datetime order: 0
Non-positive calculated duration: 0
Duration mismatch: 0
```

### Validation Summary

| Validation Rule | Invalid Records | Decision |
|---|---:|---|
| `pickup_datetime < dropoff_datetime` | 0 | No action required |
| Calculated duration `> 0` | 0 | No action required |
| `trip_duration` matches datetime difference | 0 | No action required |

### Cleaning Rule

A timestamp record will be considered invalid if:

```text
pickup_datetime >= dropoff_datetime
```

or if:

```text
calculated_duration <= 0
```

or if:

```text
trip_duration != calculated_duration
```

Such records would be flagged and removed during cleaning because they cannot represent a valid trip duration.

### Cleaning Decision

No records will be removed for timestamp-related issues.

The TRAIN dataset contains:

```text
Invalid datetime order      → 0
Non-positive duration       → 0
Duration mismatch           → 0
```

Therefore, the timestamp information can be retained without modification.

### Important Principle

The timestamp fields will not be altered merely to make records pass validation.

```text
Raw timestamps
      ↓
Validate chronological order
      ↓
Validate calculated duration
      ↓
Validate recorded duration
      ↓
No violations found
      ↓
Retain timestamps unchanged
```

### Conclusion

**Step 4.4 – Impossible Timestamps: PASS**

No impossible timestamp relationships or duration inconsistencies were found in the TRAIN dataset.

No timestamp-related cleaning is required.

### Current Step 4 Status

```text
4.1 Missing Values        → PASS → No action
4.2 Duplicates            → PASS → No action
4.3 Invalid Records       → 83 records identified for removal
4.4 Impossible Timestamps → PASS → No action
```

### 4.5 – Handle Invalid Coordinates

### Objective

Identify and handle geographic coordinates that are clearly inconsistent with the expected NYC taxi operating region.

### Validation

Using the broad NYC bounding box:

```text
Longitude: -75 to -72
Latitude : 40 to 42
```

we identified:

| Dataset | Records Outside Region | Percentage |
|---|---:|---:|
| TRAIN | 38 | 0.0026% |
| TEST | 19 | 0.0030% |

Inspection of the affected records shows that several coordinates are far outside the expected NYC region, including locations around California, Virginia/DC, Massachusetts, and other distant areas.

These are therefore treated as geographic anomalies for this NYC taxi dataset.

### Cleaning Rule

A record is considered geographically invalid when its pickup or dropoff coordinates fall outside the expected NYC operating region:

```text
-75 <= longitude <= -72
 40 <= latitude  <= 42
```

### Cleaning Decision

Remove records containing pickup or dropoff coordinates outside the defined geographic boundary.

```text
TRAIN → Remove 38 records
TEST  → Remove 19 records
```

The original raw datasets will remain unchanged. The coordinate filtering will be applied only during the cleaning pipeline.

### Conclusion

**Step 4.5 – Invalid Coordinates: COMPLETE**

A small number of geographically anomalous records were identified and will be removed based on the defined NYC geographic boundary.

### One Thing to Keep in Mind

We're deliberately using the **broad bounding box** rather than a very tight NYC polygon. This is a conservative cleaning rule:

```text
Broad NYC boundary
       ↓
Remove only clearly distant observations
       ↓
Preserve legitimate NYC-area trips
```

That is a better approach for our project than aggressively removing every coordinate that looks unusual.

### 4.6 – Handle Invalid Passenger Counts

### Objective

Review unusually high passenger counts and determine whether they should be removed from the dataset.

### Validation

After identifying the invalid `passenger_count = 0` records in Step 4.3, the remaining unusual values were inspected.

| Dataset | Values > 6 | Records |
|---|---|---:|
| TRAIN | 7, 8, 9 | 5 |
| TEST | 9 | 2 |

The affected records were inspected individually and showed valid timestamps, geographic coordinates, and positive trip durations. No clear evidence of data corruption was found.

### Cleaning Decision

Values greater than 6 will **not** be removed solely because they are rare.

```text
passenger_count = 0
        ↓
Invalid → Remove

passenger_count = 7, 8, 9
        ↓
Rare but not proven invalid → Retain
```

### 4.7 – Handle Invalid Target Values

### Objective

Ensure that the target variable `trip_duration` contains valid positive values and is consistent with the pickup and dropoff timestamps.

### Validation

The TRAIN dataset was checked for:

```text
Missing trip_duration        → 0
Non-positive trip_duration   → 0
Duration mismatch            → 0
```

### Cleaning Rule

The target must satisfy:

```
trip_duration > 0
```

and:

```
trip_duration =
dropoff_datetime - pickup_datetime
```

### Cleaning Decision

No target values will be removed or modified because all TRAIN records satisfy the defined validation rules.

Long-duration trips are **not removed in this step**. They will be evaluated separately under:

```
Step 4.8 – Handle Justified Outliers
```

### Conclusion

**Step 4.7 – Invalid Target Values: PASS**

The target variable is complete, positive, and consistent with the recorded timestamps. No target-value cleaning is required.

### Current Step 4 status

```text
4.1 Missing Values          → PASS → No action
4.2 Duplicates              → PASS → No action
4.3 Invalid Records         → 60 TRAIN + 23 TEST identified
4.4 Impossible Timestamps   → PASS → No action
4.5 Invalid Coordinates     → 38 TRAIN + 19 TEST identified
4.6 Passenger Counts        → >6 retained
4.7 Invalid Target Values   → PASS → No action
```

### 4.8 – Handle Justified Outliers

### Objective

Identify extreme `trip_duration` values that are clearly unrealistic for a NYC taxi trip and should be removed.

### Validation

The analysis identified:

```text
IQR outliers                  → 74,220 records (5.09%)
Trips > 24 hours              → 4 records
Trips >= 86,000 seconds       → 887 records
Maximum duration              → 3,526,282 seconds (~40.8 days)
```

The four trips exceeding 24 hours have durations ranging from approximately 22 to 41 days, which is not realistic for a taxi trip.

A separate cluster of 887 trips occurs very close to 24 hours. These records are suspicious but are not removed at this stage without stronger evidence.

### Cleaning Decision

```
Trip duration > 24 hours
        ↓
Clearly unrealistic
        ↓
Remove
```

Therefore:

```
TRAIN → Remove 4 records
```

The near-24-hour observations (`>= 86,000 seconds`) will be retained for now and documented as a potential anomaly.

### Important Principle

Statistical outlier status alone is not sufficient reason for removal.

```
IQR outlier
    ≠
Invalid record
```

Only clearly unrealistic observations supported by domain reasoning are removed.

### Conclusion

**Step 4.8 – Justified Outliers: COMPLETE**

- 74,220 statistical outliers identified.
- 4 trips exceeding 24 hours classified as clearly unrealistic and marked for removal.
- Near-24-hour cluster retained pending further evidence.

### 4.9 – Ensure Consistent Data Types

### Objective

Ensure that all dataset columns use explicit and consistent data types before the cleaned dataset is generated.

### Expected Data Types

| Column | Expected Type |
|---|---|
| `id` | string |
| `vendor_id` | integer |
| `pickup_datetime` | datetime |
| `dropoff_datetime` | datetime |
| `passenger_count` | integer |
| `pickup_longitude` | float |
| `pickup_latitude` | float |
| `dropoff_longitude` | float |
| `dropoff_latitude` | float |
| `store_and_fwd_flag` | string |
| `trip_duration` | integer |

### Validation Result

The TRAIN dataset was loaded with the expected explicit data types:

```text
id                    → string
vendor_id             → Int64
pickup_datetime       → datetime64[ns]
dropoff_datetime      → datetime64[ns]
passenger_count       → Int64
pickup_longitude      → float64
pickup_latitude       → float64
dropoff_longitude     → float64
dropoff_latitude      → float64
store_and_fwd_flag    → string
trip_duration         → Int64
```

### Cleaning Decision

The cleaning pipeline will explicitly enforce these data types when loading the raw dataset.

No information is changed; only the representation of each field is standardized.

### Conclusion

Step 4.9 – Data Types: PASS

All required fields have defined and consistent data types suitable for the downstream cleaning and feature-engineering pipeline.


### Step 4 status

We now have:

```text
4.1 Missing Values        → PASS → No action
4.2 Duplicates            → PASS → No action
4.3 Invalid Records       → Remove 60 TRAIN + 23 TEST
4.4 Timestamps            → PASS → No action
4.5 Coordinates           → Remove 38 TRAIN + 19 TEST
4.6 Passenger Counts      → >6 retained
4.7 Target Values         → PASS → No action
4.8 Outliers              → Remove 4 TRAIN records >24h
4.9 Data Types            → PASS → Enforce types
```

### 4.10 – Verify Cleaned Dataset

The cleaning pipeline was executed successfully (using [scripts/clean.py](../scripts/clean.py)) and the final datasets were verified against the findings from Steps 4.1–4.8.

### TRAIN Verification

- Original records: **1,458,644**
- Missing values removed: **0**
- Duplicate rows/IDs removed: **0**
- Invalid passenger counts removed: **60**
- Invalid coordinate records removed: **38**
- Invalid timestamp records removed: **0**
- Invalid target records removed: **0**
- Extreme outliers (>24 hours) removed: **4**
- **Final records: 1,458,542**
- **Total removed: 102**

Record count verification:

```
1,458,644 - (60 + 38 + 4) = 1,458,542
```

### TEST Verification

- Original records: **625,134**
- Missing values removed: **0**
- Duplicate rows/IDs removed: **0**
- Invalid passenger counts removed: **23**
- Invalid coordinate records removed: **19**
- Target/timestamp-specific cleaning: **Not applicable**
- **Final records: 625,092**
- **Total removed: 42**

Record count verification:

```
625,134 - (23 + 19) = 625,092
```

### Final Validation

The cleaning results match the anomalies identified during the earlier data-quality analysis.

The cleaned datasets are stored in:

```
data/interim/train_clean.csv
data/interim/test_clean.csv
```

The raw datasets under `data/raw/` remain unchanged.

## **Conclusion:** Phase 3B Step 4 – Data Cleaning has been successfully completed and validated.