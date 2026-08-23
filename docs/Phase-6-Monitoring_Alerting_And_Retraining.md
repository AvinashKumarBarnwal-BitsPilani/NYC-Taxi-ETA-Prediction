# Phase 6 — Monitoring, Alerting & Retraining

## Table of Contents

- [Phase 6.1 — Prediction Logging](#phase-61)
- [Phase 6.2 — Performance Monitoring](#phase-62)
- [Phase 6.3 — Drift Detection](#phase-63)
- [Phase 6.4 — Simulate Realistic Drift](#phase-64)
- [Phase 6.5 — Alerting](#phase-65)
- [Phase 6.6 — Retraining Trigger](#phase-66)
- [Phase 6.7 — Retraining Pipeline](#phase-67)
- [Phase 6.8 — End-to-End Monitoring Demo](#phase-68)

---

## Objective

Phase 6 introduces the model monitoring and retraining foundation required to operate the NYC Taxi ETA prediction model beyond initial deployment.

The phase focuses on:

- Capturing prediction activity from the deployed API.
- Measuring model performance when ground truth becomes available.
- Establishing a baseline for future performance comparison.
- Detecting data/model degradation in later steps.
- Supporting automated retraining decisions.

The implementation is intentionally lightweight and focused on demonstrating the end-to-end monitoring and retraining workflow.

---

<a id="phase-61"></a>
## Phase 6.1 — Prediction Logging

### Objective

Capture every prediction request made through the deployed `/predict` API so that prediction activity can be monitored over time.

### Implementation

A prediction logger was integrated into the inference flow.

The logging functionality is implemented in `src/monitoring/prediction_logger.py`, which captures and persists successful prediction requests from the deployed API.

For each successful prediction, the following information is persisted:

```text
timestamp
vendor_id
passenger_count
store_and_fwd_flag
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
predicted_eta
```

Predictions are stored in:
```text
data/monitoring/prediction_logs.csv
```

### Validation

The following were verified:

- Prediction logger implemented.
- `/predict` API integrated with logging.
- Prediction records persisted to CSV.
- Existing CSV records are preserved.
- Subsequent predictions are appended without duplicating the header.


**Result**

**Phase 6.1 Prediction Logging: ✅ COMPLETED**

---
<a id="phase-62"></a>
## Phase 6.2 — Performance Monitoring

**Objective**
Measure the performance of the deployed model by comparing predicted ETA with the corresponding ground-truth trip duration.

Performance monitoring uses:
- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — R-squared

MAPE is not used in this project.

---

### 6.2.1 — Ground Truth Source

For the controlled monitoring evaluation, the validation dataset is used as the ground-truth source:

data/split/X_val.csv
data/split/y_val.csv


Validation data contains:
```text
X_val → 291,709 records × 8 features
y_val → 291,709 records × 1 target
```

The target column is:

trip_duration


Each validation feature record is paired with its corresponding ground-truth target.

**Status: 6.2.1 – COMPLETED ✅**

---

### 6.2.2 — Performance Monitoring Dataset

The persisted final model and preprocessor are used to generate predictions for the validation records.

```text
X_val
  ↓
Final Preprocessor
  ↓
Final XGBoost Model
  ↓
predicted_eta
```

The corresponding `trip_duration` value provides:

actual_eta


The resulting monitoring dataset contains:
```text
vendor_id
passenger_count
store_and_fwd_flag
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
predicted_eta
actual_eta
```

It is persisted as:

data/monitoring/performance_dataset.csv


This dataset provides a controlled prediction-vs-ground-truth evaluation window for monitoring.

**Status: 6.2.2 – COMPLETED ✅**

---

### 6.2.3 — Performance Metrics

The following metrics are calculated:

**MAE**
Measures the average absolute difference between predicted and actual ETA.

**RMSE**
Measures prediction error while giving greater weight to larger errors.

**R²**
Measures how well the model explains the variation in the actual trip duration.

**Current Baseline Metrics**

| Metric | Baseline |
|---|---|
| MAE | 492.22 sec |
| RMSE | 3223.65 sec |
| R² | 0.0208 |

**Status: 6.2.3 – COMPLETED ✅**

---

### 6.2.4 — Performance Report

The current performance evaluation is persisted to:

data/monitoring/performance_report.json


The report contains:
- Evaluation dataset information
- Number of evaluated records
- MAE
- RMSE
- R²

The report represents the current performance evaluation and can be regenerated for a new monitoring window.

**Status: 6.2.4 – COMPLETED ✅**

---

### 6.2.5 — Baseline Metrics

The initial validation performance is established as the model's baseline:

data/monitoring/baseline_metrics.json


The baseline contains:
- MAE
- RMSE
- R²

The baseline is created once and preserved for future comparison.

```text
Current Performance
        │
        ▼
performance_report.json
        │
        │ compare
        ▼
baseline_metrics.json
        │
        ▼
Performance Degradation
```

**Status: 6.2.5 – COMPLETED ✅**

---

### 6.2.6 — Batch Prediction Support

The existing `PredictionPipeline.predict()` method continues to support the deployed API's single-record prediction contract.

A separate `predict_batch()` method was added for offline monitoring/evaluation of multiple validation records.

This prevents batch evaluation from changing the behavior of the production `/predict` API.

```text
Online API
    ↓
predict()
    ↓
Single prediction


Offline Monitoring
    ↓
predict_batch()
    ↓
Batch predictions
```

**Status: 6.2.6 – COMPLETED ✅**

---

### 6.2.7 — Validation & Tests

Dedicated performance-monitoring tests were added in `tests/test_performance_monitoring.py` covering:
- MAE, RMSE and R² calculation.
- Missing required columns.
- Missing prediction/ground-truth values.
- Empty monitoring datasets.
- Baseline creation and persistence.

**Test Result**
```text
5 passed
```

Existing inference-related tests were also executed after introducing batch prediction support:
```text
test_end_to_end_pipeline.py       → PASSED
test_preprocessor_persistence.py  → PASSED
```

The single-record `/predict` API was additionally verified manually after the changes.

**Result**

**Phase 6.2 — Performance Monitoring: ✅ COMPLETED**

---

## Phase 6 Progress

```text
Phase 6 — Monitoring & Retraining

├── Phase 6.1 — Prediction Logging
│   ├── Logger implementation       ✅
│   ├── API integration             ✅
│   ├── CSV persistence             ✅
│   └── Append behavior             ✅
│
├── Phase 6.2 — Performance Monitoring
│   ├── Ground truth source         ✅
│   ├── Monitoring dataset          ✅
│   ├── MAE / RMSE / R²             ✅
│   ├── Performance report          ✅
│   ├── Baseline                    ✅
│   ├── Batch prediction support    ✅
│   └── Tests                       ✅
│
└── Phase 6.3 — Drift Detection        ⏭️ NEXT
```

<a id="phase-63"></a>
## Phase 6.3 — Drift Detection

### Objective

Detect changes in the distribution of model input features between the training data and current production data.

The implementation uses:

- **PSI (Population Stability Index)** for all monitored features.
- **KS Test (Kolmogorov–Smirnov)** for the continuous `distance_km` feature.
- Configurable thresholds to classify significant drift.
- A machine-readable drift report for downstream monitoring and alerting.

---

### 6.3.1 — Define Drift Baseline

The training feature dataset is used as the reference distribution:

```text
data/split/X_train.csv
```

The baseline contains 1,166,833 records × 8 features.

Baseline distributions are created for all monitored features:
```text
vendor_id
passenger_count
store_and_fwd_flag
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
distance_km
```

Discrete/categorical features store value proportions, while `distance_km` uses quantile-based bins.

The baseline is persisted as:
```text
data/monitoring/drift_baseline.json
```

**Status: 6.3.1 – COMPLETED ✅**

---

### 6.3.2 — Prepare Current Production Data

Current production requests are sourced from the prediction logs generated by Phase 6.1: `data/monitoring/prediction_logs.csv`


Only the 8 monitored input features are extracted.

`timestamp` and `predicted_eta` are excluded because this step focuses on input feature drift.

The prepared production dataset is persisted as: `data/monitoring/current_production_data.csv`


The dataset is used as the current distribution for drift comparison.

**Status: 6.3.2 – COMPLETED ✅**

---

### 6.3.3 — Implement PSI

**Population Stability Index (PSI)** is used to measure how much the current production feature distribution differs from the training baseline.

```text
X_train
   ↓
Baseline Distribution
   │
   │ compare
   ▼
Current Production Distribution
   ↓
PSI
```

PSI is calculated for all 8 monitored features.

The implementation also handles categories that are present in one distribution but not the other.

**Status: 6.3.3 – COMPLETED ✅**

---

### 6.3.4 — KS Test

The two-sample Kolmogorov–Smirnov test is additionally applied to: `distance_km`


The KS test produces:
- KS statistic
- p-value

This provides an additional statistical comparison for the continuous distance feature.

**Status: 6.3.4 – COMPLETED ✅**

---

### 6.3.5 — Establish Drift Thresholds

The following thresholds are used:

| Metric | Threshold | Drift Condition |
|---|---|---|
| PSI | 0.20 | PSI ≥ 0.20 |
| KS p-value | 0.05 | p-value < 0.05 |

The thresholds are defined as configurable constants in: `src/monitoring/drift_detection.py`


Threshold interpretation is kept separate from the statistical calculations so that downstream alerting can use the resulting drift status.

**Status: 6.3.5 – COMPLETED ✅**

---

### 6.3.6 — Generate Drift Report

PSI and KS results are combined into a drift report: `data/monitoring/drift_report.json`

The report contains:
- Baseline and current data sources.
- Current evaluation record count.
- PSI value and drift status for each feature.
- KS statistic and p-value for `distance_km`.
- Configured thresholds.
- Overall drift status.

This report provides the output required by later monitoring and alerting steps.

**Status: 6.3.6 – COMPLETED ✅**

---

### 6.3.7 — Drift Detection Tests

Dedicated tests were added in: `tests/test_drift_detection.py`

The tests cover:
- PSI for identical distributions.
- PSI detection for changed distributions.
- KS test result validation.
- PSI threshold classification.
- KS p-value threshold classification.

**Test Result**
```text
5 passed
```

Performance monitoring tests were also retained and executed together:
```text
Performance Monitoring → 5 passed
Drift Detection        → 5 passed
Total                   → 10 passed
```

**Status: 6.3.7 – COMPLETED ✅**

---

### 6.3.8 — End-to-End Validation

The complete drift detection workflow was validated:

```text
X_train.csv
    ↓
Drift Baseline
    ↓
prediction_logs.csv
    ↓
Current Production Data
    ↓
PSI + KS Test
    ↓
Threshold Classification
    ↓
drift_report.json
```

The complete project test suite was also executed using:
```bash
python -m pytest -v
```

The generated drift artifacts were verified successfully.

The current production dataset contains only a small number of manually generated requests and is therefore used only to validate the implementation. Realistic drift simulation is handled separately in Phase 6.4.

**Status: 6.3.8 – COMPLETED ✅**

---

### Phase 6.3 Files Introduced/Modified

For quick reference, the important implementation changes from this phase are:

```text
src/monitoring/drift_detection.py
tests/test_drift_detection.py

data/monitoring/drift_baseline.json
data/monitoring/current_production_data.csv
data/monitoring/drift_report.json
```

### Result

**Phase 6.3 — Drift Detection: ✅ COMPLETED**

---

<a id="phase-64"></a>
## Phase 6.4 — Simulate Realistic Drift

**Objective**
Demonstrate realistic feature drift by simulating a rush-hour / traffic surge scenario on production-like data.

The simulation intentionally changes:
- `pickup_hour` distribution by concentrating records around morning and evening rush hours.
- `distance_km` distribution to represent increased trip distances during traffic surge conditions.

The resulting drifted dataset is evaluated using the existing PSI and KS-based drift detection implementation.

---

### 6.4.1 — Define Rush-Hour Drift Scenario

A realistic traffic surge scenario was selected instead of artificially modifying random feature values.

The simulation represents increased taxi activity during:
```text
Morning rush hour → 07:00–09:00
Evening rush hour → 17:00–19:00
```

The scenario also introduces a moderate shift in `distance_km` to represent longer trips during congested traffic conditions.

**Status: 6.4.1 – COMPLETED ✅**

---

### 6.4.2 — Generate Drifted Dataset

A normal production-like dataset was created by sampling 100,000 records from the training feature distribution.

The drift simulation uses this dataset as the source and generates: `data/monitoring/drifted_production_data.csv`


The generated dataset retains the same 8 monitored features and record count.

```text
Normal production-like data
          ↓
Rush-hour simulation
          ↓
Distance distribution shift
          ↓
Drifted production data
```

**Implementation**

`src/monitoring/drift_simulation.py`


**Status: 6.4.2 – COMPLETED ✅**

---

### 6.4.3 — Run Drift Detection on Normal Data

The normal production-like dataset was evaluated against the training-data drift baseline.

The normal dataset contained:
```text
100,000 records
8 monitored features
```

The drift detector produced negligible PSI values and:
```text
Overall drift detected: False
```

This establishes the expected no-drift behavior before simulation.

**Status: 6.4.3 – COMPLETED ✅**

---

### 6.4.4 — Run Drift Detection on Simulated Data

The drift detector was executed against the simulated rush-hour dataset.

**Key Result**
```text
pickup_hour PSI = 0.745811
```

This is significantly above the configured PSI drift threshold of 0.2.

The distance distribution also changed:
```text
Normal distance mean   ≈ 3.42 km
Drifted distance mean  ≈ 4.13 km
```

The KS test for `distance_km` also detected a statistically significant distribution change:
```text
KS p-value = 0.0
```

**Overall Result**
```text
Overall drift detected: True
```

**Status: 6.4.4 – COMPLETED ✅**

---

### 6.4.5 — Compare Normal vs Drifted Data

Normal and simulated drift reports were preserved and compared.

**Comparison Results**

| Metric | Normal | Drifted |
|---|---|---|
| Records | 100,000 | 100,000 |
| pickup_hour PSI | 0.000164 | 0.745811 |
| distance_km PSI | 0.000033 | 0.041543 |
| KS p-value (distance_km) | 0.9908 | 0.0000 |
| Overall drift | False | True |

The comparison confirms that the simulated rush-hour scenario produces detectable feature distribution drift.

**Comparison Artifact**

`data/monitoring/drift_comparison_report.json`


**Intermediate Reports**
```text
data/monitoring/normal_drift_report.json
data/monitoring/drifted_drift_report.json
```

**Implementation**

`src/monitoring/drift_comparison.py`


**Status: 6.4.5 – COMPLETED ✅**

---

### 6.4.6 — Validate Simulation & Tests

The simulation was validated against the intended behavior:
- Normal dataset contains 100,000 records.
- Drifted dataset contains 100,000 records.
- Both datasets contain the same 8 monitored features.
- Rush-hour records increased from approximately 30.5% to 72.3%.
- Average distance increased from approximately 3.42 km to 4.13 km.
- Normal dataset produced no detected drift.
- Drifted dataset produced detected drift.
- `pickup_hour` PSI exceeded the configured drift threshold.
- KS test detected the distance distribution change.

The production drift-detection configuration was restored to use: `data/monitoring/prediction_logs.csv`


**Status: 6.4.6 – COMPLETED ✅**

---

### Files Added / Modified

```text
src/monitoring/
├── drift_simulation.py       ← Added
└── drift_comparison.py       ← Added

src/monitoring/drift_detection.py
    ← Updated temporarily for simulation
    ← Production configuration restored to prediction_logs.csv

data/monitoring/
├── normal_production_data.csv
├── drifted_production_data.csv
├── normal_drift_report.json
├── drifted_drift_report.json
└── drift_comparison_report.json
```

---

### Result

**Phase 6.4 — Simulate Realistic Drift: ✅ COMPLETED**


<a id="phase-65"></a>

## Phase 6.5 — Alerting

**Objective**
Convert model performance and feature-drift signals into actionable threshold-based alerts.

The implementation uses:
- PSI for feature drift alerts.
- MAE degradation for model-performance alerts.
- Python logging for alert visibility.
- A persisted JSON report for alert results.

No external alerting infrastructure such as Prometheus, Grafana, email, or Slack was introduced.

---

### 6.5.1 — Define Alert Thresholds

Alert thresholds were defined centrally for PSI and MAE degradation.

**PSI Alert Thresholds**

```text
PSI < 0.10
    → NO_ALERT

0.10 ≤ PSI ≤ 0.25
    → WARNING

PSI > 0.25
    → ALERT
```

The existing PSI drift-detection threshold of `0.20` was retained separately from the alert-severity thresholds.

```text
PSI_WARNING_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD   = 0.20
PSI_ALERT_THRESHOLD   = 0.25
```

**MAE Alert Threshold**

A model-performance alert is triggered when current MAE degrades by more than 20% relative to the established baseline.

```text
MAE degradation > 20%
        ↓
      ALERT
```

```text
MAE_DEGRADATION_THRESHOLD = 0.20
```

The relative degradation approach avoids using an arbitrary fixed MAE value and allows the alert threshold to remain tied to the established model baseline.

**Status: 6.5.1 – COMPLETED ✅**

---

### 6.5.2 — Implement Alert Evaluation

Threshold-based alert evaluation was implemented in: `src/monitoring/alerting.py`


The implementation provides:

```text
classify_psi_alert()
    → Classifies individual feature PSI severity.

evaluate_mae_alert()
    → Evaluates MAE degradation against baseline.

evaluate_alerts()
    → Combines PSI and MAE alert results.
```

The alerting layer consumes the existing monitoring reports rather than recalculating PSI or MAE.

```text
Drift Report
      │
      └── PSI
           ↓
      PSI Alert

Performance Report + Baseline
      │
      └── MAE
           ↓
      MAE Alert
```

An overall alert is triggered when:
```text
PSI ALERT
    OR
MAE ALERT
```

**Status: 6.5.2 – COMPLETED ✅**

---

### 6.5.3 — Generate Alert Report

The alerting implementation reads:
```text
data/monitoring/drift_report.json
data/monitoring/performance_report.json
data/monitoring/baseline_metrics.json
```

and generates:
`data/monitoring/alert_report.json`


The alert report contains:
- PSI severity for each monitored feature.
- PSI values.
- Baseline MAE.
- Current MAE.
- MAE degradation.
- Configured alert thresholds.
- Overall alert status.

**For the simulated drift scenario:**

```text
pickup_hour PSI = 0.7458
        ↓
PSI > 0.25
        ↓
ALERT
```

The current MAE remained equal to the baseline:

```text
Baseline MAE = 492.22 sec
Current MAE  = 492.22 sec
Degradation = 0%
        ↓
No MAE alert
```

**Overall Result**

```text
Overall alert = True
```

The alert was triggered by feature drift.

**Status: 6.5.3 – COMPLETED ✅**

---

### 6.5.4 — Validate Alerting

Alerting behavior was validated for both PSI and MAE conditions.

**PSI Validation**

```text
PSI 0.05 → NO_ALERT
PSI 0.15 → WARNING
PSI 0.30 → ALERT
```

**MAE Validation**

```text
Baseline MAE = 492.22
Current MAE  = 492.22
→ No alert

Baseline MAE = 492.22
Current MAE  = 650
→ 32.05% degradation
→ ALERT
```

**The simulated drift scenario was also validated:**

```text
pickup_hour PSI = 0.7458
→ ALERT

MAE degradation = 0%
→ No MAE alert

Overall alert = TRUE
```

The alerting implementation compiled successfully and the generated alert report was verified.

**Status: 6.5.4 – COMPLETED ✅**

---

### Files Added / Modified

```text
src/monitoring/
└── alerting.py
    ← Added threshold-based alert evaluation

data/monitoring/
└── alert_report.json
    ← Generated alert report
```

Existing monitoring artifacts consumed by the alerting layer:
```text
data/monitoring/
├── drift_report.json
├── performance_report.json
└── baseline_metrics.json
```

---

### Result

**Phase 6.5 — Alerting: ✅ COMPLETED**


<a id="phase-66"></a>
## Phase 6.6 — Retraining Trigger

**Objective**
Implement a controlled retraining trigger that defines the controlled rule for deciding when retraining of current production model should be considered, without automatically retraining it whenever drift is detected.

We deliberately avoided:
```text
Drift detected → Immediately retrain our current production model
```

Instead, retraining is considered when either significant feature drift or model performance degradation crosses a defined threshold.

```text
Monitoring Results
        ↓
Evaluate Drift / Performance
        ↓
 ┌────────────────────────────┐
 │ PSI > 0.25                 │
 │ OR                         │
 │ MAE degradation > 20%      │
 └────────────────────────────┘
        ↓
Retraining Candidate
```

---

### 6.6.1 — Define Retraining Trigger Rules

Two conditions were defined as retraining triggers.

**Significant Feature Drift**

```text
PSI > 0.25
```

A feature with PSI above `0.25` is considered to have significant distribution drift and can trigger retraining.

**Model Performance Degradation**

```text
Current MAE > Baseline MAE × 1.20
```

Equivalent to:
```text
MAE degradation > 20%
```

The thresholds were kept consistent with the alerting and monitoring design from Phase 6.5.

**Status: 6.6.1 – COMPLETED ✅**

---

### 6.6.2 — Implement Retraining Decision

Implemented `retraining_trigger.py` to evaluate both conditions.

**The decision logic is:**

```text
Significant PSI drift?
       OR
MAE degradation > 20%?
       ↓
YES → Retraining Candidate
NO  → No Retraining
```

The implementation evaluates:
- PSI values from the drift report
- Baseline MAE
- Current MAE
- MAE degradation
- Triggered features

The decision is represented using:
```text
retraining_candidate = True / False
```

**Status: 6.6.2 – COMPLETED ✅**

---

### 6.6.3 — Generate Retraining Trigger Report

A dedicated retraining trigger report was generated containing:
- Configured thresholds
- Overall retraining decision
- Drift trigger status
- Features that exceeded the PSI threshold
- Baseline MAE
- Current MAE
- MAE degradation
- Performance trigger status

**Report location**

data/monitoring/retraining_trigger_report.json


**Example from the validated drift scenario**

```text
Retraining candidate: True

Drift trigger:
    True

Triggered features:
    pickup_hour

Performance trigger:
    False
```

This provides an auditable explanation of why retraining was considered.

**Status: 6.6.3 – COMPLETED ✅**

---

### 6.6.4 — Validate Retraining Trigger

The retraining trigger was validated against three scenarios.

**Scenario 1 — Normal Data**

```text
Drift trigger:       False
Performance trigger: False
Retraining candidate: False
```

Result:
```text
NO RETRAINING
```

**Scenario 2 — Significant PSI Drift**

Using the simulated rush-hour drift:

```text
Drift trigger:       True
Triggered feature:   pickup_hour
Performance trigger: False
Retraining candidate: True
```

Result:
```text
RETRAINING CANDIDATE
```

**Scenario 3 — MAE Degradation >20%**

A degraded MAE scenario was tested:

```text
Baseline MAE: 492.22
Current MAE:  650
MAE degradation ≈ 32.05%
```

Therefore:
```text
Drift trigger:       False
Performance trigger: True
Retraining candidate: True
```

Result:
```text
RETRAINING CANDIDATE
```

**Status: 6.6.4 – COMPLETED ✅**

---

### Files Added / Changed

**Source**

src/monitoring/retraining_trigger.py


Implemented:
- Retraining threshold definitions
- PSI-based trigger evaluation
- MAE degradation evaluation
- Retraining candidate decision
- Retraining trigger report generation

**Generated Artifact**

data/monitoring/retraining_trigger_report.json


---

### Phase 6.6 Final Outcome

**Phase 6.6 — Retraining Trigger: ✅ COMPLETE**

---


## Phase 6.7 — Retraining Pipeline

**Objective**
Implement a controlled retraining workflow that reuses the existing Phase 4 training pipeline instead of rebuilding it.

The Phase 4 final model remains the production baseline, while every retrained model is stored separately as a versioned candidate.

```text
Retraining Trigger
        ↓
Retraining Data
        ↓
Existing Data Pipeline
        ↓
Existing Preprocessing
        ↓
Candidate XGBoost Model
        ↓
Candidate Evaluation
        ↓
Candidate vs Production
        ↓
Promotion Decision
        ↓
Promote / Retain Production
```

---

### 6.7.1 — Define Retraining Data Source

For this academic project, no new production dataset is available.

Therefore, the existing Phase 3 ML-ready training and validation datasets are reused as the retraining data source.

```text
data/split/X_train.csv
data/split/X_val.csv
data/split/y_train.csv
data/split/y_val.csv
```

This allows the complete retraining workflow to be demonstrated without modifying the original dataset or introducing artificial training data.

**Status: 6.7.1 – COMPLETED ✅**

---

### 6.7.2 — Reuse Existing Data Pipeline

The retraining pipeline reuses the existing Phase 4 training components for:
- Data loading
- Feature preparation
- Train/validation data handling
- Data contract validation

The existing `data_contract.py` functionality is reused to ensure that the retraining data maintains the same expected structure as the original training pipeline.

**Result**
```text
X_train = 1,166,833 × 10
X_val   =   291,709 × 10
```

Data contract validation passed successfully.

**Status: 6.7.2 – COMPLETED ✅**

---

### 6.7.3 — Reuse Existing Preprocessing

The existing Phase 4 preprocessing configuration and feature representation are reused.

No new preprocessing logic is introduced.

This ensures that the candidate model receives the same feature representation and preprocessing configuration as the production model.

```text
Existing preprocessing
        ↓
X_train: 10 features
X_val:   10 features
```

This avoids training/serving inconsistencies between the production and retraining pipelines.

**Status: 6.7.3 – COMPLETED ✅**

---

### 6.7.4 — Train Candidate Model

The existing Phase 4 XGBoost configuration is reused.

The selected hyperparameters are loaded from:

artifacts/tuning/best_params.json


A new XGBoost model is trained using the retraining dataset.

The candidate model is not written over the production model.

It is stored separately:
```text
artifacts/
└── retrained/
    └── v1/
        └── model.joblib
```

This provides versioned retraining history.

**Status: 6.7.4 – COMPLETED ✅**

---

### 6.7.5 — Evaluate Candidate Model

The newly trained candidate model is evaluated using the existing validation dataset.

The following regression metrics are calculated:
- MAE
- RMSE
- R²

**Current Candidate Results**

| Metric | Candidate |
|---|---|
| MAE | 492.2197 |
| RMSE | 3223.6490 |
| R² | 0.020799 |

Candidate metrics are persisted to:

artifacts/retrained/v1/metrics.json


**Status: 6.7.5 – COMPLETED ✅**

---

### 6.7.6 — Candidate vs Production Comparison

The candidate model is compared against the existing Phase 4 production model.

**Current Comparison**

| Metric | Production | Candidate |
|---|---|---|
| MAE ↓ | 449.6382 | 492.2197 |
| RMSE ↓ | 3203.0254 | 3223.6490 |
| R² ↑ | 0.033288 | 0.020799 |

The candidate performed worse than the production model.

Therefore:
```text
candidate_better = False
```

The comparison is persisted to:

artifacts/retrained/v1/comparison_report.json


**Status: 6.7.6 – COMPLETED ✅**

---

### 6.7.7 — Model Promotion Decision

A promotion decision is made based on candidate-vs-production performance.

**If candidate is better**
```text
Candidate better
      ↓
PROMOTE_CANDIDATE
      ↓
Candidate replaces production model
```

**If candidate is not better**
```text
Candidate not better
      ↓
RETAIN_PRODUCTION
      ↓
Production model remains unchanged
```

**For the current run**

```text
Candidate better: False
Decision: RETAIN_PRODUCTION
Production model changed: False
```

The current production model was therefore not modified.

Promotion decision is persisted to:

artifacts/retrained/v1/promotion_decision.json


**Status: 6.7.7 – COMPLETED ✅**

---

### 6.7.8 — Validate Retraining Pipeline

The complete retraining workflow was executed end-to-end and successfully validated.

**Training Execution**

Candidate model successfully trained and saved:

artifacts/retrained/v1/model.joblib


**Candidate Evaluation**

MAE, RMSE and R² were successfully calculated and persisted.

**Candidate Comparison**

Candidate was successfully compared against the production model.

**Promotion Decision**

The pipeline correctly identified that the candidate was inferior and retained the existing production model.

```text
Candidate better → False
        ↓
RETAIN_PRODUCTION
        ↓
Production model unchanged
```

**Status: 6.7.8 – COMPLETED ✅**

---

### Files Changed / Added

**Source Code**

src/training/retraining.py


New Phase 6 retraining module containing:
- Retraining data loading
- Existing preprocessing reuse
- Candidate model training
- Candidate evaluation
- Production comparison
- Promotion decision
- Candidate promotion logic

**Generated Retraining Artifacts**
```text
artifacts/retrained/v1/
├── model.joblib
├── metrics.json
├── comparison_report.json
└── promotion_decision.json
```

**Production Model**

artifacts/final/model/final_model.joblib

Not modified during the current retraining run.

---

### Phase 6.7 Final Outcome

**Phase 6.7 — Retraining Pipeline: ✅ COMPLETE**

---

<a id="phase-68"></a>

# Phase 6 — Monitoring, Alerting & Retraining

## 6.8 — End-to-End Monitoring Demo

This phase demonstrated the complete production monitoring → drift detection → alerting → retraining → evaluation → promotion decision lifecycle using the components implemented in Phase 6.

---

### 6.8.1 — Prepare Production Monitoring State

**Objective**
Generate production-like prediction logs from the deployed API and prepare the monitoring state.

The Production API was executed and prediction requests generated `prediction_logs.csv`.

Verified the generated log:
```powershell
Get-Content .\data\monitoring\prediction_logs.csv -TotalCount 3
```

**Sample Output**
```text
timestamp,vendor_id,passenger_count,store_and_fwd_flag,pickup_hour,pickup_day_of_week,pickup_month,is_weekend,distance_km,predicted_eta
2026-08-20T14:50:26.978589+00:00,2,1,N,1,1,5,0,9.529875,2008.8819580078125
2026-08-21T11:37:55.817336+00:00,2,1,N,1,1,5,0,9.529875,2008.8819580078125
```

Verified monitoring source files:
```powershell
Get-ChildItem .\src\monitoring\ -File
```

Monitoring components included:
```text
prediction_logger.py
performance_monitor.py
drift_detection.py
alerting.py
retraining_trigger.py
```

**Status: 6.8.1 – COMPLETED ✅**

---

### 6.8.2 — Run Performance Monitoring

**Objective**
Generate model performance metrics using validation data and the production model.

Executed:
```bash
python -m src.monitoring.performance_monitor
```

**The monitoring pipeline:**
```text
Validation data
      ↓
Production model
      ↓
Predictions
      ↓
Performance dataset
      ↓
MAE / RMSE / R²
      ↓
Performance report
```

**Generated**
```text
data/monitoring/performance_dataset.csv
data/monitoring/performance_report.json
```

**Sample Output**
```text
Performance dataset shape: (291709, 10)

MAE  = 492.2197
RMSE = 3223.6490
R²   = 0.0208
```

The existing baseline was preserved:
```text
Existing baseline found
Existing baseline preserved - MAE=492.2197, RMSE=3223.6490, R2=0.0208
```

This ensured that monitoring execution did not overwrite the established baseline.

**Status: 6.8.2 – COMPLETED ✅**

---

### 6.8.3 — Run Drift Monitoring

**Objective**
Demonstrate drift monitoring for both normal and drifted production-like data.

The permanent monitoring configuration continues to use:
data/monitoring/prediction_logs.csv


For the E2E demonstration, the previously generated simulation datasets were temporarily used as accumulated production-like observations.

**Normal Scenario**

The normal production dataset was temporarily used as the monitoring input.

Executed:
```powershell
Copy-Item .\data\monitoring\normal_production_data.csv .\data\monitoring\prediction_logs.csv -Force
```

Then:
```bash
python -m src.monitoring.drift_detection
```

**Sample Result**
```text
Current production records available for drift analysis: 100000

PSI - pickup_hour: 0.000164
PSI - distance_km: 0.000033

KS Test - distance_km:
p_value = 0.990808

Overall drift detected: False
```

This validated the normal/no-drift scenario.

**Drifted Scenario**

The intentionally drifted dataset was then temporarily used to simulate changed production feature distributions:
```powershell
Copy-Item .\data\monitoring\drifted_production_data.csv .\data\monitoring\prediction_logs.csv -Force
```

Executed:
```bash
python -m src.monitoring.drift_detection
```

**Sample Result**
```text
Current production records available for drift analysis: 100000

PSI - pickup_hour: 0.745811
PSI - distance_km: 0.041543

KS Test - distance_km:
p_value = 0.000000

Overall drift detected: True
```

The intentionally shifted `pickup_hour` feature was correctly identified:
```text
pickup_hour
PSI = 0.745811
drift_detected = true
```

This demonstrated that the drift detection pipeline successfully identifies a significant feature distribution change.

**Status: 6.8.3 – COMPLETED ✅**

---

### 6.8.4 — Evaluate Alerts

**Objective**
Evaluate detected drift and model performance against the defined alert thresholds.

Executed:
```bash
python -m src.monitoring.alerting
```

**Generated**
data/monitoring/alert_report.json


**Sample Result**
```text
pickup_hour:
    PSI = 0.745811
    severity = ALERT

MAE:
    baseline = 492.2197
    current  = 492.2197
    degradation = 0.0
    alert = false

overall_alert = true
```

The alert was triggered because:
```text
PSI 0.745811 > 0.25
```

No separate email/notification infrastructure was introduced. The project uses the implemented Python logger + monitoring report approach.

**Status: 6.8.4 – COMPLETED ✅**

---

### 6.8.5 — Evaluate Retraining Trigger

**Objective**
Determine whether the detected monitoring condition qualifies as a retraining candidate.

Executed:
```bash
python -m src.monitoring.retraining_trigger
```

**Generated**
data/monitoring/retraining_trigger_report.json


**Sample Result**
```text
Retraining candidate: True
```

**Report showed**
```text
Drift trigger:
    triggered = true
    threshold = 0.25
    features = ["pickup_hour"]

Performance trigger:
    triggered = false
    degradation = 0.0
    threshold = 0.2
```

Therefore:
```text
Significant feature drift
        ↓
Retraining candidate = TRUE
```

**Status: 6.8.5 – COMPLETED ✅**

---

### 6.8.6 — Execute Retraining Pipeline

**Objective**
Execute the previously implemented Phase 6.7 retraining pipeline.

Executed:
```bash
python -m src.training.retraining
```

**The existing pipeline was reused:**
```text
Retraining data
      ↓
Existing preprocessing
      ↓
Existing XGBoost configuration
      ↓
Candidate model
      ↓
Candidate evaluation
      ↓
Candidate vs Production
```

**Candidate Metrics**
```text
MAE  = 492.2197
RMSE = 3223.6490
R²   = 0.020799
```

Candidate artifacts were stored separately under:
artifacts/retrained/v1/


**Status: 6.8.6 – COMPLETED ✅**

---

### 6.8.7 — Execute Promotion Decision

**Objective**
Promote the candidate only when it performs better than the existing production model.

**Production vs Candidate**

| Metric | Production | Candidate |
|---|---|---|
| MAE | 449.6382 | 492.2197 |
| RMSE | 3203.0254 | 3223.6490 |
| R² | 0.033288 | 0.020799 |

**Result**
```text
Candidate better than production: False
```

**Promotion Decision**
```text
RETAIN_PRODUCTION
```

**Generated**
artifacts/retrained/v1/promotion_decision.json


```json
{
    "decision": "RETAIN_PRODUCTION",
    "candidate_better": false,
    "production_model_changed": false
}
```

Thus, the existing Phase 4 production model was retained.

The design also keeps candidate models versioned separately, allowing future retraining iterations such as:
```text
artifacts/retrained/v1/
artifacts/retrained/v2/
artifacts/retrained/v3/
...
```

without overwriting the production model.

**Status: 6.8.7 – COMPLETED ✅**

---

### 6.8.8 — Validate Complete Lifecycle

**Objective**
Verify that the complete monitoring and retraining lifecycle executed successfully and that the production state was restored.

**Verify Production Prediction Logs**

The original API-generated prediction log was restored:
```powershell
Copy-Item .\data\monitoring\prediction_logs_api_sample.csv .\data\monitoring\prediction_logs.csv -Force
```

Verified:
```powershell
Get-Content .\data\monitoring\prediction_logs.csv -TotalCount 3
```

The original API log structure and records were restored.

**Verify Production Model**
```powershell
Get-Item .\artifacts\final\model\final_model.joblib |
Select-Object FullName,Length,LastWriteTime
```

Production model remained at:
artifacts/final/model/final_model.joblib


with its original timestamp, confirming that the production model was not replaced.

**Verify Promotion Decision**
```powershell
Get-Content .\artifacts\retrained\v1\promotion_decision.json
```

Result:
```json
{
    "decision": "RETAIN_PRODUCTION",
    "candidate_better": false,
    "production_model_changed": false
}
```

**Verify Candidate Artifacts**
```powershell
Get-ChildItem .\artifacts\retrained\v1\
```

Verified:
```text
model.joblib
metrics.json
comparison_report.json
promotion_decision.json
```

**Status: 6.8.8 – COMPLETED ✅**

---

### E2E Lifecycle Demonstrated

```text
Production API
      ↓
Prediction Logs
      ↓
Performance Monitoring
      ↓
Drift Detection
      ↓
Significant Drift
      ↓
Alert
      ↓
Retraining Candidate
      ↓
Candidate Training
      ↓
Candidate Evaluation
      ↓
Candidate vs Production
      ↓
Candidate worse
      ↓
RETAIN_PRODUCTION
```

---

### Demo Cleanup

During the E2E demonstration, `normal_production_data.csv` and `drifted_production_data.csv` were temporarily used as production-like monitoring inputs to provide sufficient data for meaningful drift validation. The actual API-generated `prediction_logs.csv` was preserved through a temporary backup and restored after the demonstration.

The temporary backup was then removed, leaving the project in its original production state.

---

## Phase 6 — COMPLETE ✅

With 6.8 completed, the project now demonstrates the complete M2 → M3 → M4 → M5 ML lifecycle:

```text
Model
  ↓
Monitor
  ↓
Detect Drift / Performance Degradation
  ↓
Alert
  ↓
Trigger Retraining
  ↓
Train Candidate
  ↓
Evaluate
  ↓
Compare
  ↓
Promote only if better
```

**Phase 6 is officially complete. 🎯**





