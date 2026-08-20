# Phase 4 - Model Building and Training

## Table of Contents

- [Phase 4.1 - Modeling Strategy & Data Contract](#phase-41)
- [Phase 4.2 - Establish Baseline Model](#phase-42)
- [Phase 4.3 - Train Candidate Models](#phase-43)
- [Phase 4.4 - Evaluate & Compare Models](#phase-44)
- [Phase 4.5 — Hyperparameter Tuning](#phase-45)
- [Phase 4.6 — Final Model Selection & Retraining](#phase-46)
- [Phase 4.7 — Final Model Handover](#phase-47)

---

## Phase 4 - Start to End Flow (With Phase 3 & Phase 5)

The overall Phase 4 workflow is:

```text                  
                       PHASE 3
                   Data Engineering
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
    X_train_processed.csv │   X_val_processed.csv
    y_train.csv           │   y_val.csv
              │   preprocessor.joblib
              │           │           │
              └───────────┼───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │      PHASE 4.1      │
                │                     │
                │ Modeling Strategy   │
                │        +            │
                │  Data Contract      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      PHASE 4.2      │
                │                     │
                │   Baseline Model    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      PHASE 4.3      │
                │                     │
                │  Candidate Models   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      PHASE 4.4      │
                │                     │
                │  Evaluation &       │
                │  Comparison         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      PHASE 4.5      │
                │                     │
                │  Hyperparameter     │
                │  Tuning             │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      PHASE 4.6      │
                │                     │
                │  Select & Retrain   │
                │  Final Model        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      PHASE 4.7      │
                │                     │
                │  Save Model &       │
                │  Handover Artifacts │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │       PHASE 5       │
                 │    ML Engineering   │
                 │     / Deployment    │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
       preprocessor.joblib      final_model.joblib
           (Phase 3)               (Phase 4)
                 │                     │
                 ▼                     │
           Raw API Request             │
                 │                     │
                 ▼                     │
       preprocessor.transform()        │
                 │                     │
                 ▼                     │
          10 model features ───────────┘
                       │
                       ▼
              final_model.predict()
                       │
                       ▼
                 Predicted ETA
```

<a id="phase-41"></a>

## Phase 4.1 – Modeling Strategy & Data Contract

### 1. Overview

Phase 4 focuses on building, evaluating, tuning, and selecting the machine learning model for the NYC Taxi ETA Prediction project.

Before model training begins, Phase 4.1 establishes two important foundations:

1. **Modeling Strategy**
2. **Phase 3 → Phase 4 Data Contract**

The purpose of this phase is to ensure that:

- The machine learning problem is clearly defined.
- The target variable is explicitly identified.
- The evaluation metrics are standardized.
- The expected model features are fixed.
- The Phase 3 output is validated before model training.
- Unexpected changes in feature names, feature order, row counts, data types, or missing values are detected early.

Phase 4.1 does **not** perform model training, feature engineering, data cleaning, preprocessing, or train/validation splitting.

Those responsibilities belong to other phases of the project.


---

<a id="phase-42"></a>

## Phase 4.2 – Establish Baseline Model

### 4.2.1 Objective

Phase 4.2 establishes a simple baseline regression model that provides the reference performance level for subsequent machine-learning models.

The baseline is not intended to deliver useful predictive performance. It answers the fundamental question: **can candidate models perform better than a simple constant prediction?** All later models use the same validation data and Phase 4 metrics, making the baseline the minimum performance benchmark.

### 4.2.2 Baseline Model Selection

Selected model:

```python
DummyRegressor(strategy="mean")
```

`DummyRegressor` makes predictions without learning relationships between features and target. With the `mean` strategy, it predicts the mean training `trip_duration` for every validation record.

```text
Training target → mean(trip_duration) → one constant prediction → every validation record
```

The mean strategy is appropriate for the RMSE-oriented regression baseline and avoids unnecessary complexity.

### 4.2.3 Position in the Phase 4 Pipeline

The baseline model consumes the ML-ready datasets produced by Phase 3.

```text
                         PHASE 3
                   Data Engineering
                          │
                          ▼
              ML-ready processed datasets
                          │
                          ▼
                ┌─────────────────────┐
                │      PHASE 4.1      │
                │                     │
                │ Modeling Strategy   │
                │        +            │
                │  Data Contract      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      PHASE 4.2      │
                │                     │
                │  Baseline Model     │
                │                     │
                │ DummyRegressor      │
                │ strategy="mean"     │
                └──────────┬──────────┘
                           │
                           ▼
                 Validation Predictions
                           │
                           ▼
                 RMSE / MAE / R²
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        Local Report                 MLflow
     baseline_metrics.json          Tracking
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                       Baseline
                      Benchmark
                           │
                           ▼
                      PHASE 4.3
                 Candidate Models
```

The baseline therefore establishes the reference point before more sophisticated models are introduced.

### 4.2.4 Input Data

The baseline reuses the Phase 3 → Phase 4 data contract and does not load raw data, clean data, engineer features, split data, preprocess data, or fit the Phase 3 preprocessor.

| Input | Path | Dimensions / rows |
|---|---|---:|
| Training features | `data/processed/X_train_processed.csv` | 1,166,833 × 10 |
| Validation features | `data/processed/X_val_processed.csv` | 291,709 × 10 |
| Training target | `data/split/y_train.csv` | 1,166,833 |
| Validation target | `data/split/y_val.csv` | 291,709 |

Target: `trip_duration`.

### 4.2.5 Data Contract Validation

Before baseline training, the Phase 4.1 data contract validates:

- Modeling configuration, problem type, and target variable
- Expected feature names and order
- Training/validation feature compatibility
- Feature/target row alignment
- Missing values and numeric data types

**Result: Phase 4 data contract validation PASSED.**

### 4.2.6 Baseline Implementation

Implementation file:

```text
src/training/baseline.py
```

Workflow:

```text
Load configuration → validate configuration → load Phase 3 data
        ↓
Validate data contract → train DummyRegressor → predict validation data
        ↓
Calculate RMSE, MAE, R² → log MLflow run → save local metrics
```

The implementation reuses existing data-loading and validation logic rather than duplicating it.

### 4.2.7 Evaluation Metrics

| Priority | Metric | Interpretation |
|---|---|---|
| Primary | RMSE | Penalizes larger prediction errors; lower is better. |
| Secondary | MAE | Average absolute error; lower is better. |
| Context | R² | Proportion of target variance explained. |

### 4.2.8 Baseline Results

| Metric | Baseline result | Rounded |
|---|---:|---:|
| RMSE | 3258.366782573327 | 3258.3668 |
| MAE | 641.4636558199387 | 641.4637 |
| R² | -0.00040622054028571775 | -0.000406 |

These values are the Phase 4 benchmark. Future candidates must demonstrate meaningful improvement, especially over the baseline RMSE of **3258.3668**.

### 4.2.9 Interpretation of Baseline Results

The baseline predicts the same training-target mean for every trip, so it does not use any of the 10 model features. Its R² is approximately zero, consistent with a mean-based constant predictor that provides no explanatory power beyond the reference benchmark.

### 4.2.10 MLflow Experiment Tracking

| Field | Value |
|---|---|
| Experiment | `NYC-Taxi-ETA-Model-Development` |
| Experiment ID | 1 |
| Run | `baseline_dummy_regressor_mean` |
| Run ID | `70b3a6c4523646b1a3cfa7663dae0638` |
| Status | Finished |
| Local UI | `http://127.0.0.1:5000` |

### 4.2.11–4.2.13 MLflow Parameters, Metrics, and Tags

Logged parameters:

```text
model_type=DummyRegressor    strategy=mean       problem_type=regression
target=trip_duration         training_rows=1166833
validation_rows=291709       feature_count=10
```

Logged metrics match the local report exactly:

```text
rmse=3258.366782573327
mae=641.4636558199387
r2=-0.00040622054028571775
```

Logged tags:

```text
phase=4.2    model_role=baseline    metric_priority=rmse
```

### 4.2.14 Local Baseline Artifact

Metrics are persisted in:

```text
reports/baseline_metrics.json
```

```json
{
  "model": "DummyRegressor",
  "strategy": "mean",
  "training_rows": 1166833,
  "validation_rows": 291709,
  "metrics": {
    "rmse": 3258.366782573327,
    "mae": 641.4636558199387,
    "r2": -0.00040622054028571775
  }
}
```

The baseline model itself is not persisted as a production model. A production artifact will be created only after candidate comparison, tuning, selection, and retraining.

### 4.2.15 Testing

Dedicated tests:

```text
tests/test_baseline.py
```

Coverage includes mean-prediction behavior, RMSE/MAE/R² calculations, metric output contract, and perfect-prediction behavior.

```powershell
python -m pytest tests/test_baseline.py -v
```

Result: **4 passed**.

```powershell
python -m pytest tests/test_modeling_data_contract.py tests/test_baseline.py -v
```

Result: **8 passed** — Phase 4.1: 4/4, Phase 4.2: 4/4.

### 4.2.16 End-to-End Execution Verification

```powershell
python -m src.training.baseline
```

The run confirmed data-contract validation, baseline-model training, MLflow run completion, metric persistence, and successful Phase 4.2 completion.

### 4.2.17 Reproducibility and Tracking Architecture

```text
Phase 4.2
   ├── Local JSON report → baseline_metrics.json
   └── MLflow run → parameters, metrics, tags, source/Git context
                         ↓
                   Baseline benchmark
```

This provides both a project-local evaluation artifact and experiment-tracking evidence.

### 4.2.18 Relationship with DVC

Phase 4.2 consumes Phase 3 datasets already versioned through DVC; it does not create a new dataset version.

```text
DVC    → data versioning and pipeline reproducibility
MLflow → model experiments, parameters, and metrics
```

### 4.2.19 Relationship with Phase 5

The baseline is not deployed. Phase 4 will produce a final trained model after candidate modeling, evaluation, tuning, and selection. Phase 5 will use the Phase 3 `preprocessor.joblib` and final model artifact for inference.

### 4.2.20 Phase 4 Progress

```text
4.1 Modeling Strategy & Data Contract   ✅
4.2 Establish Baseline Model             ✅
4.3 Train Candidate Models               ⏳
4.4 Evaluate & Compare Models            ⏳
4.5 Hyperparameter Tuning                ⏳
4.6 Select & Retrain Final Model         ⏳
4.7 Save Model & Evaluation Artifacts    ⏳
```

### 4.2.21 Completion Criteria

The baseline strategy, `DummyRegressor(strategy="mean")`, data-contract reuse, validation predictions, metrics, JSON artifact, MLflow run with parameters/metrics/tags, tests, and MLflow UI verification are complete.

### 4.2.22 Status

**PHASE 4.2 – COMPLETED SUCCESSFULLY ✅**

The established benchmark is:

```text
RMSE = 3258.3668
MAE  = 641.4637
R²   = -0.000406
```

This baseline is the reference point for Phase 4.3 – Train Candidate Models.

---

<a id="phase-43"></a>

## Phase 4.3 – Train Candidate Models

### 4.3.1 – Candidate Model Configuration

#### 4.3.1.1 Objective

The objective of this step is to define the initial candidate machine learning models and their starting configuration before model training begins.

The candidate model configuration is maintained in:

```text
configs/modeling.yaml
```

This keeps model selection and initial hyperparameters outside the Python training implementation.

The configuration-driven approach provides:

- clear separation between configuration and code;
- easier modification of candidate models;
- reproducible initial model configurations;
- simpler experimentation;
- a clean foundation for later hyperparameter tuning.

The parameters defined in this step are **initial candidate configurations only**.

They are not considered optimized hyperparameters.

Hyperparameter optimization will be performed later in:

```
Phase 4.5 – Hyperparameter Tuning
```

---

#### 4.3.1.2 Candidate Model Selection

Three candidate regression algorithms were selected for Phase 4.3:

```
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
```

The candidate set was intentionally kept small.

The objective of Phase 4 is to demonstrate a complete and reproducible ML Engineering workflow rather than evaluate a large number of algorithms.

The candidate models provide three different modeling approaches:

| Candidate | Modeling Approach | Purpose |
|---|---|---|
| Linear Regression | Linear model | Fast and interpretable reference model |
| Random Forest | Bagging / tree ensemble | Capture nonlinear relationships and feature interactions |
| XGBoost | Gradient-boosted trees | Strong candidate for tabular regression |

The baseline model from Phase 4.2 remains separate:

```
DummyRegressor(strategy="mean")
```

It serves as the minimum performance benchmark.

The candidate models must demonstrate improvement over this baseline.

---

#### 4.3.1.3 Configuration Structure

The candidate configuration was added to:

```
configs/modeling.yaml
```

The following configuration was introduced:

```yaml
candidates:
  linear_regression:
    enabled: true
  random_forest:
    enabled: true
    params:
      n_estimators: 50
      max_depth: 15
      min_samples_leaf: 2
      random_state: 42
      n_jobs: -1
  xgboost:
    enabled: true
    params:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 1.0
      random_state: 42
      n_jobs: -1
      objective: reg:squarederror
      tree_method: hist
```

---

#### 4.3.1.4 Candidate Enablement

Each candidate contains an `enabled` flag.

Example:

```yaml
linear_regression:
  enabled: true
```

This allows a candidate model to be included or excluded from the training workflow without modifying Python source code.

For the current Phase 4.3 implementation:

| Candidate | Status |
|---|---|
| Linear Regression | enabled |
| Random Forest | enabled |
| XGBoost | enabled |

Therefore, all three candidates will participate in the initial model-training experiment.

---

#### 4.3.1.5 Linear Regression Configuration

Linear Regression is configured as:

```yaml
linear_regression:
  enabled: true
```

No model-specific parameters are currently required.

The resulting estimator will therefore use the standard scikit-learn configuration:

```python
LinearRegression()
```

**Purpose**

Linear Regression provides:

- a fast candidate model;
- an interpretable linear relationship;
- a useful reference against nonlinear models;
- a low-complexity model for comparison.

It provides an important comparison point before evaluating tree-based models.

---

#### 4.3.1.6 Random Forest Configuration

Random Forest is configured as:

```yaml
random_forest:
  enabled: true
  params:
    n_estimators: 50
    max_depth: 15
    min_samples_leaf: 2
    random_state: 42
    n_jobs: -1
```

**Parameter Explanation**

##### `n_estimators`

```
50
```

The model initially uses 50 decision trees.

This provides a reasonable starting point while avoiding unnecessary computational cost during the initial candidate experiment.

The value is **not considered optimal**.

##### `max_depth`

```
15
```

This limits the maximum depth of individual trees.

The purpose is to control model complexity and avoid unnecessarily deep trees during the initial experiment.

##### `min_samples_leaf`

```
2
```

This requires each leaf to contain at least two training samples.

It provides a small amount of regularization compared with allowing leaves containing a single observation.

##### `random_state`

```
42
```

A fixed random seed is used to make the initial experiment reproducible.

##### `n_jobs`

```
-1
```

This allows Random Forest to use all available CPU cores during training.

The project contains more than one million training records, so parallel tree construction is useful for reducing training time.

---

#### 4.3.1.7 XGBoost Configuration

XGBoost is configured as:

```yaml
xgboost:
  enabled: true
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 1.0
    random_state: 42
    n_jobs: -1
    objective: reg:squarederror
    tree_method: hist
```

**Parameter Explanation**

##### `n_estimators`

```
200
```

The initial model uses 200 boosting rounds/trees.

This is an initial candidate configuration and will be revisited during hyperparameter tuning.

##### `max_depth`

```
6
```

This controls the maximum depth of each boosted tree.

A depth of 6 provides sufficient capacity to capture nonlinear relationships while keeping the initial model configuration reasonably controlled.

##### `learning_rate`

```
0.1
```

This controls the contribution of each boosting iteration.

A value of 0.1 is used as the initial candidate setting.

It is not considered an optimized value.

##### `subsample`

```
0.8
```

Each boosting iteration uses approximately 80% of the training observations.

This introduces stochasticity into the boosting process and can help control overfitting.

##### `colsample_bytree`

```
1.0
```

All available input features are considered for each tree.

The Phase 4 model input currently contains only five processed features, so there is little justification for restricting the feature subset at this stage.

##### `random_state`

```
42
```

A fixed seed is used to improve reproducibility.

##### `n_jobs`

```
-1
```

All available CPU cores can be used during training.

##### `objective`

```
reg:squarederror
```

This explicitly configures XGBoost for a regression problem using squared-error loss.

This is consistent with the project's regression objective and RMSE-based primary evaluation metric.

##### `tree_method`

```
hist
```

The histogram-based tree construction method is used to provide a computationally efficient training approach for the large training dataset.

The project contains approximately:

```
1,166,833 training records
```

so computational efficiency is an important consideration.

---

#### 4.3.1.8 Initial Configuration vs Hyperparameter Tuning

The parameters defined here are **initial candidate settings**.

They should not be interpreted as the final or optimal values.

The workflow is intentionally separated into:

```
4.3 Candidate Models
       │
       │ Initial configurations
       ▼
4.4 Evaluate & Compare
       │
       │ Identify promising candidates
       ▼
4.5 Hyperparameter Tuning
       │
       │ Search parameter space
       ▼
4.6 Final Model Selection
```

This separation prevents the initial candidate experiment from becoming an uncontrolled hyperparameter search.

---

#### 4.3.1.9 Why the Configuration is Externalized

Model parameters could have been hard-coded directly into the Python implementation.

For example:

```python
RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    ...
)
```

However, that approach would make experimentation less maintainable.

Instead, the configuration is maintained in:

```
configs/modeling.yaml
```

and the Python implementation consumes the configuration.

The resulting architecture is:

```
configs/modeling.yaml
       │
       │ Candidate names
       │ Initial parameters
       ▼
Model Factory
       │
       ▼
Candidate Estimators
       │
       ▼
Training Workflow
```

This allows model configurations to be changed without modifying the core training workflow.

---

#### 4.3.1.10 Dependency Verification

The required libraries were already available in the project's virtual environment.

Verified versions:

```
scikit-learn: 1.9.0
xgboost: 3.4.0
```

No additional dependency installation was required for this step.

The project already includes the required machine learning libraries in its dependency configuration.

---

#### 4.3.1.11 Configuration Validation

The candidate configuration was validated using:

```powershell
python -c "import yaml; from pathlib import Path; p=Path('configs/modeling.yaml'); c=yaml.safe_load(p.read_text(encoding='utf-8')); print('Candidates:'); print(yaml.safe_dump(c['candidates'], sort_keys=False))"
```

The configuration was successfully loaded and produced:

```yaml
Candidates:
linear_regression:
  enabled: true
random_forest:
  enabled: true
  params:
    n_estimators: 50
    max_depth: 15
    min_samples_leaf: 2
    random_state: 42
    n_jobs: -1
xgboost:
  enabled: true
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 1.0
    random_state: 42
    n_jobs: -1
    objective: reg:squarederror
    tree_method: hist
```

Therefore:

| Validation | Status |
|---|---|
| Configuration loading | PASSED |
| Candidate definitions | PASSED |
| Candidate enablement | PASSED |

---

#### 4.3.1.12 Completion Criteria

| Requirement | Status |
|---|---|
| Candidate model strategy defined | ✅ |
| Linear Regression configured | ✅ |
| Random Forest configured | ✅ |
| XGBoost configured | ✅ |
| Candidate enable/disable mechanism defined | ✅ |
| Initial model parameters externalized to YAML | ✅ |
| Configuration successfully loaded | ✅ |
| scikit-learn dependency verified | ✅ |
| XGBoost dependency verified | ✅ |
| Hyperparameter tuning deferred to Phase 4.5 | ✅ |

---

#### 4.3.1.13 Status

**PHASE 4.3.1 – COMPLETED SUCCESSFULLY ✅**

The initial candidate model configuration has been defined and validated.

The candidate set is:

```
Baseline
   │
   ├── Linear Regression
   ├── Random Forest Regressor
   └── XGBoost Regressor
```

All three candidate models are currently enabled.

The initial configurations are stored centrally in:

```
configs/modeling.yaml
```

The next implementation step is:

```
4.3.2 – Model Factory
```

which will convert the configured candidate names and parameters into actual regression estimators.

---

### 4.3.2 Implement Reusable Candidate-Model Factory

#### 4.3.2.1 Objective

The objective of this step is to implement a reusable model factory responsible for constructing the candidate regression models defined in Phase 4.3.1.

The model factory provides a single, centralized interface for creating configured candidate models.

The factory currently supports:

```text
Linear Regression
Random Forest Regressor
XGBoost Regressor
```

The factory receives:

- candidate model name;
- optional model-specific parameters.

It returns the corresponding configured regression estimator.

---

#### 4.3.2.2 Why a Model Factory is Used

Without a model factory, the training workflow would need to contain model-specific construction logic such as:

```python
LinearRegression()
RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    ...)
XGBRegressor(
    n_estimators=200,
    max_depth=6,
    ...)
```

Embedding this logic directly inside the training workflow would make the training code increasingly difficult to maintain as more models and experiments are introduced.

Instead, model construction is isolated in:

```
src/training/model_factory.py
```

The resulting architecture is:

```
configs/modeling.yaml
       │
       │ Candidate model name
       │ Model parameters
       ▼
┌──────────────────────────┐
│      Model Factory       │
│                          │
│  create_model(...)       │
└────────────┬─────────────┘
             │
      ┌──────┼──────────────┐
      │      │              │
      ▼      ▼              ▼
   Linear  Random       XGBoost
Regression Forest      Regressor
```

This keeps model construction separate from the training, evaluation, and experiment-tracking logic.

---

#### 4.3.2.3 Implementation Location

The model factory was implemented in:

```
src/training/model_factory.py
```

The module is dedicated to model construction.

It does not perform:

- data loading;
- data preprocessing;
- model training;
- model evaluation;
- hyperparameter tuning;
- MLflow experiment tracking;
- model artifact persistence.

Those responsibilities belong to other Phase 4 components.

---

#### 4.3.2.4 Supported Models

The factory currently supports exactly three candidate models:

```
SUPPORTED_MODELS = {
    "linear_regression",
    "random_forest",
    "xgboost",
}
```

The mapping is:

| Configuration Name | Estimator |
|---|---|
| `linear_regression` | `LinearRegression` |
| `random_forest` | `RandomForestRegressor` |
| `xgboost` | `XGBRegressor` |

This matches the candidate model configuration defined in Phase 4.3.1.

---

#### 4.3.2.5 Factory Interface

The primary factory function is:

```python
create_model(
    model_name,
    params=None,
)
```

The interface accepts:

**`model_name`**

The configured candidate model name.

Examples:

```
linear_regression
random_forest
xgboost
```

**`params`**

An optional dictionary containing model-specific parameters.

Example:

```python
{
    "n_estimators": 50,
    "max_depth": 15,
    "random_state": 42,
}
```

The factory passes these parameters to the corresponding estimator.

---

#### 4.3.2.6 Model Construction Logic

The factory maps each supported model name to its corresponding estimator.

Conceptually:

```
create_model("linear_regression")
       │
       ▼
  LinearRegression()

create_model("random_forest", params)
       │
       ▼
  RandomForestRegressor(**params)

create_model("xgboost", params)
       │
       ▼
  XGBRegressor(**params)
```

The model factory therefore provides one consistent interface for constructing all Phase 4 candidate models.

---

#### 4.3.2.7 Parameter Handling

The factory accepts optional parameters:

```python
params: dict[str, Any] | None = None
```

If parameters are not provided, an empty dictionary is used.

This allows:

```python
create_model("linear_regression")
```

to construct the default Linear Regression estimator.

For parameterized models:

```python
create_model(
    "random_forest",
    {
        "n_estimators": 50,
        "max_depth": 15,
    },
)
```

the supplied parameters are passed directly to the estimator.

This keeps the factory generic and allows the model configuration to remain externalized in:

```
configs/modeling.yaml
```

---

#### 4.3.2.8 Unsupported Model Handling

The factory explicitly validates the requested model name.

If an unsupported model is requested, it raises:

```
ValueError
```

For example:

```python
create_model("unsupported_model")
```

results in an error indicating that the requested model is not supported and listing the supported model names.

This prevents silent fallback behavior.

A misspelled model name therefore fails immediately instead of accidentally creating the wrong estimator.

---

#### 4.3.2.9 Defensive Implementation

The factory also contains a defensive fallback check.

Even though supported model names are validated before model construction, an unexpected implementation mismatch results in a `RuntimeError`.

This ensures that a model cannot be listed as supported without having a corresponding construction implementation.

---

#### 4.3.2.10 Separation from Phase 3 Preprocessing

The model factory does not contain any preprocessing logic.

Phase 4 receives the ML-ready datasets generated by Phase 3:

```
X_train_processed
X_val_processed
```

The expected model features are already established by the Phase 4.1 data contract:

```
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

The Phase 3 fitted preprocessor is persisted separately as:

```
data/processed/preprocessor.joblib
```

Phase 4 does not refit or recreate this preprocessor.

The factory therefore focuses exclusively on the model estimator.

---

#### 4.3.2.11 Relationship with Configuration

The relationship between the configuration and factory is:

```
configs/modeling.yaml
       │
       │
       ├── linear_regression
       ├── random_forest + parameters
       └── xgboost + parameters
                   │
                   ▼
            model_factory.py
                   │
                   ▼
         Configured estimators
```

This provides a clean separation between:

```
Configuration
       ↓
Model construction
       ↓
Model training
```

The training workflow can therefore remain independent of the specific estimator construction details.

---

#### 4.3.2.12 Unit Testing

Dedicated unit tests were implemented in:

```
tests/test_model_factory.py
```

The tests verify the following behaviors.

**Test 1 – Linear Regression**

Verifies that:

```python
create_model("linear_regression")
```

returns:

```
LinearRegression
```

**Test 2 – Random Forest**

Verifies that:

```python
create_model("random_forest", params)
```

returns:

```
RandomForestRegressor
```

and that supplied parameters such as:

```
n_estimators
max_depth
```

are correctly applied.

**Test 3 – XGBoost**

Verifies that:

```python
create_model("xgboost", params)
```

returns:

```
XGBRegressor
```

and that supplied parameters are correctly applied.

**Test 4 – Unsupported Model**

Verifies that requesting an unsupported model:

```python
create_model("unsupported_model")
```

raises the expected:

```
ValueError
```

---

#### 4.3.2.13 Test Execution

The dedicated test suite was executed using:

```powershell
pytest tests/test_model_factory.py -v
```

The result was:

```
4 passed in 2.79s
```

Detailed result:

```
test_create_linear_regression       PASSED
test_create_random_forest           PASSED
test_create_xgboost                 PASSED
test_unsupported_model_raises_error PASSED
```

Therefore:

```
Model factory tests: 4/4 PASSED
```

---

#### 4.3.2.14 Dependency Verification

The model factory was verified against the installed project dependencies.

The development environment contains:

```
scikit-learn: 1.9.0
xgboost:      3.4.0
```

These versions successfully imported and instantiated the required candidate estimators.

No additional dependency changes were required for this step.

---

#### 4.3.2.15 Current Phase 4.3 Architecture

After completing 4.3.1 and 4.3.2, the Phase 4.3 architecture is:

```
                    configs/modeling.yaml
                            │
                            │ Candidate configuration
                            │
                            ▼
                ┌───────────────────────┐
                │     Model Factory     │
                │                       │
                │   create_model(...)   │
                └───────────┬───────────┘
                            │
                     ┌──────┼──────────────┐
                     │      │              │
                     ▼      ▼              ▼
         LinearRegression  RandomForest   XGBRegressor
                     │      │              │
                     └──────┼──────────────┘
                            │
                            ▼
                    Training Workflow
```

The training workflow itself is intentionally not implemented as part of this step.

That will be addressed in:

```
4.3.3 – Candidate Training Workflow
```

---

#### 4.3.2.16 Engineering Benefits

The reusable model factory provides several engineering benefits.

**Centralized model construction**

All candidate estimator construction is maintained in one location.

**Reduced duplication**

The training workflow does not need separate construction logic for every model.

**Configuration-driven experimentation**

Model parameters can originate from:

```
configs/modeling.yaml
```

rather than being hard-coded throughout the training implementation.

**Fail-fast behavior**

Unsupported model names result in an explicit error.

**Easier future extension**

A future candidate model can be added by:

1. adding the model to the supported model set;
2. adding its construction logic;
3. adding its configuration;
4. adding corresponding unit tests.

The training workflow can continue using the same factory interface.

---

#### 4.3.2.17 Scope Boundaries

The model factory is intentionally limited to model construction.

| Responsibility | Model Factory |
|---|---|
| Read configuration | ❌ |
| Load training data | ❌ |
| Validate data contract | ❌ |
| Preprocess data | ❌ |
| Construct estimator | ✅ |
| Train estimator | ❌ |
| Generate predictions | ❌ |
| Calculate metrics | ❌ |
| MLflow tracking | ❌ |
| Hyperparameter tuning | ❌ |
| Save model artifact | ❌ |

This separation keeps the Phase 4 implementation modular.

---

#### 4.3.2.18 Completion Criteria

| Requirement | Status |
|---|---|
| Reusable model factory implemented | ✅ |
| Linear Regression supported | ✅ |
| Random Forest supported | ✅ |
| XGBoost supported | ✅ |
| Model-specific parameters supported | ✅ |
| Unsupported model validation implemented | ✅ |
| Model construction separated from training | ✅ |
| Preprocessing kept outside model factory | ✅ |
| Unit tests implemented | ✅ |
| Linear Regression test passing | ✅ |
| Random Forest test passing | ✅ |
| XGBoost test passing | ✅ |
| Unsupported-model test passing | ✅ |
| Dedicated test suite | **4/4 passed** |

---

#### 4.3.2.19 Status

**PHASE 4.3.2 – COMPLETED SUCCESSFULLY ✅**

The reusable candidate-model factory has been implemented and verified.

The factory currently supports:

```
Linear Regression
Random Forest Regressor
XGBoost Regressor
```

All dedicated tests are passing:

```
4/4 PASSED
```

The next implementation step is:

```
4.3.3 – Candidate Training Workflow
```

The training workflow will use the model factory to construct the configured candidates and will train them using the Phase 3 ML-ready training dataset.

---

### 4.3.3 Implement Candidate Training Workflow

**Purpose**
Implement the initial training workflow for all enabled candidate regression models using the existing Phase 3 ML-ready datasets.

**Implementation**

Created:
- `src/training/candidate_training.py`

The workflow:
- Loads `configs/modeling.yaml`.
- Loads Phase 3 ML-ready training and validation datasets.
- Validates the Phase 3 → Phase 4 data contract.
- Identifies enabled candidate models.
- Creates models using `src/training/model_factory.py`.
- Trains candidates using `X_train` / `y_train` only.
- Generates predictions on `X_val`.
- Calculates RMSE, MAE and R².
- No preprocessing is performed or refitted in Phase 4.

**Candidate Models**
- LinearRegression
- RandomForestRegressor
- XGBRegressor

**Initial Results**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 3205.0706 | 467.0761 | 0.032053 |
| Random Forest | 3236.4813 | 464.3314 | 0.012988 |
| XGBoost | 3203.3086 | 451.0966 | 0.033117 |

Current best candidate by primary metric (RMSE): **XGBoost**.

**Verification**

Candidate training completed successfully for all three models using:
- Training rows: 1,166,833
- Validation rows: 291,709
- Features: 10

Phase 4 data-contract validation passed.

**Status: 4.3.3 – COMPLETED ✅**

---

### 4.3.4 Add MLflow Tracking

**Purpose**
Track each candidate-model training run and its evaluation metrics using MLflow.

**Implementation**

MLflow experiment:
- `NYC-Taxi-ETA-Model-Development`

One MLflow run is created per candidate model.

Each run records:
- Model type
- Problem type
- Target
- Training/validation row counts
- Feature count
- RMSE
- MAE
- R²
- Phase/model-role/metric-priority tags

**Candidate Runs**
- `candidate_linear_regression`
- `candidate_random_forest`
- `candidate_xgboost`

**Verification**

MLflow UI was verified locally at:
- `http://127.0.0.1:5000`

The experiment contains all three candidate runs along with the Phase 4.2 baseline run:
- `baseline_dummy_regressor_mean`
- `candidate_linear_regression`
- `candidate_random_forest`
- `candidate_xgboost`

MLflow tracking successfully records the candidate metrics and run metadata.

**Status: 4.3.4 – COMPLETED ✅**

---

### 4.3.5 Persist Intermediate Candidate Models

**Purpose**
Persist the trained candidate models so they can be reused during later Phase 4 activities without retraining.

**Implementation**

The trained candidate models are persisted using `joblib`.

Artifacts are saved under the configured `models/` directory:

```text
models/
├── linear_regression.joblib
├── random_forest.joblib
└── xgboost.joblib
```

Persistence is handled by `persist_candidate_model()` in:

src/training/candidate_training.py


The function creates the model directory when required and saves each trained candidate as:

<model_name>.joblib


**Verification**

Candidate training was executed successfully and all three model artifacts were created.

| Artifact | Size |
|---|---|
| `linear_regression.joblib` | 1.1 KB |
| `random_forest.joblib` | ~37 MB |
| `xgboost.joblib` | ~0.77 MB |

**Status: 4.3.5 – COMPLETED ✅**

### 4.3.6 Add Tests

**Purpose**
Add focused tests covering the candidate training workflow.

**Implementation**

Added focused tests in:
```text
tests/test_candidate_training.py
```

The tests verify:
- Enabled candidate models are loaded from `modeling.yaml`.
- Persisted candidate models can be reloaded successfully.

**Verification**

```text
2 passed
```

**Status: 4.3.6 – COMPLETED ✅**

---

### 4.3.7 Execute & Verify Candidates

**Purpose**
Execute the complete candidate training workflow end-to-end and verify all outputs.

**Implementation**

Executed the complete candidate training workflow for:
```text
Linear Regression
Random Forest
XGBoost
```

Verified:
- All candidates trained successfully.
- Validation metrics were generated.
- MLflow runs were created.
- Candidate `.joblib` artifacts were created.
- All persisted models were successfully reloaded.
- Focused candidate-training tests passed.

Model artifacts:
```text
models/
├── linear_regression.joblib
├── random_forest.joblib
└── xgboost.joblib
```

**Final candidate metrics from the verified run**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 3205.0706 | 467.0761 | 0.032053 |
| Random Forest | 3236.4813 | 464.3314 | 0.012988 |
| XGBoost | 3203.3086 | 451.0966 | 0.033117 |

XGBoost is currently the best candidate by RMSE. Final model selection remains part of later Phase 4 activities.

**Status: 4.3.7 – COMPLETED ✅**

---

### 4.3 Progress

```text
4.3.1 Candidate Model Configuration       ✅
4.3.2 Reusable Candidate-Model Factory    ✅
4.3.3 Candidate Training Workflow         ✅
4.3.4 MLflow Tracking                     ✅
4.3.5 Persist Candidate Models            ✅
4.3.6 Add Tests                           ✅
4.3.7 Execute & Verify Candidates         ✅
```

---
<a id="phase-44"></a>
## Phase 4.4 Evaluate & Compare Models

### 4.4.1 Define Evaluation & Comparison Contract

**Purpose**
Define a common evaluation contract for all candidate models.

**Evaluation Metrics**

| Priority | Metric | Direction |
|---|---|---|
| Primary | RMSE | Lower is better |
| Secondary | MAE | Lower is better |
| Context | R² | Higher is better |

The Phase 4.2 baseline remains the reference benchmark:
- RMSE = 3258.3668
- MAE = 641.4637
- R² = -0.000406

All candidates are evaluated on the same validation dataset using the same metrics.

**Status: 4.4.1 – COMPLETED ✅**

---

### 4.4.2 Implement Model Evaluation Workflow

**Purpose**
Implement a workflow to evaluate all persisted candidate models on the validation dataset.

**Implementation**

Created:

src/training/model_evaluation.py


The workflow:
- Loads and validates the Phase 4 modeling configuration.
- Loads the Phase 3 training/validation datasets.
- Validates the Phase 3 → Phase 4 data contract.
- Loads persisted candidate models from `models/`.
- Generates validation predictions.
- Calculates RMSE, MAE and R² for each candidate.
- No model training or preprocessing is performed in this step.

**Result**

```text
linear_regression → RMSE: 3205.0706
random_forest     → RMSE: 3236.4813
xgboost           → RMSE: 3203.3086
```

**Status: 4.4.2 – COMPLETED ✅**

---

### 4.4.3 Generate Model Comparison Results

**Purpose**
Persist a consolidated comparison of baseline and candidate model performance.

**Implementation**

Evaluation results are persisted to:

reports/candidate_model_comparison.json


The report contains:
- `primary_metric`
- `secondary_metric`
- `context_metric`
- `baseline`
- `candidates`

**Current Comparison**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline | 3258.3668 | 641.4637 | -0.000406 |
| Linear Regression | 3205.0706 | 467.0761 | 0.032053 |
| Random Forest | 3236.4813 | 464.3314 | 0.012988 |
| XGBoost | 3203.3086 | 451.0966 | 0.033117 |

Current best candidate: **XGBoost** based on the primary metric, RMSE.

**Status: 4.4.3 – COMPLETED ✅**

---

### 4.4.4 Add Evaluation Tests

**Purpose**
Add tests to verify the correctness of the evaluation workflow and comparison report.

**Implementation**

Created:

tests/test_model_evaluation.py


Tests cover:
- Required evaluation metrics are returned.
- Comparison report is persisted as valid JSON.
- Comparison report contains all required metrics.

**Execution**

```bash
python -m pytest tests/test_model_evaluation.py -v
```

**Result**

```text
3 passed
```

**Status: 4.4.4 – COMPLETED ✅**

---

### 4.4.5 Execute & Verify Model Evaluation

**Purpose**
Execute the full evaluation workflow end-to-end and verify all outputs.

**Implementation**

Full evaluation workflow executed using:
```bash
python -m src.training.model_evaluation
```

**Verified**
- Phase 4 data contract validation PASSED
- 3 candidate models loaded
- All candidate models evaluated
- Comparison report persisted successfully

Report verified at:

reports/candidate_model_comparison.json


**Final Test Verification**

```bash
python -m pytest tests/test_model_evaluation.py -v
```

**Result**

```text
3 passed
```

**Status: 4.4.5 – COMPLETED ✅**

---

### 4.4 Status

**PHASE 4.4 – COMPLETED SUCCESSFULLY ✅**

```text
4.4.1 Evaluation & Comparison Contract     ✅
4.4.2 Model Evaluation Workflow            ✅
4.4.3 Model Comparison Report              ✅
4.4.4 Evaluation Tests                     ✅
4.4.5 Execute & Verify                     ✅
```

---
<a id="phase-45"></a>
## Phase 4.5 — Hyperparameter Tuning

**Objective**
Optimize the selected candidate model using automated hyperparameter search and identify the best-performing parameter combination based on validation RMSE.

---

### 4.5.1 — Define Tuning Strategy & Search Space

**Implementation**
- Tuning framework: Optuna
- Search algorithm: TPE (Tree-structured Parzen Estimator)
- Number of trials: 20
- Optimization metric: RMSE
- Direction: Minimize
- Search space defined in `configs/modeling.yaml`

**XGBoost Search Space**

| Hyperparameter | Range |
|---|---|
| n_estimators | 100 – 500 |
| max_depth | 3 – 10 |
| learning_rate | 0.01 – 0.2 |
| subsample | 0.6 – 1.0 |
| colsample_bytree | 0.6 – 1.0 |
| min_child_weight | 1 – 10 |
| reg_alpha | 0.0 – 1.0 |
| reg_lambda | 0.1 – 10.0 |

**Status: 4.5.1 – COMPLETED ✅**

---

### 4.5.2 — Select Tuning Model

Based on Phase 4.4 candidate evaluation, **XGBoost** was selected for hyperparameter tuning.

**Reason**
- Best candidate RMSE: 3203.3086
- Best candidate MAE: 451.0966
- Best candidate R²: 0.033117

**Status: 4.5.2 – COMPLETED ✅**

---

### 4.5.3 — Execute Optuna Tuning

**Implementation**

Implemented `src/training/hyperparameter_tuning.py`.

**Workflow**

```text
Load configuration
      ↓
Load & validate training/validation data
      ↓
Create Optuna TPE study
      ↓
Run 20 trials
      ↓
Train XGBoost for each trial
      ↓
Evaluate validation RMSE
      ↓
Select best parameters
```

**Final Result**

Best RMSE: **3203.0254**

Best parameters:
```text
n_estimators      = 450
max_depth         = 3
learning_rate     = 0.068392
subsample         = 0.783994
colsample_bytree  = 0.951392
min_child_weight  = 8
reg_alpha         = 0.496764
reg_lambda        = 6.072246
```

All 20/20 trials completed successfully.

**Status: 4.5.3 – COMPLETED ✅**

---

### 4.5.4 — MLflow Tracking

**Implementation**

Hyperparameter tuning runs are tracked under:

Experiment:

NYC-Taxi-ETA-Model-Development


MLflow captures:
- Number of trials
- Optimization metric and direction
- Tuning model
- Sampler
- Best RMSE
- Best hyperparameters
- Run ID

**Best Tuning Run**

```text
Run: hyperparameter_tuning_xgboost
Run ID: 89f04a9cbfc74f75a7b7ef550bba4ff7
Best RMSE: 3203.0254
```

The run and its parameters were verified in the MLflow UI.

**Status: 4.5.4 – COMPLETED ✅**

---

### 4.5.5 — Analyze Best Parameters

Optuna identified the following configuration as the best among the 20 trials:

```text
n_estimators      = 450
max_depth         = 3
learning_rate     = 0.068392
subsample         = 0.783994
colsample_bytree  = 0.951392
min_child_weight  = 8
reg_alpha         = 0.496764
reg_lambda        = 6.072246
```

**Comparison with the original XGBoost candidate**

| Metric | Candidate | Tuned |
|---|---|---|
| RMSE | 3203.3086 | 3203.0254 |

The improvement is small, indicating that current model performance is likely more constrained by the available features than by XGBoost hyperparameters.

**Status: 4.5.5 – COMPLETED ✅**

---

### 4.5.6 — Add Tuning Tests

**Implementation**

Created:

tests/test_hyperparameter_tuning.py


Tests verify:
- Tuning configuration is valid.
- Optuna objective executes correctly.
- Objective returns a numeric RMSE value.

**Result**

```text
2 passed
```

**Status: 4.5.6 – COMPLETED ✅**

---

### 4.5.7 — Execute & Verify Tuning

**Verification**

Final verification confirmed:
- 20 trials completed
- Best RMSE: 3203.0254
- Best parameters identified
- MLflow run created successfully
- MLflow parameters and metrics captured

MLflow search also confirmed the tuning run with:
```text
n_trials = 20
optimization_metric = rmse
best_rmse = 3203.025445
```

**Status: 4.5.7 – COMPLETED ✅**

---

### Phase 4.5 — Completed ✅

```text
4.5 Hyperparameter Tuning
│
├── 4.5.1 Define Tuning Strategy & Search Space     ✅
├── 4.5.2 Select Tuning Model                       ✅
├── 4.5.3 Execute Optuna Tuning                     ✅
├── 4.5.4 MLflow Tracking                           ✅
├── 4.5.5 Analyze Best Parameters                   ✅
├── 4.5.6 Add Tuning Tests                          ✅
└── 4.5.7 Execute & Verify Tuning                   ✅
```

**Key outcome:** XGBoost was tuned using Optuna TPE with 20 trials, producing a best validation RMSE of 3203.0254, with the complete tuning run tracked in MLflow.

---
<a id="phase-46"></a>
## Phase 4.6 — Final Model Selection & Retraining

**Objective**
Select the final model configuration, retrain a fresh model using the selected hyperparameters, validate its performance, and verify the final model before artifact handover.

---

### 4.6.1 — Define Final Model Selection Criteria

**Criteria**
- Primary metric: RMSE — lower is better.
- Secondary metric: MAE — lower is better.
- Context metric: R² — higher is better.
- Model must outperform the baseline.
- RMSE is the primary selection criterion; MAE is used as a tie-breaker.

**Status: 4.6.1 – COMPLETED ✅**

---

### 4.6.2 — Select Best Candidate + Tuned Configuration

Based on Phase 4.4 candidate evaluation and Phase 4.5 tuning:

- Selected model: **XGBoost**
- Configuration: Optuna-tuned
- Selection metric: Validation RMSE

**Selected Parameters**

```text
n_estimators      = 450
max_depth         = 3
learning_rate     = 0.068392
subsample         = 0.783994
colsample_bytree  = 0.951392
min_child_weight  = 8
reg_alpha         = 0.496764
reg_lambda        = 6.072246
```

Best validation RMSE:

3203.0254


**Status: 4.6.2 – COMPLETED ✅**

---

### 4.6.3 — Retrain Final Model

**Implementation**

Implemented:

src/training/final_model.py


**Workflow**

```text
Load configuration
      ↓
Load & validate Phase 3 data
      ↓
Create fresh XGBoost model
      ↓
Apply selected Optuna parameters
      ↓
Train on training dataset
```

Training completed successfully on:

Training data: 1,166,833 rows × 10 columns


The final model was not persisted at this stage; artifact persistence is handled in Phase 4.7.

**Status: 4.6.3 – COMPLETED ✅**

---

### 4.6.4 — Final Model Validation

**Implementation**

Implemented:

src/training/final_model_validation.py


The freshly retrained model was evaluated against the validation dataset.

**Final Validation Metrics**

| Metric | Result |
|---|---|
| RMSE | 3203.0254 |
| MAE | 449.6382 |
| R² | 0.033288 |

The final RMSE exactly matched the tuned XGBoost RMSE, confirming reproducibility of the selected configuration.

**Status: 4.6.4 – COMPLETED ✅**

---

### 4.6.5 — Add Tests

**Implementation**

Created:

tests/test_final_model.py


Tests verify:
- Selected Optuna parameters are used by the final model.
- Final model evaluation returns RMSE, MAE and R².

**Result**

```text
2 passed
```

**Status: 4.6.5 – COMPLETED ✅**

---

### 4.6.6 — Execute & Verify

**Verification**

Final verification confirmed:

```text
Phase 3 → Phase 4 data contract       ✅
Selected configuration                ✅
Final model training                  ✅
Validation                            ✅
RMSE reproducibility                  ✅
Unit tests                            ✅
```

**Final Model Result**

| Field | Value |
|---|---|
| Model | XGBoost |
| RMSE | 3203.0254 |
| MAE | 449.6382 |
| R² | 0.033288 |

**Status: 4.6.6 – COMPLETED ✅**

---

### Phase 4.6 — Completed ✅

```text
4.6.1  Define Final Model Selection Criteria        ✅
4.6.2  Select Best Candidate + Tuned Configuration  ✅
4.6.3  Retrain Final Model                          ✅
4.6.4  Final Model Validation                       ✅
4.6.5  Add Tests                                    ✅
4.6.6  Execute & Verify                             ✅
```

---
<a id="phase-47"></a>
## Phase 4.7 — Final Model Handover

**Objective**
Prepare the final trained model and supporting artifacts as a well-defined handover package for the upcoming ML Engineering phase.

The handover package contains the final model, preprocessor, validation metrics, model metadata, and a manifest describing the available artifacts.

---

### 4.7.1 — Define Handover Artifact Contract

**Implementation**

Defined the final artifact contract and standardized the expected handover structure:

```text
artifacts/final/
├── model/
│   └── final_model.joblib
├── preprocessing/
│   └── preprocessor.joblib
├── metrics/
│   └── final_metrics.json
├── metadata/
│   └── model_metadata.json
└── manifest.json
```

This establishes a clear contract between Phase 4 (Model Building) and Phase 5 (ML Engineering).

**Status: 4.7.1 – COMPLETED ✅**

---

### 4.7.2 — Persist Final Model

**Implementation**

Updated the final model training workflow to persist the trained XGBoost model.

Artifact:

artifacts/final/model/final_model.joblib


The model is trained using the hyperparameters selected by Optuna and persisted for downstream consumption.

**Status: 4.7.2 – COMPLETED ✅**

---

### 4.7.3 — Persist Preprocessor

**Implementation**

Persisted the preprocessing transformer used by the ML pipeline.

Artifact:

artifacts/final/preprocessing/preprocessor.joblib


Persisting the preprocessor ensures that future inference can apply the same transformations used during model development.

**Status: 4.7.3 – COMPLETED ✅**

---

### 4.7.4 — Save Final Metrics & Model Metadata

**Implementation**

Removed hard-coded metrics and hyperparameters from the final artifact generation workflow.

The persisted final model is loaded and evaluated against the validation dataset to generate the actual metrics dynamically.

**Final Validation Metrics**

| Metric | Value |
|---|---|
| RMSE | 3203.0254 |
| MAE | 449.6382 |
| R² | 0.033288 |

Metrics artifact:

artifacts/final/metrics/final_metrics.json


Model metadata captures:
- Model type
- Model stage
- Selected hyperparameters
- Training/validation dataset information
- Actual validation metrics
- Paths to final artifacts

Metadata artifact:

artifacts/final/metadata/model_metadata.json


The selected hyperparameters are dynamically loaded from:

artifacts/tuning/best_params.json

ensuring that the handover metadata reflects the actual Optuna-selected configuration.

**Status: 4.7.4 – COMPLETED ✅**

---

### 4.7.5 — Create Handover Manifest

**Implementation**

Created a final manifest describing the four core handover artifacts:

artifacts/final/manifest.json


The manifest references:
- model
- preprocessor
- metrics
- metadata

The manifest generation also validates that all required artifacts exist before creating the manifest.

**Status: 4.7.5 – COMPLETED ✅**

---

### 4.7.6 — Add Handover Artifact Tests

**Implementation**

Added automated tests to verify the final handover package.

Tests validate:
- Required artifacts exist
- Manifest references the correct artifacts
- Final metrics contain RMSE, MAE and R²
- Model metadata contains required information
- Dataset metadata is populated

**Result**

```text
4 passed
```

**Status: 4.7.6 – COMPLETED ✅**

---

### 4.7.7 — Execute & Verify Handover

**Verification**

Executed the complete Phase 4 handover verification.

**Regression Verification**

```text
40 tests passed
```

No test failures were observed.

The final artifact package was verified:

```text
artifacts/final/
├── manifest.json
├── model/
│   └── final_model.joblib
├── preprocessing/
│   └── preprocessor.joblib
├── metrics/
│   └── final_metrics.json
└── metadata/
    └── model_metadata.json
```

**Additional Consistency Checks**

```text
PARAMETERS MATCH: True
METRICS MATCH: True
MANIFEST ARTIFACTS: 4
```

Therefore, the final model handover package is complete, internally consistent, and ready for Phase 5 consumption.

**Status: 4.7.7 – COMPLETED ✅**

---

### Phase 4.7 — Outcome

Phase 4.7 successfully produced a complete final model handover package containing:

- ✅ Final XGBoost model
- ✅ Preprocessor
- ✅ Actual validation metrics
- ✅ Model metadata
- ✅ Handover manifest
- ✅ Automated artifact tests
- ✅ Full regression verification

---

### Phase 4 Final Status

**Phase 4 — Model Building: COMPLETE ✅**

The output of Phase 4 is now ready to become the input/contract for Phase 5 — ML Engineering.





