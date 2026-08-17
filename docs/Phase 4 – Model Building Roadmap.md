# Phase 4 – Model Building

## 1. Where We Are Coming From

Project: **NYC Taxi ETA / Trip Duration Prediction**.

This is a tabular regression problem: predict taxi `trip_duration` from trip-related information. Phase 4 will train and compare models, track experiments and hyperparameters, identify the best model, and create reproducible model artifacts.

### Phase 3 Is Complete

```text
Raw NYC Taxi Data
        ↓
Data Cleaning → Data Contract Validation → Feature Engineering
        ↓
Train / Validation Split → Preprocessing → DVC Pipeline → Automated Testing
        ↓
ML-Ready Data
```

Phase 4 must not repeat Phase 3 work.

## 2. Phase 3 → Phase 4 Contract

### Features

```text
data/processed/X_train_processed.csv
data/processed/X_val_processed.csv
```

Both contain exactly five processed features:

1. `numerical__distance_km`
2. `categorical__vendor_id_1.0`
3. `categorical__vendor_id_2.0`
4. `categorical__store_and_fwd_flag_N`
5. `categorical__store_and_fwd_flag_Y`

### Target

```text
data/split/y_train.csv
data/split/y_val.csv
```

Target: `trip_duration`.

```text
5 numerical/encoded features → Regression Model → trip_duration
```

### Phase 4 Must Not Do

- Read or clean raw `train.csv`
- Revalidate raw data
- Engineer or recalculate features
- Perform a new train/validation split
- Fit a new encoder, scaler, or preprocessing transformer
- Refit preprocessing on validation data

Those are Phase 3 responsibilities.

## 3. Phase 4 Objective

Systematically experiment with multiple regression models, compare them using appropriate metrics, tune the strongest candidates, select a final model based on evidence, and produce reproducible model artifacts for Phase 5.

The project should include at least two tracked experiments and a justified best-model selection.

## 4. Phase 4 Roadmap

```text
Phase 4 – Model Building
├── 4.1 Define Modeling Strategy
├── 4.2 Establish Baseline Model
├── 4.3 Train Candidate Models
├── 4.4 Evaluate & Compare Models
├── 4.5 Hyperparameter Tuning
├── 4.6 Select Final Model
└── 4.7 Save Model + Evaluation Artifacts
```

### 4.1 – Define Modeling Strategy

Inputs: `X_train_processed`, `X_val_processed`, `y_train`, and `y_val`.

- Problem type: supervised regression
- Target: `trip_duration`
- Primary metric: RMSE
- Secondary metric: MAE
- Context metric: R²

MAE expresses average absolute error; RMSE penalizes large ETA errors; R² expresses explained target variance. Establish this hierarchy before reviewing model results.

### 4.2 – Establish Baseline Model

Train a `DummyRegressor` using the mean or median target. Every candidate model must demonstrate improvement over the baseline.

```text
Baseline → Model A → Model B → Tuned Model
```

### 4.3 – Train Candidate Models

Use a small, meaningful candidate set rather than many algorithms:

| Candidate | Example | Purpose |
|---|---|---|
| Linear regression | `LinearRegression` | Fast, interpretable reference |
| Tree ensemble | `RandomForestRegressor` | Nonlinear relationships and interactions |
| Gradient boosting | XGBoost or LightGBM | Strong tabular-regression candidate |

Two to four meaningful candidates are sufficient. The goal is experimentation quality, not algorithm quantity.

### 4.4 – Evaluate & Compare Models

Evaluate every model on the same untouched validation set.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | … | … | … |
| Linear Regression | … | … | … |
| Random Forest | … | … | … |
| XGBoost | … | … | … |

```text
TRAIN       → fit and tune models
VALIDATION  → evaluate models only
```

### 4.5 – Hyperparameter Tuning

Tune only the strongest candidate models using controlled Optuna searches. For XGBoost, candidates may include `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.

Small, controlled search > huge hyperparameter search.

Each important MLflow run records at minimum:

- Model and parameters
- MAE, RMSE, and R²
- Training information

### 4.6 – Select Final Model

Select the final model through evidence, not popularity:

```text
Baseline → Candidate Models → Metrics → Tuning → Final Comparison → Selection
```

Consider validation performance, metric hierarchy, generalization behavior, reasonable complexity, and training/inference considerations.

### 4.7 – Save Model + Evaluation Artifacts

At minimum, create:

```text
models/
├── final_model.<format>
└── model_metadata.json

reports/
├── model_comparison.csv
├── final_model_metrics.json
└── experiment_summary.md
```

## 5. Phase 4 → Phase 5 Contract

Phase 4 must provide a loadable final model and exact input schema for REST API inference.

Feature order is critical:

```text
distance_km, vendor_id_1.0, vendor_id_2.0,
store_and_fwd_flag_N, store_and_fwd_flag_Y
```

## 6. Preprocessing → API Issue

Phase 4 trains on already preprocessed features, but the Phase 5 API must accept raw trip details.

```text
API Request (raw trip details)
        ↓
Phase 3 preprocessing transformation
        ↓
Five model features
        ↓
Final model
        ↓
Predicted trip_duration
```

Phase 5 must not fit a new transformer; it must use Phase 3's exact transformation. The eventual serving interface should use either the Phase 3 preprocessing artifact plus the model, or a bundled preprocessor-plus-model inference pipeline.

## 7. Phase 4 → Phase 5 Handover Package

```text
PHASE 4 OUTPUT
├── Model
│   └── final_model.joblib
├── Model Metadata
│   └── model_metadata.json
├── Evaluation
│   ├── model_comparison.csv
│   └── final_model_metrics.json
├── Experiment Tracking
│   └── MLflow experiments/runs
└── Inference Contract
    ├── expected features
    ├── feature order
    ├── target definition
    └── model version
```

Phase 5 owns packaging, inference, FastAPI, request validation, error handling, Docker, API testing, and deployment.

## 8. Ownership Boundaries

### Phase 4 Owns

- Model selection and baseline
- Candidate-model training, evaluation, and tuning
- Experiment tracking
- Final-model selection and artifact
- Model-evaluation evidence

### Phase 5 Owns

- Model packaging and loading
- Inference pipeline and REST API
- Request validation and error handling
- Docker, API testing, and deployment

## 9. Phase 4 Expected Evidence

For the M3 experimentation and reproducibility evidence, retain:

- At least two meaningful MLflow experiments/runs
- Baseline and candidate-model results
- Hyperparameter-tuning results
- Final comparison table
- Selected model with justification
- Final model artifact
- Reproducible configuration

## 10. Complete Phase 4 Mental Model

```text
PHASE 3: ML-ready X + y
        ↓
4.1 Modeling Strategy
        ↓
4.2 Baseline
        ↓
4.3 Candidate Models
        ↓
4.4 Evaluate & Compare
        ↓
4.5 Hyperparameter Tuning
        ↓
4.6 Final Model
        ↓
4.7 Save Model + Evidence
        ↓
Final model + metadata + metrics/MLflow
        ↓
PHASE 5: ML Engineering
```

## Bottom Line

Phase 4 starts directly with model building. It does not revisit cleaning, feature engineering, splitting, or preprocessing.
