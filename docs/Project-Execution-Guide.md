# NYC Taxi ETA Prediction — A Guide on How to Run This Project

## Table of Contents

- [Project Information & Important Links](#project-information--important-links)
- [1. Prerequisites](#1-prerequisites)
- [2. Approach 1 — Run Using Docker](#2-approach-1--run-using-docker)
    - [2.1 Pull the Docker Image](#21-pull-the-docker-image)
    - [2.2 Run the Container](#22-run-the-container)
    - [2.3 Open FastAPI Swagger UI](#23-open-fastapi-swagger-ui)
    - [2.4 Test the Prediction API](#24-test-the-prediction-api)
    - [2.5 Test Input Validation](#25-test-input-validation)
- [3. Approach 2 — Run Using Local Python Environment](#3-approach-2--run-using-local-python-environment)
    - [3.1 Create and Activate Virtual Environment](#31-create-and-activate-virtual-environment)
    - [3.2 Install Dependencies](#32-install-dependencies)
- [4. Run FastAPI Locally](#4-run-fastapi-locally)
- [5. Test DVC Reproducibility](#5-test-dvc-reproducibility)
    - [5.1 Check Current DVC Status](#51-check-current-dvc-status)
    - [5.2 Make a Small Pipeline Parameter Change](#52-make-a-small-pipeline-parameter-change)
    - [5.3 Check What DVC Detects](#53-check-what-dvc-detects)
    - [5.4 Reproduce the Pipeline](#54-reproduce-the-pipeline)
- [6. Test MLflow Experiment Tracking](#6-test-mlflow-experiment-tracking)
- [7. Monitoring, Drift Detection, Alerting & Retraining Test](#7-monitoring-drift-detection-alerting--retraining-test)
    - [7.1 Normal Production Scenario](#71-normal-production-scenario)
    - [7.2 Controlled Drift Scenario](#72-controlled-drift-scenario)
    - [7.3 Alert Evaluation](#73-alert-evaluation)
    - [7.4 Retraining Trigger](#74-retraining-trigger)
    - [7.5 Execute Retraining Pipeline](#75-execute-retraining-pipeline)
- [8. Complete Lifecycle Demonstrated](#8-complete-lifecycle-demonstrated)

---

## Project Information & Important Links

| Item | Details |
|---|---|
| **Project Group** | Group 16 |
| **GitHub Repository** | [NYC Taxi ETA Prediction – GitHub](https://github.com/AvinashKumarBarnwal-BitsPilani/NYC-Taxi-ETA-Prediction/tree/main) |
| **Docker Image** | `avinashkumarb6/nyc-taxi-eta-prediction:4.0` |
| **Docker Pull Command** | `docker pull avinashkumarb6/nyc-taxi-eta-prediction:4.0` |
| **Demo Video** | [Google Drive – Demo Video](https://drive.google.com/drive/folders/1YYPeb-C7R9w8SKwIZDQfIl6UUNAg8o8h?usp=sharing) |
| **Raw Dataset** | [Google Drive – Raw Dataset](https://drive.google.com/drive/folders/1URvs4j9SUXeW9_KiSOQTV-TA8kRL2AHH?usp=sharing) |

---
## 1. Prerequisites

The project can be evaluated using either of the following approaches:

- **a)** Docker — Recommended
- **b)** Local Python environment

For DVC and MLflow demonstrations, use the Local Python environment.

---

## 2. Approach 1 — Run Using Docker

Docker is the recommended and simplest way to test the deployed application.

### 2.1 Pull the Docker Image

```bash
docker pull avinashkumarb6/nyc-taxi-eta-prediction:4.0
```

### 2.2 Run the Container

```bash
docker run --name nyc-taxi-eta -p 8000:8000 avinashkumarb6/nyc-taxi-eta-prediction:4.0
```

### 2.3 Open FastAPI Swagger UI

Open the following in a browser:

http://localhost:8000/docs


Swagger UI displays the `/predict` endpoint and its required input schema.

### 2.4 Test the Prediction API

Use the `/predict` endpoint → **Try it out** and provide:

```json
{
  "vendor_id": 1,
  "passenger_count": 2,
  "store_and_fwd_flag": "N",
  "pickup_hour": 10,
  "pickup_day_of_week": 2,
  "pickup_month": 5,
  "is_weekend": 0,
  "distance_km": 5.2
}
```

Click **Execute**.

A successful request returns the predicted ETA.

### 2.5 Test Input Validation

For example, use:

```json
{
  "vendor_id": 1,
  "passenger_count": 7,
  "store_and_fwd_flag": "N",
  "pickup_hour": 10,
  "pickup_day_of_week": 2,
  "pickup_month": 5,
  "is_weekend": 0,
  "distance_km": 5.2
}
```

`passenger_count = 7` should be rejected by the API validation because the configured acceptable range is 1–6.

You can similarly test an invalid month:

pickup_month = 13

which should also be rejected.

---

## 3. Approach 2 — Run Using Local Python Environment

From the project root:
```bash
cd <project-root>
```

### 3.1 Create and Activate Virtual Environment

```bash
python -m venv .venv
```

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3.2 Install Dependencies

```bash
pip install -r requirements-lock.txt
```

---

## 4. Run FastAPI Locally

From the project root:
```bash
python -m uvicorn src.api.main:app --reload
```

Then open:

http://localhost:8000/docs


Use the same `/predict` sample request shown in the Docker section.

---

## 5. Test DVC Reproducibility

DVC is used to manage and reproduce the data engineering pipeline.

### 5.1 Check Current DVC Status

```bash
dvc status
```

Expected result when everything is up to date:
```text
Data and pipelines are up to date.
```

### 5.2 Make a Small Pipeline Parameter Change

Open:

params.yaml


Modify the configured `validation_size` value. Make sure it's a new value.

For example, if currently:
```yaml
validation_size: 0.2
```

change it temporarily to:
```yaml
validation_size: 0.32
```

### 5.3 Check What DVC Detects

```bash
dvc status
```

DVC should identify the affected pipeline stage(s).

### 5.4 Reproduce the Pipeline

```bash
dvc repro
```

DVC will rerun the affected stages based on the changed parameter.

After completion again run:
```bash
dvc status
```

The pipeline should be back to an up-to-date state.

> **Note:** The parameter can be restored to its original value after this demonstration.

---

## 6. Test MLflow Experiment Tracking

MLflow uses a local SQLite database for experiment tracking. The MLflow database path is machine-specific.

From the project root, use:
```bash
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri "sqlite:///./mlflow.db" --host 127.0.0.1 --port 5000 --workers 1
```

Open:

http://127.0.0.1:5000


The MLflow UI should display:

NYC-Taxi-ETA-Model-Development

and the recorded experiment runs.

For a hyperparameter tuning test, run this script. Let the 20 iterations complete and we can see an entry for this in the MLflow DB in MLflow UI:
```bash
python -m src.training.hyperparameter_tuning
```

> **Important:** Run the MLflow command from the project root, where `mlflow.db` is located.

---

## 7. Monitoring, Drift Detection, Alerting & Retraining Test

The following steps demonstrate the complete monitoring lifecycle.

### 7.1 Normal Production Scenario

First establish the normal production state.

**Verify normal production dataset**
```bash
python -c "import pandas as pd; print('Normal rows:', len(pd.read_csv('./data/monitoring/normal_production_data.csv')))"
```

Expected:
```text
Normal rows: 100000
```

**Make normal production data the active monitoring input**
```powershell
Copy-Item .\data\monitoring\normal_production_data.csv .\data\monitoring\prediction_logs.csv -Force
```

**Verify active monitoring data**
```bash
python -c "import pandas as pd; print('Active prediction log rows:', len(pd.read_csv('./data/monitoring/prediction_logs.csv')))"
```

Expected:
```text
Active prediction log rows: 100000
```

**Run drift detection**
```bash
python -m src.monitoring.drift_detection
```

Expected:
```text
Overall drift detected: False
```

### 7.2 Controlled Drift Scenario

To validate the monitoring system, a controlled drift dataset is provided.

**Temporarily introduce controlled drift**
```powershell
Copy-Item .\data\monitoring\drifted_production_data.csv .\data\monitoring\prediction_logs.csv -Force
```

**Run drift detection again**
```bash
python -m src.monitoring.drift_detection
```

The output should show significant drift, including:
```text
PSI - pickup_hour: approximately 0.745811
```
and:
```text
Overall drift detected: True
```

### 7.3 Alert Evaluation

Run:
```bash
python -m src.monitoring.alerting
```

View the generated alert report:
```powershell
Get-Content .\data\monitoring\alert_report.json
```

The report should contain an alert for `pickup_hour` and:
```json
"severity": "ALERT"
```
with:
```json
"overall_alert": true
```

### 7.4 Retraining Trigger

To evaluate the rules/threshold value to decide if we should retrain our model or not.

Run:
```bash
python -m src.monitoring.retraining_trigger
```

View the generated report:
```powershell
Get-Content .\data\monitoring\retraining_trigger_report.json
```

Expected:
```json
"retraining_candidate": true
```
and:
```json
"triggered": true
```
under the drift trigger.

### 7.5 Execute Retraining Pipeline

Run:
```bash
python -m src.training.retraining
```

The retraining pipeline:
- Reuses the existing Phase 4 training configuration
- Trains a candidate model
- Evaluates the candidate
- Compares it against the current production model
- Makes a controlled promotion decision

The candidate model is stored separately under:

artifacts/retrained/v1/


You can inspect:
```text
artifacts/retrained/v1/metrics.json
artifacts/retrained/v1/comparison_report.json
artifacts/retrained/v1/promotion_decision.json
```

**Possible promotion outcomes:**

**a) If the candidate performs better:**
```text
Candidate better than production: True
Model promotion decision: PROMOTE_CANDIDATE
```

**b) If the candidate does not perform better:**
```text
Candidate better than production: False
Model promotion decision: RETAIN_PRODUCTION
```

The system therefore does not blindly promote a retrained model.

---

## 8. Complete Lifecycle Demonstrated

The monitoring demonstration validates the complete lifecycle:

```text
Production API
      ↓
Prediction Logs
      ↓
Performance Monitoring
      ↓
Drift Detection
      ↓
    Alert
      ↓
Retraining Trigger
      ↓
Retraining Pipeline
      ↓
Candidate Model
      ↓
Candidate Evaluation
      ↓
Promotion Decision
```


