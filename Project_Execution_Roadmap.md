# 🚀 Project Execution Roadmap

> **Official Submission Date:** 24-Aug-2026
>
> **Team Target Completion Date:** **23-Aug-2026**
>
> We aim to complete all development work by **23-Aug-2026**, keeping one full day as a buffer for final testing, documentation, presentation, and unexpected issues.

---

# 📊 Phase 3 – Data Engineering (M2)

### 🎯 Objective

Build a reliable, reproducible data pipeline that converts raw NYC Taxi trip data into a clean, validated, feature-engineered dataset ready for model training.

---

## Step 1 – Understand the Dataset

- Study the dataset
- Identify target variable
- Understand every feature
- Prepare data dictionary
- Identify missing values and data types

---

## Step 2 – Build Data Ingestion Pipeline

- Read raw dataset
- Config-driven data loading
- Logging
- Exception handling
- Modular pipeline

---

## Step 3 – Validate Incoming Data

Perform validation checks:

- Required columns
- Data types
- Missing values
- Duplicate records
- Invalid timestamps
- Negative trip duration
- Invalid GPS coordinates

---

## Step 4 – Exploratory Data Analysis (EDA)

Perform exploratory analysis.

Generate:

- Missing value analysis
- Target distribution
- Correlation analysis
- Outlier analysis
- Feature distributions

---

## Step 5 – Feature Engineering

Create meaningful features such as:

- Pickup Hour
- Pickup Day
- Weekend Flag
- Month
- Trip Distance
- Rush Hour Indicator

---

## Step 6 – Build Data Preprocessing Pipeline

- Missing value imputation
- Feature encoding
- Feature scaling
- Train/Test split

---

## Step 7 – Dataset Versioning using DVC

- Version processed dataset
- Track dataset changes
- Ensure reproducibility

---

## 📦 Expected Deliverables

- Data ingestion pipeline
- Data validation pipeline
- Feature engineering pipeline
- Data preprocessing pipeline
- EDA report
- Processed dataset
- DVC versioned dataset

---

## 📅 Target Completion

**13-Aug-2026**

---

# 🤖 Phase 4 – Model Development (M3)

### 🎯 Objective

Train, compare, and track multiple machine learning models while selecting the best model using reproducible experimentation.

---

## Step 1 – Build Baseline Model

Train a simple baseline model.

Example:

- Linear Regression

---

## Step 2 – Train Multiple Models

Train and compare models such as:

- Linear Regression
- Random Forest
- XGBoost
- LightGBM

---

## Step 3 – Hyperparameter Tuning

Perform model tuning using:

- GridSearchCV
- Random Search

---

## Step 4 – MLflow Experiment Tracking

Track:

- Parameters
- Metrics
- Artifacts
- Trained Models

---

## Step 5 – Model Comparison

Compare models using:

- RMSE
- MAE
- R² Score

Select the best-performing model.

---

## Step 6 – Save the Best Model

Store the selected model for deployment.

---

## 📦 Expected Deliverables

- Baseline model
- Multiple trained models
- MLflow experiment logs
- Model comparison report
- Best trained model

---

## 📅 Target Completion

**17-Aug-2026**

---

# 🚀 Phase 5 – Deployment (M4)

### 🎯 Objective

Package the trained model and expose it through a production-style REST API.

---

## Step 1 – Build Prediction Pipeline

Create a reusable prediction pipeline.

---

## Step 2 – Develop REST API

Implement FastAPI endpoints.

Example:

- GET /
- POST /predict

---

## Step 3 – Input Validation

Validate incoming requests using Pydantic.

---

## Step 4 – Dockerize the Application

Create:

- Dockerfile
- docker-compose.yml

---

## Step 5 – Test the API

Verify predictions using:

- Swagger UI
- Postman
- cURL

---

## 📦 Expected Deliverables

- Prediction pipeline
- FastAPI application
- Docker image
- Working REST API
- API test results

---

## 📅 Target Completion

**20-Aug-2026**

---

# 📈 Phase 6 – Monitoring & Retraining (M5)

### 🎯 Objective

Monitor deployed model performance, detect drift, and define a retraining strategy.

---

## Step 1 – Prediction Logging

Log:

- Input features
- Predictions
- Timestamp

---

## Step 2 – Model Monitoring

Monitor:

- Prediction distribution
- Feature distribution
- Missing values

---

## Step 3 – Drift Simulation

Simulate scenarios such as:

- Rush hour traffic
- Holiday traffic
- Seasonal variations

---

## Step 4 – Drift Detection

Implement:

- Population Stability Index (PSI)
- KS Test
- Evidently monitoring

---

## Step 5 – Retraining Strategy

Define retraining triggers based on:

- Data drift
- Performance degradation
- Scheduled retraining

---

## Step 6 – End-to-End Demonstration

Demonstrate the complete ML pipeline:

```

Raw Data
│
▼
Data Ingestion
│
▼
Data Validation
│
▼
Feature Engineering
│
▼
Model Training
│
▼
MLflow Tracking
│
▼
Best Model
│
▼
FastAPI Deployment
│
▼
Prediction
│
▼
Monitoring
│
▼
Retraining

```

---

## 📦 Expected Deliverables

- Prediction logs
- Monitoring dashboard
- Drift detection report
- Retraining strategy
- End-to-end ML pipeline demonstration

---

## 📅 Target Completion

**23-Aug-2026**

---

# 🎯 Final Project Milestones

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Phase 3 – Data Engineering | 13-Aug-2026 | ⏳ Planned |
| Phase 4 – Model Development | 17-Aug-2026 | ⏳ Planned |
| Phase 5 – Deployment | 20-Aug-2026 | ⏳ Planned |
| Phase 6 – Monitoring & Retraining | 23-Aug-2026 | ⏳ Planned |
| Buffer Day (Testing, Documentation & Demo) | 24-Aug-2026 | ⏳ Planned |

---

# 🤝 Team Collaboration

- Work in dedicated feature branches.
- Create Pull Requests for completed features.
- Discuss major design changes before implementation.
- Keep documentation updated alongside code changes.
- Commit changes regularly with meaningful commit messages.