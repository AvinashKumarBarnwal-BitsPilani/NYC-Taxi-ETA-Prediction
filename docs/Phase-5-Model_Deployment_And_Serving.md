# Phase 5 - Model Deployment And Serving

## Table of Contents

- [Phase 5.1 - Build Prediction Pipeline](#phase-51)
- [Phase 5.2 - Develop REST API](#phase-52)
- [Phase 5.3 - Input Validation](#phase-53)
- [Phase 5.4 - Dockerize the Application](#phase-54)
- [Phase 5.5 — Test API](#phase-55)
- [Phase 5.6 — Publish Docker Image](#phase-56)

---

<a id="phase-51"></a>
## Phase 5.1 — Build Prediction Pipeline

**Objective**
Create a reusable inference pipeline that consumes the 8 raw input features, applies the persisted Phase 4 preprocessor, loads the final XGBoost model, and returns an ETA prediction.

---

### 5.1.1 — Create Reusable Prediction Pipeline

**Implementation**

Created:

```text 
src/inference/
├── init.py
└── prediction_pipeline.py
```

Implemented `PredictionPipeline` to:
- Load `final_model.joblib`
- Load `preprocessor.joblib`
- Validate required artifacts exist
- Transform raw input using the persisted preprocessor
- Generate prediction using the final XGBoost model
- Return ETA as `float`

**Inference Flow**

```text
Raw Input
   ↓
Preprocessor
   ↓
5 Model-ready Features
   ↓
Final XGBoost Model
   ↓
ETA
```

**Status: 5.1.1 – COMPLETED ✅**

---

### 5.1.2 — Inference Smoke Test

**Implementation**

Initially tested with Phase 4 processed data and discovered a schema mismatch.

The persisted preprocessor expects 8 raw features:
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

Corrected the smoke test to use:

data/split/X_val.csv


**Verified the complete inference path**

```text
8 Raw Features
      ↓
preprocessor.joblib
      ↓
5 Model Features
      ↓
final_model.joblib
      ↓
ETA Prediction
```

**Sample Prediction**

```text
Predicted ETA: 2020.2095 seconds
```

The smoke test successfully loaded both persisted artifacts and generated a prediction.

**Status: 5.1.2 – COMPLETED ✅**

---

### 5.1.3 — Add Prediction Pipeline Tests

**Implementation**

Created:

     tests/test_prediction_pipeline.py


Added 5 tests covering:
- Artifact loading
- Numeric prediction generation
- Invalid input type handling
- Missing model artifact handling
- Missing preprocessor artifact handling

**Result**

```text
5 passed
```

**Status: 5.1.3 – COMPLETED ✅**

---

### 5.1.4 — Execute & Verify

**Verification**

Ran the complete project test suite after adding the inference pipeline.

```text
45 tests
45 passed
0 failed
```

Also re-ran the real inference smoke test successfully:

```text
Prediction artifacts loaded successfully
Predicted ETA: 2020.20947265625
```

**Status: 5.1.4 – COMPLETED ✅**

---

### Final Phase 5.1 Status

```text
5.1 Build Prediction Pipeline
│
├── 5.1.1 Reusable PredictionPipeline   ✅
├── 5.1.2 Inference Smoke Test          ✅
├── 5.1.3 Prediction Pipeline Tests     ✅
└── 5.1.4 Execute & Verify              ✅
```

**Phase 5.1 — COMPLETE ✅**

---
<a id="phase-52"></a>
## Phase 5.2 — Develop REST API

**Objective**
Expose the prediction pipeline through a production-style FastAPI REST API with health and prediction endpoints.

---

### 5.2.1 — Create FastAPI Application

**Implementation**

Created:

```text 
src/api/
├── init.py
└── main.py
```

Implemented the FastAPI application with:
- API metadata
- `GET /` health endpoint
- Integration with `PredictionPipeline`

Verified FastAPI and Uvicorn installation and successfully imported the application.

**Status: 5.2.1 – COMPLETED ✅**

---

### 5.2.2 — Implement `GET /`

**Implementation**

Implemented the health-check endpoint:

GET /


Returns:
```json
{
  "status": "healthy",
  "service": "nyc-taxi-eta-prediction"
}
```

Verified through:
- Browser
- Swagger UI
- PowerShell REST request

Response: HTTP 200 OK.

**Status: 5.2.2 – COMPLETED ✅**

---

### 5.2.3 — Implement `POST /predict`

**Implementation**

Implemented:

POST /predict


The endpoint accepts the 8 raw inference features and passes them to `PredictionPipeline`.

**Flow**

```text
REST Request
    ↓
FastAPI
    ↓
Request → DataFrame
    ↓
PredictionPipeline
    ↓
Preprocessor
    ↓
Final XGBoost Model
    ↓
predicted_eta
```

Verified through Swagger UI with a real inference payload.

**Result**

```text
HTTP 200 OK
predicted_eta = 2020.20947265625
```

**Status: 5.2.3 – COMPLETED ✅**

---

### 5.2.4 — Add API Tests

**Implementation**

Created:

tests/test_api.py


Added tests covering:
- `GET /` health response
- `POST /predict` successful response
- Prediction endpoint returning numeric ETA

All API tests passed.

**Status: 5.2.4 – COMPLETED ✅**

---

### 5.2.5 — Execute & Verify

**Implementation**

Before testing the REST endpoints, start the FastAPI application using Uvicorn:

```bash
python -m uvicorn src.api.main:app --reload
```

Keep the Uvicorn terminal running while testing the API.

Then use a separate terminal to execute:
```text
GET /
POST /predict
```

The API can also be tested interactively through Swagger UI:

http://127.0.0.1:8000/docs


**Runtime Flow**

This establishes the runtime flow:

```text
Terminal 1
    │
    └── Uvicorn / FastAPI server
             │
             │ keeps running
             ▼
Terminal 2 / Browser / Swagger
             │
             ├── GET /
             └── POST /predict
```

**Verification**

Performed final end-to-end verification of the running REST API.

Verified:
```text
GET /          → 200 OK       ✅
POST /predict  → 200 OK       ✅
Prediction     → ETA returned  ✅
Swagger UI     → Working       ✅
```

Executed the complete project test suite:
```text
48 passed
0 failed
```

**Status: 5.2.5 – COMPLETED ✅**

---

### Final Phase 5.2 Status

```text
5.2 Develop REST API
│
├── 5.2.1 Create FastAPI Application    ✅
├── 5.2.2 Implement GET /               ✅
├── 5.2.3 Implement POST /predict       ✅
├── 5.2.4 Add API Tests                 ✅
└── 5.2.5 Execute & Verify              ✅
```

**🎉 PHASE 5.2 — COMPLETE**

---
<a id="phase-53"></a>
## Phase 5.3 — Input Validation

**Objective**
Add structured request validation to the `/predict` endpoint using Pydantic, ensuring invalid inference requests are rejected before reaching the prediction pipeline.

---

### 5.3.1 — Define Prediction Request Model

**Implementation**

Created `src/api/schemas.py` with a Pydantic `PredictionRequest` model defining the 8 inference input fields:

- `vendor_id`
- `passenger_count`
- `store_and_fwd_flag`
- `pickup_hour`
- `pickup_day_of_week`
- `pickup_month`
- `is_weekend`
- `distance_km`

**Status: 5.3.1 – COMPLETED ✅**

---

### 5.3.2 — Add Field Validation

**Implementation**

Added Pydantic field constraints:

- `passenger_count > 0`
- `store_and_fwd_flag` → `N` / `Y`
- `pickup_hour` → `0–23`
- `pickup_day_of_week` → `0–6`
- `pickup_month` → `1–12`
- `is_weekend` → `0–1`
- `distance_km > 0`

**Status: 5.3.2 – COMPLETED ✅**

---

### 5.3.3 — Integrate with `/predict`

**Implementation**

Integrated `PredictionRequest` with the FastAPI `/predict` endpoint.

Modified file:

    src/api/main.py

**Inference Flow**

```text
JSON Request
    ↓
PredictionRequest
    ↓
Pydantic Validation
    ↓
model_dump()
    ↓
Pandas DataFrame
    ↓
PredictionPipeline
    ↓
ETA Prediction
```

Verified valid request returns HTTP 200 with predicted ETA.

**Status: 5.3.3 – COMPLETED ✅**

---

### 5.3.4 — Handle Validation Errors

**Verification**

Verified invalid requests are rejected before reaching the prediction pipeline.

Tested:
```text
pickup_month = 13 → HTTP 422
distance_km = -1  → HTTP 422
```

**Status: 5.3.4 – COMPLETED ✅**

---

### 5.3.5 — Update Swagger Contract

**Verification**

Verified the Pydantic constraints are automatically reflected in the FastAPI Swagger/OpenAPI schema.

Swagger displays:
- Required fields
- Field types
- Numeric ranges
- Positive-value constraints
- `N/Y` categorical constraint

**Status: 5.3.5 – COMPLETED ✅**

---

### 5.3.6 — Add Validation Tests

**Implementation**

Added validation tests to:

    tests/test_api.py

Tests added:
   - test_prediction_rejects_invalid_pickup_month
   - test_prediction_rejects_negative_distance


It Adds coverage for:
- Invalid `pickup_month`
- Negative `distance_km`

**API Test Suite**

```text
5 passed
```

**Status: 5.3.6 – COMPLETED ✅**

---

### 5.3.7 — Execute & Verify

**Verification**

Executed the complete project test suite:

```bash
python -m pytest -v
```

**Result**

```text
50 passed
23 warnings
```

All existing Phase 4, inference pipeline, API, and validation tests continue to pass.

**Status: 5.3.7 – COMPLETED ✅**

---

### Phase 5.3 Status

**✅ COMPLETED**
Want to be notified when Claude responds?

<a id="phase-54"></a>
## Phase 5.4 — Dockerize the Application

---

### 5.4.1 — Prepare Docker Configuration

**Objective**
Prepare a minimal runtime environment containing only the dependencies, application code, and persisted inference artifacts required to serve predictions.

**Runtime Dependencies**

Created:
```text
    requirements-docker.txt
```

Contains the pinned runtime dependencies required for inference:
```text
fastapi==0.141.1
uvicorn==0.52.1
joblib==1.5.3
numpy==2.5.1
pandas==2.3.3
scikit-learn==1.9.0
xgboost==3.4.0
```

The existing project-level dependency files remain unchanged:
```text
requirements.txt
requirements-lock.txt
```

**`requirements-docker.txt`** is intentionally limited to the inference/API runtime dependencies rather than the complete ML engineering environment.

**Docker Runtime Contents**

The container requires:
```text
src/
├── api/
├── inference/
└── utils/

artifacts/final/
├── model/
│   └── final_model.joblib
└── preprocessing/
    └── preprocessor.joblib
```

Training datasets, tests, reports, notebooks, development environment, etc. are not required by the inference container.

**Docker Desktop Prerequisite**

Docker Desktop must be running on the local system before building or running the container.

Verified Docker installation and daemon using:
```bash
docker --version
docker info
```

Docker Engine was confirmed running successfully.

**Status: 5.4.1 – COMPLETED ✅**

---

### 5.4.2 — Create Dockerfile

**Implementation**

Created:
```text
Dockerfile
```

**Key Configuration**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY artifacts/final ./artifacts/final

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Important Configuration**

The container starts Uvicorn using:
0.0.0.0:8000


`0.0.0.0` means Uvicorn listens on all network interfaces available inside the container, allowing Docker to route external traffic to the application.

**Status: 5.4.2 – COMPLETED ✅**

---

### 5.4.3 — Build Docker Image

### Note: For building Docker image or Running the Docker container, no need to be in the Virtual Environment (.venv)
#### To Exit .venv run this PowerShell:
 ```text   
deactivate
```

**Implementation**

Verified the application before containerization:
```bash
python -m py_compile .\src\api\main.py
```

Built the Docker image:
```bash
docker build -t nyc-taxi-eta-api:1.0 .
```

Build completed successfully:
```text
[+] Building 83.4s (11/11) FINISHED
```

Verified the image:
```bash
docker images
```

Created image:
```text
nyc-taxi-eta-api:1.0
```

The Docker image successfully contains the API code, runtime dependencies, model, and preprocessor artifacts.

**Status: 5.4.3 – COMPLETED ✅**

---

### 5.4.4 — Run Docker Container

**Start Container**

Ran:
```bash
docker run --rm --name nyc-taxi-eta-api -p 8000:8000 nyc-taxi-eta-api:1.0
```

Container successfully started with:
```text
Uvicorn running on http://0.0.0.0:8000
```

**Port Mapping**

Docker command:
-p 8000:8000


follows:
-p HOST_PORT:CONTAINER_PORT


Therefore:
```text
Windows Host : 8000
       │
       │ Docker port mapping
       ▼
Container : 8000
       │
       ▼
Uvicorn : 8000
       │
       ▼
FastAPI
```

**URL Mapping**

Inside the container, Uvicorn listens on:
http://0.0.0.0:8000


This is the server bind address, not normally the URL used by the client.

From the Windows host, the API is accessed through:
http://127.0.0.1:8000


Here `127.0.0.1` represents the Windows host's localhost.

Therefore:
```text
Client
  │
  │ http://127.0.0.1:8000
  ▼
Windows Host :8000
  │
  │ -p 8000:8000
  ▼
Docker Container :8000
  │
  ▼
Uvicorn 0.0.0.0:8000
  │
  ▼
FastAPI
```

### Why Virtual Environment (.venv) is Not Required for Docker container? (**Reproducibility Verification**)

After starting the Docker container, the prediction request was executed from a second PowerShell terminal after exiting the local `.venv`.

Therefore, the request was not dependent on our local Python virtual environment.

The flow was:
```text
Windows PowerShell
(no .venv)
      │
      │ HTTP POST
      ▼
127.0.0.1:8000
      │
      ▼
Docker Container
      │
      ├── FastAPI
      ├── PredictionPipeline
      ├── preprocessor.joblib
      └── final_model.joblib
      │
      ▼
2020.20947265625
```

This provides an important containerization/reproducibility proof: the API and model inference can run using the dependencies and artifacts packaged inside the Docker image rather than relying on the developer's local Python environment.

### Think of them as two completely separate environments
```text
Your Windows machine
│
├── Python .venv
│   ├── FastAPI
│   ├── pandas
│   ├── scikit-learn
│   └── etc.
│
└── Docker Desktop
    └── Docker Engine
        └── Container
            ├── Python 3.12
            ├── FastAPI
            ├── pandas
            ├── scikit-learn
            ├── XGBoost
            └── Model + Preprocessor
```

### Current Architecture

```text
                 WINDOWS HOST
        ┌──────────────────────────┐
        │                          │
        │ Browser / PowerShell     │
        │                          │
        │ 127.0.0.1:8000           │
        │        │                 │
        └────────┼─────────────────┘
                 │
                 │ -p 8000:8000
                 ▼
        ┌──────────────────────────┐
        │     DOCKER CONTAINER     │
        │                          │
        │      :8000               │
        │        │                 │
        │     Uvicorn              │
        │        │                 │
        │     FastAPI              │
        │        │                 │
        │    /predict              │
        │        │                 │
        │ PredictionPipeline       │
        │        │                 │
        │   ┌────┴─────┐           │
        │   ▼          ▼           │
        │ Model     Preprocessor   │
        │ .joblib      .joblib     │
        └──────────────────────────┘
```

**Status: 5.4.4 – COMPLETED ✅**

---

### 5.4.5 — Verify Containerized API

---

**API Verification**

**1. Get /Health Check - To Verify GET Request [Run this in another PowerShell terminal]**
```text
Invoke-RestMethod http://127.0.0.1:8000/
```

Returned:
200 OK


with:
```json
{
  "status": "healthy",
  "service": "nyc-taxi-eta-prediction"
}
```

Swagger UI was also accessible through:
http://127.0.0.1:8000/docs


and exposed:
```text
GET  /
POST /predict
```

**2. POST /predict : Containerized Prediction Verification** [Positive Scenario]

The same prediction payload used during local inference was sent to:
POST http://127.0.0.1:8000/predict

Run these 2 commands in PowerShell:

**1. Use our known-good payload:**
```text
$body = @{
    vendor_id = 2
    passenger_count = 1
    store_and_fwd_flag = "N"
    pickup_hour = 1
    pickup_day_of_week = 1
    pickup_month = 5
    is_weekend = 0
    distance_km = 9.529875
} | ConvertTo-Json
```
**2. Then run this:**

```text
Invoke-RestMethod `
     -Uri "http://127.0.0.1:8000/predict" `
     -Method Post `
     -ContentType "application/json" `
     -Body $body
```

Prediction returned:
```text
2020.20947265625
```

The Docker container logs confirmed:
```text
Loading final model:
artifacts/final/model/final_model.joblib

Loading preprocessor:
artifacts/final/preprocessing/preprocessor.joblib

Prediction artifacts loaded successfully

POST /predict HTTP/1.1" 200 OK
```


**3. POST /predict : Containerized Prediction Verification** [Negative Scenario]

Tested invalid Scenario:
```text
pickup_month = 13
```

Run these 2 commands in PowerShell:

**1. Use our known-good payload:**
```text
$body = @{
    vendor_id = 2
    passenger_count = 1
    store_and_fwd_flag = "N"
    pickup_hour = 1
    pickup_day_of_week = 1
    pickup_month = 13
    is_weekend = 0
    distance_km = 9.529875
} | ConvertTo-Json
```
**2. Then run this:**

```text
Invoke-RestMethod `
     -Uri "http://127.0.0.1:8000/predict" `
     -Method Post `
     -ContentType "application/json" `
     -Body $body
```

**Validation**

Result:
```text
422 Unprocessable Entity
```

This Confirms Pydantic validation continues to work inside the Docker container.

**Status: 5.4.5 – COMPLETED ✅**

---

### 5.4.6 — Add Docker/Container Tests

**Implementation**

Created:

tests/test_docker_container.py


Added five container-level smoke/integration tests:
- `test_docker_image_exists`
- `test_docker_container_is_running`
- `test_docker_health_endpoint`
- `test_docker_prediction_endpoint`
- `test_docker_validation_endpoint`

**Compile**

```bash
python -m py_compile .\tests\test_docker_container.py
```

**Run**

```bash
python -m pytest tests/test_docker_container.py -v
```

**Result**

```text
5 passed
```

These tests validate the Docker deployment boundary rather than duplicating the existing FastAPI application tests.

**Status: 5.4.6 – COMPLETED ✅**

---

### 5.4.7 — Add Docker Compose

### Why Docker Compose?

Docker Compose lets us define the complete container configuration in a single file and start it with one command instead of managing multiple Docker commands manually.

It also makes it easier to manage multiple services/containers together as the application grows.

**Implementation**

Created:
```text
docker-compose.yml
```

**Configuration**

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: nyc-taxi-eta-api:1.0
    container_name: nyc-taxi-eta-api
    ports:
      - "8000:8000"
```

Validated the Compose configuration:
```bash
docker compose config
```

Compose provides a simpler way to build and start the application:
```bash
docker compose up --build
```

**Port Mapping**

```text
Host :8000
    ↓
Container :8000
    ↓
Uvicorn
    ↓
FastAPI
```

**Status: 5.4.7 – COMPLETED ✅**

---

### 5.4.8 — Execute & Verify with Docker Compose

**Implementation**

Started the application using:
```bash
docker compose up --build
```

Verified the Compose-managed container:
```bash
docker compose ps
```

**Health Check**

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
```

Result:
```text
200 OK
```

**Prediction Test**

Sent the valid prediction payload to:

POST /predict


Result:
```text
200 OK
predicted_eta = 2020.20947265625
```

**Validation Test**

Sent an invalid request with:

pickup_month = 13


Result:
```text
422 Unprocessable Entity
```

**Status: 5.4.8 – COMPLETED ✅**

---

### Final Docker Compose Flow

```text
docker compose up --build
          │
          ▼
     Docker Image
          │
          ▼
     API Container
          │
          ▼
   Port 8000:8000
          │
          ▼
       FastAPI
          │
    ┌─────┴─────┐
    ▼           ▼
   GET        POST
    /       /predict
    │           │
   200      ┌───┴────┐
            ▼        ▼
          Valid    Invalid
           200       422
            │
            ▼
       ETA Prediction
```

---

### Phase 5.4 Status

**✅ COMPLETE**
Want to be notified when Claude responds?

## 5.5 — Test API

---

### 5.5.1 — Swagger UI

First first start the application with this command (Use PowerShell)
```text
docker compose up --build
```

**Verification**

Tested the deployed API using Swagger UI:

http://127.0.0.1:8000/docs


Verified:
- `GET /` → `200 OK`
- `POST /predict` with valid input → `200 OK`
- Prediction returned: `2020.20947265625`
- Invalid `pickup_month = 13` → `422 Unprocessable Entity`

**Status: 5.5.1 – COMPLETED ✅**

---

### 5.5.2 — Postman

**Implementation**

Created Postman collection:

ML-NYC-Taxi-ETA-API


Tested:
- `GET /` Health Check → `200 OK`
- `POST /predict` with valid input → `200 OK`
- `POST /predict` with `pickup_month = 13` → `422 Unprocessable Entity`

Valid prediction:
```text
predicted_eta = 2020.20947265625
```

**Status: 5.5.2 – COMPLETED ✅**

---

### 5.5.3 — cURL / PowerShell

**Verification**

Tested the containerized API from the host using PowerShell.

**GET `/`**

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/"
```

Result:
```text
200 OK
healthy
```

**POST `/predict`**

Sent the validated taxi feature payload. Use the same Payload what we used earlier:
```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Result:
```text
200 OK
predicted_eta = 2020.20947265625
```

**Invalid Request**

Tested:

pickup_month = 13


Result:
```text
422 Unprocessable Entity
```

Validation message:
```text
Input should be less than or equal to 12
```

**Status: 5.5.3 – COMPLETED ✅**

---

### 5.5 Completion

All API clients successfully verified the deployed prediction service:

```text
Swagger UI  → GET 200 | POST 200 | Validation 422
Postman     → GET 200 | POST 200 | Validation 422
PowerShell  → GET 200 | POST 200 | Validation 422
```

<a id="phase-56"></a>
## 5.6 — Publish Docker Image

The containerized application image was published to a public Docker Hub repository so that external users can run the application without setting up the Python environment or rebuilding the image locally.

---

### Docker Hub Repository

```text
avinashkumarb6/nyc-taxi-eta-prediction
```

### Published image:
```text
avinashkumarb6/nyc-taxi-eta-prediction:1.0
```

---

### Publish Workflow

The existing local image was tagged with the Docker Hub repository name:
```bash
docker tag nyc-taxi-eta-api:1.0 avinashkumarb6/nyc-taxi-eta-prediction:1.0
```

The image was then pushed to Docker Hub:
```bash
docker push avinashkumarb6/nyc-taxi-eta-prediction:1.0
```

---

### Purpose

Publishing the image provides a reproducible deployment artifact that can be consumed externally without:
- Installing Python dependencies
- Creating a virtual environment
- Rebuilding the Docker image
- Downloading or configuring the ML model separately

An external user can pull and run the published image directly:
```bash
docker pull avinashkumarb6/nyc-taxi-eta-prediction:1.0

docker run --rm -p 8000:8000 `
  avinashkumarb6/nyc-taxi-eta-prediction:1.0
```

API:

http://127.0.0.1:8000/docs


---

### External Reproducibility Verification

The final verification will simulate a clean external environment:

```text
Remove local image
        ↓
docker pull from Docker Hub
        ↓
docker run
        ↓
GET /
        ↓
POST /predict
        ↓
Verify prediction
```

This confirms that the publicly published image is independently runnable and contains all required application dependencies and model artifacts.

**Status: 5.6 – COMPLETED ✅**


**Phase 5 — Deployment (M4): COMPLETE ✅**








