# Phase 2 – Project Initialization

# 📑 Table of Contents

## Environment Setup
- [Step 1 – Create the Project Structure](#step-1--create-the-project-structure)
- [Step 2 – Initialize the Git Repository](#step-2--initialize-the-git-repository)
- [Step 3 – Create the Python Virtual Environment](#step-3--create-the-python-virtual-environment)
- [Step 4 – Activate the Virtual Environment](#step-4--activate-the-virtual-environment)

## Dependency Management
- [Step 5 – Create `requirements.txt`](#step-5--create-requirementstxt)
- [Step 6 – Install Project Dependencies](#step-6--install-project-dependencies)

## ML Engineering Tooling
- [Step 7 – Configure `.gitignore`](#step-7--configure-gitignore)
- [Step 8 – Initialize DVC](#step-8--initialize-dvc)
- [Step 9 – Configure MLflow](#step-9--configure-mlflow)
- [Step 10 – Finalize Development Environment](#step-10--finalize-development-environment)

  <details>
  <summary>What will be verified in this step?</summary>

  ```text
  ├── Verify Python
  ├── Verify Git
  ├── Verify DVC
  ├── Verify MLflow
  ├── Freeze Dependencies
  └── Phase 2 Complete ✅

---
# Step 1 – Create the Project Structure

## Objective

Create a clean and modular repository structure that will support the complete Machine Learning Engineering lifecycle.

---

## Why This Step?

A well-organized repository makes the project easier to understand, maintain, and extend.

Instead of placing all scripts in a single directory, the project is organized around the Machine Learning lifecycle, with each major stage assigned its own module.

This structure also encourages reusable code and clear separation of responsibilities.

---

## Implementation

Create the following project structure:

```text
NYC-Taxi-ETA-Prediction/

├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── features/
│   ├── training/
│   ├── inference/
│   ├── monitoring/
│   ├── api/
│   ├── pipelines/
│   └── utils/
│
├── scripts/
├── models/
├── docs/
├── reports/
├── tests/
├── configs/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── README.md
└── .gitignore
```

The folder structure was created using PowerShell.

---

## Verification

Verify that the folders have been created successfully.

```powershell
tree /A
```

The output should display the complete repository hierarchy.

---

## Discussion

The repository is intentionally organized around the Machine Learning lifecycle rather than individual assignments.

Each folder has a clearly defined responsibility:

- `data/` stores datasets.
- `src/` contains reusable source code.
- `scripts/` stores utility scripts.
- `docs/` contains project documentation.
- `reports/` stores project reports.
- `configs/` centralizes configuration files.

This modular organization improves maintainability and scalability.

---

## Checkpoint ✅

At the end of this step:

- Project repository created.
- Folder hierarchy established.
- Base project files created.

---

# Step 2 – Initialize the Git Repository

### Objective

Initialize a local Git repository to enable version control throughout the project.

---

### Why This Step?

Version control is a fundamental practice in Machine Learning Engineering.

Git allows us to:

- Track every change made to the project.
- Maintain a complete commit history.
- Collaborate effectively.
- Reproduce previous versions if needed.

The project brief also requires incremental commits that reflect weekly progress.

---

### Implementation

Initialize the repository:

```powershell
git init
```

(Optional) Rename the default branch:

```powershell
git branch -m main
```

Verify the repository status:

```powershell
git status
```

---

### Verification

Expected output:

```text
Initialized empty Git repository...
On branch main

No commits yet
```

Git should also display all project files as **Untracked Files**, indicating that the repository is ready for tracking.

---

### Discussion

At this stage, Git creates a hidden `.git` directory inside the project root.

This directory stores:

- Commit history
- Branch information
- Configuration
- Object database

None of the project files are modified.

Git simply begins tracking the project.

---

### Checkpoint ✅

At the end of this step:

- Git repository initialized.
- Default branch renamed to **main**.
- Repository ready for version control.

---

# Step 3 – Create the Python Virtual Environment

## Objective

Create an isolated Python environment dedicated to this project.

---

## Why This Step?

Different Python projects often require different versions of libraries.

Using a virtual environment ensures that all project dependencies remain isolated from the system-wide Python installation and from other Python projects.

This improves reproducibility and avoids dependency conflicts.

---

## Implementation

Create the virtual environment:

```powershell
python -m venv .venv
```

---

## Verification

Verify that the virtual environment has been created successfully.

```powershell
Get-ChildItem
```

The project root should now contain:

```text
.venv/
```

Inside `.venv`, Python creates its own isolated interpreter and package manager.

---

## Discussion

Creating a virtual environment does not install any project dependencies.

Instead, it creates an isolated Python installation containing:

- Python interpreter
- pip
- Site-packages directory
- Activation scripts

Every package installed later will be stored inside this virtual environment rather than the system Python.

---

## Checkpoint ✅

At the end of this step:

- Virtual environment created.
- Project-specific Python interpreter available.
- Project isolated from the system Python installation.

---

# Step 4 – Activate the Virtual Environment

### Objective

Activate the project-specific Python virtual environment.

---

### Why This Step?

Activating the virtual environment ensures that all Python packages are installed locally within the project, preventing conflicts with other Python projects on the same machine.

---

### Implementation

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the interpreter:

```powershell
where python
```

Verify Python and pip:

```powershell
python --version
pip --version
```

---

### Verification

The PowerShell prompt should display:

```text
(.venv)
```

The Python interpreter should resolve to:

```text
...\NYC-Taxi-ETA-Prediction\.venv\Scripts\python.exe
```

---

### Discussion

Activating the virtual environment updates the current terminal session to use the project's isolated Python interpreter and package manager.

---

### Checkpoint ✅

- Virtual environment activated.
- Python interpreter verified.
- Pip verified.

---

## Common Issue while Activating the Python Virtual Environment (in Windows PowerShell)

If PowerShell blocks virtual environment activation with:

```text
running scripts is disabled on this system
```

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

This changes the execution policy **only for the current PowerShell session** and is the recommended approach for local development.

---

# Step 5 – Create `requirements.txt`

## Objective

Create a centralized dependency file that defines all Python libraries required for the project.

---

## Why This Step?

A Machine Learning project typically depends on multiple third-party libraries for data processing, model training, experiment tracking, deployment, and monitoring.

Instead of installing packages manually every time, all required dependencies are listed in a single `requirements.txt` file.

This enables any developer to recreate the project environment using a single command.

---

## Implementation

Create a file named:

```text
requirements.txt
```

Add the project dependencies.

```text
# ==========================================================
# NYC Taxi ETA Prediction
# Machine Learning Engineering Project
# ==========================================================

# Core Data Processing
pandas
numpy
pyarrow

# Data Validation
pandera

# Visualization
matplotlib
plotly

# Machine Learning
scikit-learn
xgboost
lightgbm

# Hyperparameter Optimization
optuna

# Experiment Tracking
mlflow

# API
fastapi
uvicorn

# Model Serialization
joblib

# Dataset Versioning
dvc

# Monitoring
evidently

# Notebook
jupyter
ipykernel
```

---

## Note

At this stage, the dependency file contains only the required package names.

After completing the project setup and verifying the development environment, the exact package versions will be captured using:

```powershell
pip freeze > requirements.txt  # This will be done later.
```

This ensures that anyone cloning the repository can recreate the exact same software environment.

---

## Discussion

The dependencies are organized by functionality rather than listed alphabetically.

This makes the file easier to understand and maintain as the project grows.

The selected libraries directly support the project's architecture, covering:

- Data Engineering
- Machine Learning
- Experiment Tracking
- Deployment
- Monitoring

---

## Checkpoint ✅

At the end of this step:

- `requirements.txt` created.
- All project dependencies documented.
- Ready for installation.

---

# Step 6 – Install Project Dependencies

## Objective

Install all required Python libraries inside the project's virtual environment.

---

## Why This Step?

The project depends on several open-source libraries for data processing, validation, machine learning, experiment tracking, deployment, and monitoring.

Installing these dependencies inside the virtual environment ensures that the project remains isolated, reproducible, and independent of the system-wide Python installation.

---

## Implementation

Install all project dependencies using the following command:

```powershell
pip install -r requirements.txt
```

This command reads the `requirements.txt` file and installs every listed package into the active virtual environment.

---

## Verification

After installation, verify that the key libraries are available by importing them and checking their versions.

Example:

```powershell
python -c "import pandas; print(pandas.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import mlflow; print(mlflow.__version__)"
python -c "import dvc; print(dvc.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
python -c "import pandera; print(pandera.__version__)"
python -c "import evidently; print(evidently.__version__)"
```

Example output:

```text
Pandas        : 2.3.3
Scikit-learn : 1.9.0
MLflow       : 3.15.1
DVC          : 3.67.1
FastAPI      : 0.141.1
Pandera      : 0.32.1
Evidently    : 0.7.21
```

Successful version output confirms that the packages have been installed correctly inside the virtual environment.

---

## Discussion

Although `pip install` reported a successful installation, importing the major libraries provides an additional validation that the environment has been configured correctly.

This verification step helps identify missing packages or installation issues before development begins.

---

## Checkpoint ✅

At the end of this step:

- Project dependencies installed successfully.
- Core libraries verified.
- Development environment ready for project configuration.

---

# Step 7 – Configure `.gitignore`

## Objective

Configure Git to ignore generated, machine-specific, and temporary files so that only essential project files are tracked.

---

## Why This Step?

Many files created during development can be regenerated automatically and should not be committed to version control.

Using a `.gitignore` file keeps the repository clean, reduces unnecessary changes, and improves collaboration.

---

## Implementation

Create a `.gitignore` file in the project root and add the required ignore rules.

The configuration excludes:

- Python cache
- Virtual environment
- IDE settings
- Operating system files
- Jupyter checkpoints
- DVC cache
- MLflow artifacts
- Log files

---

## Verification

Run:

```powershell
git status
```

Verify that ignored files (such as `.venv/`) no longer appear in the list of untracked files.

---

## Discussion

The `.gitignore` file ensures that only source code, configuration files, and documentation are committed to the repository.

Generated files remain local and can be recreated whenever required.

---

### Files Created

- `.gitignore`

---

## Checkpoint ✅

At the end of this step:

- Git ignore rules configured.
- Generated files excluded from version control.
- Repository configured for version control and ready for further development.

---

# Step 8 – Initialize DVC

## Objective

Initialize DVC (Data Version Control) in the project to enable dataset versioning and reproducible Machine Learning workflows.

---

## Why This Step?

Git is designed to version source code, configuration files, and documentation. It is not optimized for managing large datasets or machine learning artifacts.

DVC complements Git by providing a version control mechanism for datasets and models while allowing Git to continue managing the source code.

Initializing DVC prepares the project for future dataset tracking without changing any existing files.

---

## Implementation

Initialize DVC by running:

```powershell
dvc init
```

Expected output:

```text
Initialized DVC repository.
```

---

## Verification

Verify that DVC created the required project files:

```powershell
Get-ChildItem -Force
```

Expected additions:

```text
.dvc/
.dvcignore
```

Inspect the DVC directory:

```powershell
Get-ChildItem .dvc
Get-ChildItem .dvc -Recurse
```

Initial structure:

```text
.dvc/
│
├── config
├── .gitignore
└── tmp/
    └── btime
```

---

## Understanding the Generated Files

### `.dvc/`

The `.dvc` directory is the internal working directory used by DVC.

It stores:

- DVC configuration
- Cache information
- Remote storage configuration (when configured)
- Pipeline metadata (later in the project)

Initially, the directory is intentionally minimal because no datasets have been versioned yet.

---

### `config`

The `config` file stores project-level DVC configuration.

Immediately after initialization, it is empty because no remote storage or cache settings have been configured.

As the project evolves, this file will store settings such as:

- Local cache configuration
- Azure Blob Storage
- Amazon S3
- Google Cloud Storage
- Other DVC remotes

---

### `.dvc/.gitignore`

This file tells Git which internal DVC files should remain untracked.

It is similar in purpose to the project's root `.gitignore`, but applies only to DVC's internal working directory.

---

### `tmp/`

A temporary workspace used internally by DVC while executing operations such as:

- `dvc add`
- `dvc pull`
- `dvc push`
- `dvc repro`

No user interaction is required with this directory.

---

## Discussion

Running `dvc init` does **not** version any datasets.

It simply prepares the repository for future dataset tracking.

Datasets will begin to be version-controlled only after commands such as:

```powershell
dvc add data/raw/  # Which we will run later
```

are executed during the Data Engineering phase.

---

### Files Created

- `.dvc/`
- `.dvcignore`

---

## Checkpoint ✅

At the end of this step:

- DVC initialized successfully.
- Project ready for dataset versioning.
- Repository prepared for reproducible data pipelines.

---

## Engineering Note 💡

A common misconception is that DVC replaces Git.

In reality, Git and DVC complement each other, with each tool having a clearly defined responsibility.

| Tool | Primary Responsibility |
|-------|------------------------|
| **Git** | Version control for source code, configuration files, and documentation |
| **DVC** | Version control for datasets and machine learning artifacts |
| **MLflow** | Experiment tracking, metrics, parameters, and model registry |
| **Docker** | Package the application and its runtime environment |
| **FastAPI** | Expose trained models through REST APIs |
| **Evidently** | Monitor model performance and detect data drift in production |

The strength of a modern ML Engineering project comes from combining these specialized tools into a single end-to-end workflow rather than expecting one tool to solve every problem.

---

# Step 9 – Configure MLflow

## Objective

Configure **MLflow** as the experiment tracking framework for the project.

MLflow enables reproducible Machine Learning by automatically recording every training experiment, including model parameters, evaluation metrics, artifacts, and trained models.

---

## Why Do We Need MLflow?

During model development, multiple experiments are performed while trying different algorithms, feature sets, and hyperparameters.

Without an experiment tracking system, it quickly becomes difficult to answer questions such as:

- Which model performed the best?
- Which hyperparameters produced the lowest RMSE?
- Which dataset version was used?
- Which model was finally deployed?

MLflow automatically answers these questions by recording every experiment.

Instead of manually maintaining spreadsheets or naming models like:

```text
model.pkl
model_final.pkl
model_final_latest.pkl
model_best.pkl
```

MLflow stores all experiment information in a structured and searchable format.

---

## What Does MLflow Track?

For every training run, MLflow records:

- Experiment Name
- Parameters
- Evaluation Metrics
- Model Artifacts
- Training Timestamp
- Source Code Version (when configured)

This makes every experiment reproducible and easy to compare.

---

## Implementation

### Step 1 – Verify MLflow Installation

Verify the installed version:

```powershell
mlflow --version
```

Example output:

```text
mlflow, version 3.15.1
```

---

### Step 2 – Start the MLflow Tracking Server

Launch the local MLflow UI:

```powershell
mlflow ui
```

Expected output:

```text
INFO: Uvicorn running on http://127.0.0.1:5000
```

Leave this terminal running.

---

### Step 3 – Open the MLflow Dashboard

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

The MLflow dashboard should open successfully.

Initially, only the default experiment is visible because no model training has been performed yet.

---

## Working with Multiple Terminals

After starting the MLflow UI, the first terminal remains occupied by the MLflow server.

Open a **second PowerShell terminal** for the remaining development work.

### Step 1 – Activate the Virtual Environment

Run:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you encounter the following error:

```text
...Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

Run the following command to temporarily allow PowerShell script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

> **Note**
>
> This command changes the execution policy **only for the current PowerShell session**.
> It does **not** permanently modify your system's execution policy.

Now activate the virtual environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Expected prompt:

```text
(.venv) PS D:\NYC-Taxi-ETA-Prediction>
```

---

### Typical Development Workflow

```text
Terminal 1
────────────────────────────
(.venv)

mlflow ui

Runs continuously

────────────────────────────

Terminal 2
────────────────────────────
(.venv)

Python Scripts

Git

DVC

Model Training

Data Engineering

Feature Engineering

etc.
```

The MLflow Tracking Server continues running in the first terminal while all development work is performed from the second terminal.

---

## Understanding the Generated Files

### `mlflow.db`

Unlike older MLflow versions, MLflow **3.x** creates a local SQLite database named:

```text
mlflow.db
```

This database stores experiment metadata such as:

- Experiments
- Training Runs
- Parameters
- Metrics
- Model Metadata
- Tags

This is the default backend store when no backend URI is configured.

---

### Why Not `mlruns/`?

Many tutorials reference an `mlruns/` directory.

That behavior belongs to older MLflow versions.

Since this project uses **MLflow 3.15.1**, experiment metadata is stored inside the SQLite database (`mlflow.db`) by default.

Both approaches serve the same purpose; only the storage backend differs.

---

## Update `.gitignore`

Since `mlflow.db` is a generated local artifact, it should not be committed to Git.

Update `.gitignore`:

```gitignore
# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------

# MLflow 3.x local tracking database
mlflow.db

# Older MLflow versions
mlruns/
```

---

## Verification

Verify the following:

- MLflow installed successfully.
- MLflow UI started without errors.
- Browser dashboard accessible.
- `mlflow.db` created in the project root.
- `mlflow.db` ignored by Git.

Run:

```powershell
git status
```

The file `mlflow.db` should **not** appear in the list of untracked files.

---

## Checkpoint ✅

At the end of this step:

- MLflow configured successfully.
- Local experiment tracking server running.
- Dashboard accessible through the browser.
- Local SQLite backend initialized.
- Git configured to ignore generated MLflow metadata.

---

# Step 10 – Finalize Development Environment

## Objective

Perform a final verification of the development environment before beginning the Data Engineering phase.

This step ensures that all core tools have been installed, configured, and verified successfully.

---

## Why This Step?

Before writing any data processing or model training code, it is important to verify that the entire development environment is healthy.

Identifying configuration issues at this stage prevents interruptions during later phases of the project.

Think of this as the **Pre-Flight Checklist** before starting development.

---

# Verification Checklist

## 1. Verify Python Environment

Verify that the virtual environment is active.

```powershell
python --version
```

Example output:

```text
Python 3.12.x
```

Verify pip:

```powershell
pip --version
```

Expected result:

- Python available
- pip available
- Virtual environment active

✅ Status: Passed

---

## 2. Verify Git

Verify Git installation:

```powershell
git --version
```

Verify repository status:

```powershell
git status
```

Expected result:

- Git initialized
- Repository accessible
- Source files tracked correctly

✅ Status: Passed

---

## 3. Verify DVC

Verify DVC installation:

```powershell
dvc version
```

Verify DVC repository:

```powershell
Get-ChildItem .dvc
```

Expected result:

- DVC installed
- `.dvc` directory present
- `.dvcignore` created

✅ Status: Passed

---

## 4. Verify MLflow

Verify MLflow installation:

```powershell
mlflow --version
```

Verify MLflow UI:

```powershell
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

Expected result:

- MLflow dashboard accessible
- SQLite backend (`mlflow.db`) created
- `mlflow.db` ignored by Git

✅ Status: Passed

---

## 5. Freeze Project Dependencies

Record the exact package versions installed in the virtual environment.

Generate a frozen dependency list:

```powershell
pip freeze > requirements-lock.txt
```

Example:

```text
pandas==2.3.3
scikit-learn==1.9.0
mlflow==3.15.1
dvc==3.67.1
...
```

> **Note**
>
> Throughout development, `requirements.txt` remains maintained dependency list by us.
>
> At project completion, `requirements-lock.txt` provides the exact package versions required to reproduce the environment.

---

## Final Checklist

| Component | Status |
|-----------|:------:|
| Python Environment | ✅ |
| Git Repository | ✅ |
| Virtual Environment | ✅ |
| Project Dependencies | ✅ |
| DVC | ✅ |
| MLflow | ✅ |
| Project Structure | ✅ |

---

## Phase 2 Summary

By completing Phase 2, the project now has:

- A reproducible Python environment.
- Git version control.
- Data Version Control (DVC).
- MLflow experiment tracking.
- Organized project structure.
- Dependency management.
- Ready-to-use development environment.

The project is now fully prepared for Data Engineering.

---

## Phase 2 Complete ✅

The development environment has been successfully initialized.

The project is now ready to begin **Phase 3 – Data Engineering**, where the NYC Taxi dataset will be ingested, validated, explored, and prepared for feature engineering.
