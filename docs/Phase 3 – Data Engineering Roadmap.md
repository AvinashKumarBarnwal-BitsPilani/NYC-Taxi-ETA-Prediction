# 📊 Phase 3 – Data Engineering

## Table of Contents

- [Phase 3A - Data Understanding & Validation](./Phase%203A%20-%20Data%20Understanding-Validation.md)
  - [Step 1 - Understand the Dataset](./Phase%203A%20-%20Data%20Understanding-Validation.md#step-1--understand-the-dataset)
    - [1.1 Inventory the Raw Dataset](./Phase%203A%20-%20Data%20Understanding-Validation.md#11--inventory-the-raw-dataset)
    - [1.2 Understand the Dataset Schema](./Phase%203A%20-%20Data%20Understanding-Validation.md#12--understand-the-dataset-schema)
    - [1.3 Inspect Data Types](./Phase%203A%20-%20Data%20Understanding-Validation.md#13--inspect-data-types)
    - [1.4 Initial Missing-Value Check](./Phase%203A%20-%20Data%20Understanding-Validation.md#14--initial-missing-value-check)
    - [1.5 Initial Statistics & Cardinality](./Phase%203A%20-%20Data%20Understanding-Validation.md#15--initial-statistics--cardinality)
    - [1.6 Complete Dataset Row Counts](./Phase%203A%20-%20Data%20Understanding-Validation.md#16--complete-dataset-row-counts)
    - [1.7 Train vs Test Schema](./Phase%203A%20-%20Data%20Understanding-Validation.md#17--train-vs-test-schema)
    - [1.8 Complete Missing-Value Check](./Phase%203A%20-%20Data%20Understanding-Validation.md#18--complete-missing-value-check)
    - [1.9 Complete Target Analysis](./Phase%203A%20-%20Data%20Understanding-Validation.md#19--complete-target-analysis)
    - [1.10 Datetime Range & Consistency](./Phase%203A%20-%20Data%20Understanding-Validation.md#110--datetime-range--consistency)
    - [1.11 Create & Validate 10% Development Dataset](./Phase%203A%20-%20Data%20Understanding-Validation.md#111--create--validate-10-development-dataset)
  
  - [Step 2 - Define the Data Contract & Validation Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#step-2--define-the-data-contract--validation-rules)
    - [2.1 Define Expected Schema/Columns](./Phase%203A%20-%20Data%20Understanding-Validation.md#21--define-expected-schemacolumns)
    - [2.2 Define Required Columns](./Phase%203A%20-%20Data%20Understanding-Validation.md#22--define-required-columns)
    - [2.3 Define Expected Data Types](./Phase%203A%20-%20Data%20Understanding-Validation.md#23--define-expected-data-types)
    - [2.4 Define Categorical Value Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#24--define-categorical-value-rules)
    - [2.5 Define Numeric Range Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#25--define-numeric-range-rules)
    - [2.6 Define Datetime Consistency Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#26--define-datetime-consistency-rules)
    - [2.7 Define Target Validation Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#27--define-target-validation-rules)
    - [2.8 Define Missing-Value Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#28--define-missing-value-rules)
    - [2.9 Define Duplicate Handling Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#29--define-duplicate-handling-rules)
    - [2.10 Define Invalid-Record Rules](./Phase%203A%20-%20Data%20Understanding-Validation.md#210--define-invalid-record-rules)
    - [2.11 Create Data Validation Contract](./Phase%203A%20-%20Data%20Understanding-Validation.md#211--create-data-validation-contract)
    - [2.12 Validate the Contract Against the Dataset](./Phase%203A%20-%20Data%20Understanding-Validation.md#212--validate-the-contract-against-the-dataset)

- [Phase 3B - Data Quality Analysis & Cleaning](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md)
  - [Step 3 - Data Quality Analysis](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#31--analyze-missing-values)
    - [3.1 Analyze Missing Values](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#31--analyze-missing-values)
    - [3.2 Analyze Duplicate Records](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#32--analyze-duplicate-records)
    - [3.3 Analyze Invalid Values](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#33--analyze-invalid-values)
    - [3.4 Analyze Outliers](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#34--analyze-outliers)
    - [3.5 Analyze Impossible Timestamps](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#35--analyze-impossible-timestamps)
    - [3.6 Analyze Invalid Geographical Coordinates](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#36--analyze-invalid-geographical-coordinates)
    - [3.7 Analyze Unexpected Categorical Values](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#37--analyze-unexpected-categorical-values)
    - [3.8 Analyze Target Distribution](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#38--analyze-target-distribution)
    - [3.9 Analyze Extremely Short Trips](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#39--analyze-extremely-short-trips)
    - [3.10 Analyze Extremely Long Trips](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#310--analyze-extremely-long-trips)
    - [3.11 Analyze Unrealistic Passenger Counts](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#311--analyze-unrealistic-passenger-counts)
    - [3.12 Data Quality Report](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#312--data-quality-report)
    - [Visual Data Quality Analysis](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#visual-data-quality-analysis)
   
  - [Step 4 - Data Cleaning](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#step-4--data-cleaning)
    - [4.1 Handle Missing Values](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#41--handle-missing-values)
    - [4.2 Handle Duplicates](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#42--handle-duplicates)
    - [4.3 Handle Invalid Records](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#43--handle-invalid-records)
    - [4.4 Handle Impossible Timestamps](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#44--handle-impossible-timestamps)
    - [4.5 Handle Invalid Coordinates](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#45--handle-invalid-coordinates)
    - [4.6 Handle Invalid Passenger Counts](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#46--handle-invalid-passenger-counts)
    - [4.7 Handle Invalid Target Values](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#47--handle-invalid-target-values)
    - [4.8 Handle Justified Outliers](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#48--handle-justified-outliers)
    - [4.9 Ensure Consistent Data Types](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#49--ensure-consistent-data-types)
    - [4.10 Verify Cleaned Dataset](./Phase%203B%20%E2%80%93%20Data-Quality-Analysis-Cleaning.md#410--verify-cleaned-dataset)

- [Phase 3C - Feature Engineering & ML-Ready Pipeline](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md)
  - [Step 5 - Feature Engineering](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-51--understand-feature-engineering-input)
    - [5.1 Understand Feature Engineering Input](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-51--understand-feature-engineering-input)
    - [5.2 Identify Prediction-Time Features](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-52--identify-prediction-time-features)
    - [5.3 Create Datetime Features](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-53--create-datetime-features)
    - [5.4a Inspect Geographic Inputs](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-54a--inspect-geographic-inputs)
    - [5.4b Create Haversine Distance](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-54b--create-haversine-distance)
    - [5.4c Analyze Geographic Feature](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-54c--analyze-geographic-feature)
    - [5.5 Evaluate Existing Features](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-55--evaluate-existing-features)
    - [5.6 Evaluate/Create Rush-Hour Feature](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-56--evaluatecreate-rush-hour-feature)
    - [5.7 Check Feature Quality](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-57--check-feature-quality)
    - [5.8 Check Feature Leakage](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-58--check-feature-leakage)
    - [5.9 Feature Engineering Summary](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-59--feature-engineering-summary)

  - [Step 6 - Train / Validation Split & Preprocessing](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-6--train--validation-split--preprocessing)
    - [6.1 Define Train/Validation Split Strategy](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-61--define-trainvalidation-split-strategy)
    - [6.2 Implement Train/Validation Split](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-62--implement-trainvalidation-split)
    - [6.3 Feature Scaling & Normalization](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-63--feature-scaling--normalization)
    - [6.4 Encoding Categorical Features](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-64--encoding-categorical-features)
    - [6.5 Verify Preprocessed Datasets](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-65--verify-preprocessed-datasets)

  - [Step 7 - Build the Data Engineering Code Pipeline](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-7--build-the-data-engineering-code-pipeline)
    - [7.1 Design Modular Pipeline Architecture](./Phase%203C%20%E2%80%93%20Feature%20Engineering%20%26%20ML-Ready%20Pipeline.md#step-71--design-modular-pipeline-architecture)

---

# Step 7 – Build the Data Engineering Code Pipeline

**Goal:** Move the data-engineering logic from exploratory notebooks into reusable production-style Python modules.

### Tasks

Create modular code for:

```text
data loading
     ↓
validation
     ↓
cleaning
     ↓
feature engineering
     ↓
preprocessing
```

Potential structure:

```text
src/
├── data/
│   ├── load.py
│   ├── validate.py
│   ├── clean.py
│   ├── features.py
│   └── preprocess.py
```

The exact structure can be adjusted based on the existing project architecture.

### Important

Notebooks should primarily be used for:

```text
Exploration
Visualization
Experimentation
```

Reusable pipeline logic should live under:

```text
src/
```

### Expected Output

A clean, modular Data Engineering implementation that can run without manually executing notebook cells.

---

# Step 8 – Build the DVC Pipeline

**Goal:** Make the complete data-processing workflow reproducible.

This is where the work from Phase 2 with DVC becomes useful.

### Desired Flow

```text
Raw Data
   ↓
Data Validation
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Processed Dataset
```

DVC should track the pipeline and its dependencies.

### Tasks

- [ ] Decide what data should be DVC-tracked
- [ ] Track the appropriate raw dataset with DVC
- [ ] Define pipeline stages in `dvc.yaml`
- [ ] Define dependencies
- [ ] Define outputs
- [ ] Define parameters/configuration where required
- [ ] Run the pipeline
- [ ] Verify generated outputs
- [ ] Verify DVC status
- [ ] Verify reproducibility

### Expected Structure

Conceptually:

```text
dvc.yaml

Stage 1
Data Validation

Stage 2
Data Cleaning

Stage 3
Feature Engineering

Stage 4
Preprocessing
```

The exact number of stages can be simplified if appropriate.

### Expected Output

A working:

```text
dvc.yaml
```

that allows us to reproduce the processed dataset from the tracked raw data.

---

# Step 9 – Testing & Reproducibility

**Goal:** Make sure the pipeline is reliable before handing the data to Phase 4.

### Tasks

Create basic tests for important assumptions.

Examples:

- [ ] Required columns exist
- [ ] No invalid target values
- [ ] No impossible timestamps
- [ ] Expected data types are maintained
- [ ] Feature engineering produces expected columns
- [ ] No unexpected nulls remain
- [ ] Train/validation datasets are generated correctly
- [ ] Pipeline executes successfully
- [ ] Pipeline can be reproduced

### Reproducibility Test

The most important test:

```text
Delete generated processed data
        ↓
Run DVC pipeline again
        ↓
Processed data generated successfully
        ↓
Same pipeline → same result
```

### Project Location

```text
tests/
```

### Expected Output

A validated and reproducible Data Engineering pipeline.

---

# Step 10 – Phase 3 Integration & Handover

**Goal:** Prepare everything required by Phase 4 – Model Development.

### Tasks

- [ ] Final quality check of processed dataset
- [ ] Verify feature list
- [ ] Verify target column
- [ ] Verify train/validation datasets
- [ ] Verify preprocessing pipeline
- [ ] Verify DVC pipeline
- [ ] Verify tests
- [ ] Update documentation
- [ ] Clean notebooks / temporary files
- [ ] Commit changes
- [ ] Push feature branch
- [ ] Open Pull Request
- [ ] Explain outputs and usage to Phase 4 contributor(s)

### Phase 4 should receive

```text
Processed Dataset
        +
Feature Definitions
        +
Preprocessing Pipeline
        +
DVC Pipeline
        +
Data Validation Rules
        +
Documentation
```

### Expected Output

A **Model-Ready Data Engineering Layer**.

Phase 4 should be able to start model development without needing to redo Phase 3 work.

---

# 🎯 Phase 3 Final Deliverables

By **23 August**, Phase 3 should provide:

- [ ] Dataset understanding / EDA
- [ ] Data validation rules
- [ ] Data quality report
- [ ] Cleaning pipeline
- [ ] Feature engineering pipeline
- [ ] Train / validation split strategy
- [ ] Preprocessing pipeline
- [ ] Modular Python implementation
- [ ] DVC-tracked data
- [ ] Working `dvc.yaml`
- [ ] Basic automated tests
- [ ] Reproducible data pipeline
- [ ] Clean processed dataset
- [ ] Documentation for Phase 4

---

# 🏁 Definition of Done

Phase 3 is considered complete when:

```text
Raw Dataset
     │
     ▼
Validated
     │
     ▼
Cleaned
     │
     ▼
Feature Engineered
     │
     ▼
Preprocessed
     │
     ▼
DVC Pipeline
     │
     ▼
Reproducible Model-Ready Dataset
     │
     ▼
Phase 4 can start
```

The extra day is intentionally kept as a buffer for integration, fixes, documentation, and final submission.
