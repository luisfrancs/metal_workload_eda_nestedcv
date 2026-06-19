# Assessing Mental Workload in Office Environments using Electrodermal Activity (EDA)

**Course:** 01HXDIU - Machine Learning in Healthcare: From Theory to Practice
**Author:** PhD Luis Sigcha
**Date:** 19/05/2026

---

# Overview

This repository contains the code used to assess **mental workload in office environments** using **Electrodermal Activity (EDA)** signals. The project implements a machine learning pipeline based on **Random Forest classifiers** and evaluates model performance using **nested subject-wise cross-validation (GroupKFold)** to ensure that data from the same participant never appear in both the training and testing sets.

The repository provides a complete machine learning workflow, including feature preprocessing, model training, hyperparameter optimization, and performance evaluation.

---

# Machine Learning Pipeline

The baseline model consists of the following stages:

* Feature standardization using `StandardScaler`
* Random Forest classifier
* Hyperparameter optimization using Grid Search
* Nested subject-wise cross-validation with `GroupKFold`

The baseline pipeline is defined as:

```python
def baseline_pipeline_factory():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier())
    ])
    return pipe
```

---

# Hyperparameter Search

The Random Forest hyperparameters are optimized using nested cross-validation.

## Parameter Grid

```python
baseline_param_grid = {
    "clf__n_estimators": [100, 150, 200],
    "clf__max_depth": [3, 5],
    "clf__min_samples_split": [20, 25, 30]
}
```

---

# Cross-Validation Strategy

Model evaluation follows a **nested subject-wise cross-validation** strategy.

* **Outer loop:** GroupKFold
* **Inner loop:** GroupKFold for hyperparameter optimization
* Subjects are used as grouping variables.
* Samples from the same participant are never split between training and testing folds.
* Hyperparameters are selected exclusively using the training folds.

This approach provides an unbiased estimate of model performance on unseen participants while preventing subject information leakage.

---

# Model Training

The baseline model is trained using the `nested_grouped_training_pipeline()` function.

```python
results_bs, models_bs, metrics_source_bs, metrics_target_bs, target_split_bs = \
nested_grouped_training_pipeline(
    X_source,
    y_source,
    idLOSO_source,
    X_target,
    y_target,
    idLOSO_target,
    pipeline_factory=baseline_pipeline_factory,
    param_grid=baseline_param_grid,
    n_splits_outer=n_splits_outer,
    n_splits_inner=n_splits_inner,
    target_test_size=0.3
)
```

The pipeline performs:

* Subject-wise data partitioning using GroupKFold
* Nested hyperparameter optimization
* Random Forest model training
* Performance evaluation on the source and target datasets
* Storage of the best-performing model from each outer fold

---

# Methods

This repository includes:

* Random Forest baseline classifier
* Nested subject-wise GroupKFold cross-validation
* Hyperparameter optimization using Grid Search
* Performance evaluation on source and target datasets

---

# Requirements

The project requires the following Python packages:

* Python ≥ 3.10
* NumPy
* Pandas
* Scikit-learn
* SciPy
* Matplotlib
* Joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Reproducibility

To reproduce the experiments:

1. Prepare the source and target datasets.
2. Load the extracted EDA features.
3. Define the subject identifiers used for grouped cross-validation.
4. Configure the parameter grid.
5. Execute `nested_grouped_training_pipeline()`.

The pipeline automatically performs nested cross-validation, hyperparameter optimization, model training, and performance evaluation.

The outputs include:

* Best model for each outer fold
* Selected hyperparameters
* Source-domain performance metrics
* Target-domain performance metrics
* Predictions on the target test sets
