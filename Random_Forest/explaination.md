# Clinical Disease Risk Prediction Pipeline

An end-to-end Machine Learning pipeline designed to predict disease risk (specifically Diabetes) using preprocessed Electronic Health Record (EHR) data. 

This project addresses key healthcare ML challenges—such as handling class imbalance with **SMOTE**, avoiding data leakage, prioritizing clinical sensitivity (**Recall**) over raw accuracy, and ensuring feature-order alignment for single-patient diagnostic predictions.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Workflow & Implementation](#workflow--implementation)
3. [Key Measures Taken & Clinical Considerations](#key-measures-taken--clinical-considerations)
4. [Understanding Random Forest](#understanding-random-forest)
5. [Tech Stack & Techniques](#tech-stack--techniques)
6. [Model Evaluation & Confusion Matrix](#model-evaluation--confusion-matrix)
7. [Visualizations & Plots](#visualizations--plots)
8. [Setup & Usage](#setup--usage)

---

## Project Overview

In clinical healthcare analytics, model evaluation goes beyond simple accuracy metrics. Datasets often exhibit severe class imbalances, and missing a positive disease case (**False Negative**) carries severe medical consequences. 

The primary objective of this project is to build, evaluate, and interpret a reliable clinical machine learning pipeline that preprocesses EHR data, trains baseline and ensemble classifiers, and provides an interactive patient risk evaluator aligned with trained feature structures.

---

## Workflow & Implementation

### 1. Data Preprocessing & Cleaning
* Verified dataset integrity and checked for missing values.
* Encoded categorical attributes into numeric representations required by Scikit-Learn classifiers:
  * **Gender:** Mapped `'Male'` $\rightarrow 0$ and `'Female'` $\rightarrow 1$.
  * **Smoking History:** Mapped categories (`'never'`, `'No Info'`, `'former'`, `'current'`) to ordinal values (`0`, `1`, `2`).

### 2. Train-Test Splitting
* Separated independent feature variables ($X$) from the target column ($Y = \text{diabetes}$).
* Performed an 80/20 train-test split (`test_size=0.20`, `random_state=42`) to evaluate model generalization on unseen patient data.

### 3. Feature Alignment & Interactive Predictor
* Built a terminal-based interactive clinical prediction function (`predict_diabetes()`).
* Ensured strict column sequence alignment between user inputs and the training feature matrix:
  $$\text{Feature Order: } [\text{gender}, \text{age}, \text{hypertension}, \text{heart\_disease}, \text{smoking\_history}, \text{bmi}, \text{HbA1c\_level}, \text{blood\_glucose\_level}]$$

### 4. Cross-Validation & Metric Evaluation
* Implemented **Stratified $K$-Fold Cross-Validation** to ensure every fold maintains equal target class proportions.
* Evaluated models across clinical metrics: Sensitivity/Recall, Precision, F1-Score, and Confusion Matrix[cite: 1].

---

## Key Measures Taken & Clinical Considerations

* **Preventing Data Leakage with SMOTE:** Applied Synthetic Minority Over-sampling Technique (SMOTE) strictly to the training split ($X_{\text{train}}, Y_{\text{train}}$) after performing the train-test split. Applying SMOTE before splitting introduces synthetic information derived from test records into training, leading to artificially inflated, unreliable metrics.
* **Prioritizing Sensitivity (Recall):** In disease screening, **False Negatives are catastrophic** because a missed diagnosis delays life-saving medical intervention. The pipeline prioritizes maximizing Recall to identify as many true positive cases as possible.
* **Feature Schema Matching:** Standardized runtime inputs into a `pandas.DataFrame` schema matching `X.columns` to prevent feature swapping errors during single-patient predictions[cite: 1].

---

## Understanding Random Forest

### What is a Random Forest?
A **Random Forest** is an ensemble learning algorithm that combines multiple **Decision Trees** to construct a robust, high-performing model.

Instead of relying on a single decision tree (which is prone to overfitting and sensitive to noise), Random Forest uses **Bagging (Bootstrap Aggregating)** and **Random Feature Subsampling**:

1. **Bootstrap Sampling:** Creates multiple subsets of the original data with replacement; each tree trains on a distinct subset.
2. **Random Feature Selection:** Evaluates only a random subset of features at each split node.
3. **Majority Voting:** Aggregates predictions across all individual trees; the majority vote determines the final patient prediction.

---

## Tech Stack & Techniques

* **Language:** Python (v3.12+)[cite: 1]
* **Data Manipulation:** Pandas, NumPy[cite: 1]
* **Machine Learning:** Scikit-Learn (`sklearn`)[cite: 1]
* **Data Resampling:** Imbalanced-Learn (`imblearn`)
* **Visualization:** Matplotlib, Seaborn

### Models Trained
* **Logistic Regression:** Baseline interpretable linear model where coefficients correspond to clinical odds ratios.
* **Random Forest Classifier (`n_estimators=100`):** Ensemble tree model capable of learning non-linear, high-order feature interactions (e.g., age, BMI, and glucose combinations)[cite: 1].

---

## Model Evaluation & Confusion Matrix

### Confusion Matrix Breakdown
The **Confusion Matrix** evaluates actual ground-truth patient labels against model predictions[cite: 1]:

| | **Predicted: Healthy (0)** | **Predicted: Diabetic (1)** |
| :--- | :--- | :--- |
| **Actual: Healthy (0)** | **True Negative (TN)** <br> *(Correctly identified healthy)* | **False Positive (FP)** <br> *(False Alarm)* |
| **Actual: Diabetic (1)** | **False Negative (FN)** <br> *(Missed sick patient)* | **True Positive (TP)** <br> *(Correctly identified sick)* |

#### Matrix Results
```python
[[166,  21],
 [ 19, 162]]
