# 📈 Simple Linear Regression - Machine Learning Project

This repository contains a complete hands-on implementation of **Simple Linear Regression** from scratch using Python, Pandas, and Scikit-Learn. The goal is to predict continuous numerical values (such as Exam Scores or Salaries) based on a single input feature.

---

## 📌 Table of Contents
- [What is Simple Linear Regression?](#-what-is-simple-linear-regression)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Workflow](#-project-workflow)
- [Model Performance](#-model-performance)

---

## 🧠 What is Simple Linear Regression?

Simple Linear Regression is a supervised learning algorithm used to model the relationship between:
* **One Independent Feature (X):** The input/predictor (e.g., `StudyHours` or `YearsExperience`).
* **One Dependent Target (Y):** The output/continuous outcome (e.g., `ExamScore` or `Salary`).

### The Mathematical Equation:
$$\hat{y} = \beta_0 + \beta_1 x$$

Where:
* $\hat{y}$ = Predicted target output
* $x$ = Input feature
* $\beta_0$ = Intercept (Base value of $y$ when $x = 0$)
* $\beta_1$ = Slope / Coefficient (Change in $y$ for every 1 unit change in $x$)

---


## 📊 Evaluation Metrics

To evaluate regression performance, we use:

| Metric | Formula / Description | Why It Matters |
| :--- | :--- | :--- |
| **MSE** (Mean Squared Error) | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes larger errors by squaring them. |
| **RMSE** (Root Mean Squared Error) | $\sqrt{\text{MSE}}$ | Brings error back to original target units for easy interpretation. |
| **$R^2$ Score** | 1 - (SS_res / SS_tot) | Explains the percentage of variance captured by the model (0 to 1). |

---

## 🛠️ Project Workflow

1. **Data Loading:** Load dataset using `pandas.read_csv()`.
2. **Feature & Target Selection:** Define $X$ as a 2D DataFrame (`df[['StudyHours']]`) and $y$ as 1D Series (`df['ExamScore']`).
3. **Train-Test Split:** Divide dataset into training (80%) and testing (20%) sets using `sklearn.model_selection.train_test_split`.
4. **Model Training:** Fit `sklearn.linear_model.LinearRegression` on training data.
5. **Predictions & Evaluation:** Predict on test set and calculate $R^2$ score and RMSE.

---

## 🚀 Model Performance

For the **Study Hours vs Exam Score** dataset:
* **$R^2$ Score**
* **RMSE**

---
