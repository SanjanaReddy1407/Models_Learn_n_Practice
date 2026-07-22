# 🏡 Multiple Linear Regression - House Price Prediction Project

This repository contains an end-to-end implementation of **Multiple Linear Regression** and **Feature Engineering** using Python, Pandas, and Scikit-Learn. The goal is to predict house prices based on multiple continuous and categorical features.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [Data Preprocessing & Encoding](#-data-preprocessing--encoding)
- [Feature Engineering](#-feature-engineering)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Visualizations](#-visualizations)
- [Key Takeaways & Learnings](#-key-takeaways--learnings)

---

## 🧠 Project Overview

Unlike Simple Linear Regression (which uses only one input feature), **Multiple Linear Regression** predicts a dependent target variable ($Y$) using two or more independent features ($X_1, X_2, \dots, X_n$).

### Mathematical Equation:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$$

Where:
* $Y$ = Predicted Target (`price`)
* $\beta_0$ = Intercept
* $\beta_1, \beta_2, \dots, \beta_n$ = Coefficients (Slopes for each feature)
* $X_1, X_2, \dots, X_n$ = Input Features (`area`, `bedrooms`, `bathrooms`, etc.)

---

## 📊 Dataset Summary

* **Dataset File:** `House price DB.csv`
* **Total Rows:** 545 entries
* **Target Variable ($Y$):** `price`
* **Features ($X$):**
  * **Numerical:** `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
  * **Binary Categorical:** `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
  * **Multi-class Categorical:** `furnishingstatus` (`furnished`, `semi-furnished`, `unfurnished`)

---

## ⚙️ Data Preprocessing & Encoding

To make categorical data readable for the model:
1. **Binary Mapping:** Converted `'yes'` $\rightarrow$ `1` and `'no'` $\rightarrow$ `0` for binary columns.
2. **One-Hot Encoding:** Applied `pd.get_dummies()` on `furnishingstatus` (`drop_first=True`) to handle multi-class categories without introducing ordinal bias.

---

## 🛠️ Feature Engineering

To capture deeper domain insights and improve $R^2$ performance, the following engineered features were created:

1. **`total_amenities`:** Sum of all binary luxury amenities (`mainroad` + `guestroom` + `basement` + `hotwaterheating` + `airconditioning` + `prefarea`).
2. **`area_per_bedroom`:** Spatial density feature ($Area / Bedrooms$).
3. **`baths_per_bedroom`:** Ratio of bathrooms to bedrooms ($Bathrooms / Bedrooms$).

---

## 🚀 Model Training & Evaluation

The model was split into 70% Training and 30% Testing data using `train_test_split` (`random_state=42`).

### Performance Metrics:
* **$R^2$ Score:** `~0.6452` (Model explains **64.52%** of the variance in house prices)
* **RMSE:** `~₹12.38 Lakhs`

### Feature Insights (Top Influencers):
* **`bathrooms`:** Has the highest positive impact on house price (~₹11.17 Lakhs per extra bathroom).
* **`airconditioning`:** Increases value by ~₹6.80 Lakhs.
* **`prefarea`:** Premium area boosts house price by ~₹5.09 Lakhs.

---

## 📈 Visualizations

To evaluate prediction accuracy, a scatter plot of **Actual Prices vs. Predicted Prices** was plotted along with a $y = x$ reference line:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(Y_test / 100000, predictions / 100000, color='blue', alpha=0.7, label='Predicted Points')
plt.plot([Y_test.min() / 100000, Y_test.max() / 100000], 
         [Y_test.min() / 100000, Y_test.max() / 100000], 
         color='red', linewidth=2, linestyle='--', label='Perfect Fit Line (y=x)')

plt.xlabel('Actual Price (in Lakhs ₹)')
plt.ylabel('Predicted Price (in Lakhs ₹)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
