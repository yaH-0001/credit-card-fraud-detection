# Credit Card Fraud Detection (XGBoost vs. Random Forest)

An end-to-end machine learning pipeline built to detect fraudulent credit card transactions on the `creditcard_2023.csv` dataset[cite: 1].

## Dataset & Preprocessing
* **Source:** `creditcard_2023.csv`[cite: 1]
* **Cleaned Size:** 3,21,109 valid transaction records after removing 1 trailing empty (`NaN`) row[cite: 2].
* **Features:** PCA-transformed banking features (`V1`–`V28`) and transaction `Amount`, with `Class` (`0` = Legitimate, `1` = Fraud) as the target variable.
* **Split:** 80% training and 20% stratified testing set.

## Model Performance Comparison

| Metric (Test Set) | Random Forest | XGBoost (`hist`) |
| :--- | :---: | :---: |
| **True Non-Fraud** | 56,841[cite: 3] | 56,843[cite: 4] |
| **True Fraud Caught** | 6,997[cite: 3] | 7,348[cite: 4] |
| **False Alarms (FP)** | 22[cite: 3] | 20[cite: 4] |
| **Missed Frauds (FN)** | 362[cite: 3] | **11**[cite: 4] |
| **Accuracy** | ~99.40%[cite: 3] | **99.95%**[cite: 4] |
| **Fraud Recall** | ~95.10%[cite: 3] | **~99.85%**[cite: 4] |

## Key Insights
* **Winning Model:** XGBoost achieved an **ROC-AUC score of 0.99998** and reduced missed fraud cases from 362 down to 11[cite: 3, 4, 5].
* **Top Predictive Features:** `V14` is the strongest driver of fraud detection (importance score ~0.66), followed by `V17`, `V4`, and `V10`[cite: 5].

## Quick Start
1. Install dependencies:
   ```bash
   pip install streamlit xgboost pandas scikit-learn