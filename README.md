# Credit Card Fraud Detection (XGBoost vs. Random Forest)

An end-to-end machine learning pipeline and interactive web dashboard built to detect fraudulent credit card transactions on the `creditcard_2023.csv` dataset (`321,109` transactions).

**[Live Demo — Try the Fraud Detection App](https://credit-card-fraud-detection-atccxatzo77zxoyxpcfwvn.streamlit.app/)**

---

## Project Overview & Business Impact
* **Data Pipeline:** Cleaned and audited `321,109` PCA-transformed transactions, handling trailing null records and applying a stratified 80/20 train-test split.
* **Model Benchmarking:** Compared a baseline Random Forest classifier against an optimized XGBoost (`hist`) model to minimize undetected financial leakage.
* **Key Results:**
  * **ROC-AUC:** `0.99998`
  * **Overall Accuracy:** `99.95%`
  * **Fraud Recall:** `99.85%` — reduced missed fraud cases by **97%** compared to Random Forest (cutting false negatives from `362` down to `11` out of `7,359` test frauds).
  * **Low Customer Friction:** Limited false declines to just `20` out of `56,863` legitimate test transactions (`0.035%` false positive rate).
* **Top Fraud Signals:** Feature attribution isolated **`V14`** (`0.66` importance), **`V17`**, **`V4`**, and **`V10`** as the primary indicators for triggering risk-based step-up authentication (OTP/2FA).

---

## Testing the Live App
Open the [Live Demo](https://credit-card-fraud-detection-atccxatzo77zxoyxpcfwvn.streamlit.app/) and enter these sample signals under **Single Transaction Check**:

| Test Scenario | `V14` | `V17` | `V4` | `V10` | Expected Output |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Legitimate Transaction** | `1.50` | `1.00` | `-1.00` | `1.00` | Legitimate Transaction (Low Risk) |
| **Fraudulent Transaction** | `-15.00` | `-10.00` | `5.00` | `-8.00` | Fraudulent Transaction Detected (~99%+ Probability) |

You can also switch to the **Batch CSV Upload** tab to screen multiple transactions at once.

---

## Repository Structure
* `app.py` — Streamlit web application for real-time single-transaction scoring and batch CSV screening.
* `fraud_detection_xgb.json` — Trained XGBoost model artifact.
* `credit_card_fraud_detection.ipynb` — Full data cleaning, exploratory analysis, benchmarking, and feature importance notebook.
* `requirements.txt` — Python dependencies for local and cloud deployment.

---

## Run Locally
```bash
git clone [https://github.com/yaH-0001/credit-card-fraud-detection.git](https://github.com/yaH-0001/credit-card-fraud-detection.git)
cd credit-card-fraud-detection
pip install -r requirements.txt
python -m streamlit run app.py
