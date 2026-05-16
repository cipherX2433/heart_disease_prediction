# 🫀 Heart Disease Prediction

A machine learning project that predicts the presence or absence of heart disease using clinical features. The pipeline covers exploratory data analysis, preprocessing, model training with LightGBM and XGBoost, and hyperparameter optimization via Optuna.

---
<img width="2920" height="1532" alt="image" src="https://github.com/user-attachments/assets/8519024f-c00b-474b-bf00-3db38eefc1a1" />

[Demo App](https://heartdiseaseprediction-m4dwnvfcfigepzvsjdq7bh.streamlit.app/)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview

This project tackles binary classification — predicting whether a patient has heart disease (`Presence`) or not (`Absence`) based on 13 clinical features. The workflow includes:

- Data loading and inspection
- Stratified train/test splitting
- Model training with gradient boosting classifiers (LightGBM, XGBoost)
- Hyperparameter tuning with Optuna
- Evaluation using ROC-AUC and accuracy

---

## Dataset

The dataset contains **630,000 samples** and **13 features** (plus the target label).

| Feature | Description |
|---|---|
| `Age` | Patient age |
| `Sex` | Gender (binary encoded) |
| `Chest pain type` | Type of chest pain (1–4) |
| `BP` | Resting blood pressure |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `FBS over 120` | Fasting blood sugar > 120 mg/dl (binary) |
| `EKG results` | Resting electrocardiographic results |
| `Max HR` | Maximum heart rate achieved |
| `Exercise angina` | Exercise-induced angina (binary) |
| `ST depression` | ST depression induced by exercise relative to rest |
| `Slope of ST` | Slope of peak exercise ST segment |
| `Number of vessels fluro` | Number of major vessels colored by fluoroscopy (0–3) |
| `Thallium` | Thallium stress test result |
| **`Heart Disease`** | **Target: `Presence` / `Absence`** |

**Class distribution:**
- Absence: 347,546 (55.2%)
- Presence: 282,454 (44.8%)

---

## Project Structure

```
heart-disease-prediction/
│
├── datasets/
│   ├── train.csv
│   └── test.csv
│
├── heart_disease_prediction.ipynb   # Main notebook
├── best_params.pkl                  # Saved best hyperparameters from Optuna
└── README.md
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install dependencies:
   ```bash
   pip install lightgbm xgboost optuna scikit-learn pandas numpy matplotlib seaborn joblib
   ```

3. (Optional) If using Google Colab, mount your Google Drive and update the dataset paths in the notebook accordingly.

---

## Usage

Open and run the notebook:

```bash
jupyter notebook heart_disease_prediction.ipynb
```

To load the saved best hyperparameters from a previous Optuna study:

```python
import joblib
best_params = joblib.load("best_params.pkl")
```

---

## Modeling Approach

1. **Train/Test Split** — 80/20 stratified split to preserve class balance.
2. **LightGBM** — Fast gradient boosting used as the primary classifier.
3. **XGBoost** — Used as a secondary model within the Optuna tuning loop.
4. **Optuna Hyperparameter Optimization** — Bayesian search over key parameters (e.g., `learning_rate`, `max_depth`, `n_estimators`, `subsample`) with cross-validation to maximize ROC-AUC.
5. **Evaluation** — Models are assessed on ROC-AUC score and accuracy on the held-out test set.

---

## Results

| Metric | Value |
|---|---|
| Best model | XGBoost (Optuna-tuned) |
| Evaluation metric | ROC-AUC |
| Best hyperparameters | Saved to `best_params.pkl` |

> **Note:** Full numeric results will be updated here after a complete run. The Optuna XGBoost tuning was interrupted mid-run (KeyboardInterrupt) in the current notebook state.

---

## Dependencies

- Python 3.10+
- `lightgbm`
- `xgboost`
- `optuna`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`

---

## Acknowledgements

Dataset sourced from a heart disease prediction challenge. Feature definitions are based on the classic Cleveland Heart Disease dataset conventions.
