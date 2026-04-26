# 💳 Credit Card Fraud Detection (IsolationForest + MLflow)

This project is an **anomaly detection system** for identifying fraudulent credit card transactions using **IsolationForest (unsupervised learning)**.  
It also includes **MLflow tracking** and a reproducible ML pipeline.

## 🚀 Project Overview

- Detects fraudulent transactions using anomaly detection
- Uses **IsolationForest algorithm**
- Tracks experiments using **MLflow**
- Saves trained model for deployment
- Evaluates model using ROC-AUC and Average Precision


## 📊 Dataset

- Dataset: Credit Card Fraud Detection Dataset
- Features: PCA-transformed transaction features (V1–V28)
- Target: Class
  - 0 → Normal transaction
  - 1 → Fraud transaction


## 🧠 Model Used

### IsolationForest
- Unsupervised anomaly detection algorithm
- Learns normal transaction patterns
- Flags anomalies as fraud

## ⚙️ Workflow

1. Load dataset
2. Train-test split
3. Preprocessing (Scaling + Imputation)
4. Train IsolationForest model
5. Generate anomaly scores
6. Convert scores → fraud predictions
7. Evaluate model
8. Log results in MLflow

## 📈 Evaluation Metrics

- ROC-AUC Score: **0.95**
- Average Precision Score: **0.17**
- Confusion Matrix:
  - TN: 56797
  - FP: 67
  - FN: 68
  - TP: 30

## 📦 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- MLflow
- Joblib
- Docker (optional)

