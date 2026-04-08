import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import mlflow
import pickle

# ================= LOAD DATA =================
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ================= TRAIN-TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= PIPELINE =================
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        random_state=42
    ))
])

# ================= TRAIN =================
pipeline.fit(X_train)

# ================= PREDICT =================
y_pred = pipeline.predict(X_test)

# Convert IsolationForest output
y_pred = np.where(y_pred == -1, 1, 0)

# ================= METRICS =================
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

# ROC-AUC (important)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# ================= ANOMALY RATIO =================
detected_ratio = y_pred.mean()
true_ratio = y_test.mean()

print(f"\nTrue fraud ratio      : {true_ratio:.5f}")
print(f"Detected anomaly ratio: {detected_ratio:.5f}")

# ================= SAVE MODEL =================
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# ================= MLflow =================
mlflow.set_experiment("Credit Card Fraud Detection")

with mlflow.start_run():
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("contamination", 0.0017)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("true_fraud_ratio", true_ratio)
    mlflow.log_metric("detected_anomaly_ratio", detected_ratio)

    mlflow.log_artifact("model.pkl")

    print("\n✅ Model logged to MLflow")

print("\n🎯 Training & Evaluation Complete")

