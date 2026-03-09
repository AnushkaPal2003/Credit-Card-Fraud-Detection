import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import pickle

# ================= LOAD DATA =================
df = pd.read_csv("creditcard.csv")

print(df.head())
print(df.info())

X = df.drop("Class", axis=1)
y_true = df["Class"]  # TRUE labels (0 = normal, 1 = fraud)

# ================= PIPELINE =================
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # handles NaNs
    ("scaler", StandardScaler()),
    ("model", IsolationForest(
        n_estimators=200,
        contamination=0.0017,  # fraud ratio
        random_state=42
    ))
])

# ================= TRAIN =================
pipeline.fit(X)

# ================= PREDICT =================
y_pred = pipeline.predict(X)

# IsolationForest output:
#  1  -> normal
# -1  -> anomaly

y_pred = np.where(y_pred == -1, 1, 0)  # convert to fraud=1

# ================= METRICS =================
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("Classification Report")
print(classification_report(y_true, y_pred, digits=4))

# ================= ANOMALY RATIO =================
detected_ratio = y_pred.mean()
true_ratio = y_true.mean()

print(f"True fraud ratio      : {true_ratio:.5f}")
print(f"Detected anomaly ratio: {detected_ratio:.5f}")

# ================= SAVE MODEL =================
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# ================= MLflow =================
mlflow.set_experiment("Credit Card Fraud Detection")

with mlflow.start_run():
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("contamination", 0.0017)
    mlflow.log_metric("true_fraud_ratio", true_ratio)
    mlflow.log_metric("detected_anomaly_ratio", detected_ratio)
    mlflow.log_artifact("model.pkl")

    print("\n✅ Model logged to MLflow")

print("\n🎯 Training & Evaluation Complete")

