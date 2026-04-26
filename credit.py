import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

import mlflow
import joblib

# load data
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# build pipeline
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        random_state=42
    ))
])

# train model
pipeline.fit(X_train)

# predict scores
scores = pipeline.decision_function(X_test)

# set threshold
threshold = np.percentile(scores, 0.17)
y_pred = (scores < threshold).astype(int)

# evaluate model
print("\nConfusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

roc_auc = roc_auc_score(y_test, -scores)
ap_score = average_precision_score(y_test, -scores)

print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"Average Precision Score: {ap_score:.4f}")

# fraud ratio analysis
true_ratio = y_test.mean()
detected_ratio = y_pred.mean()

print("\nFraud Ratio Analysis")
print(f"True fraud ratio      : {true_ratio:.6f}")
print(f"Detected anomaly ratio: {detected_ratio:.6f}")

# save model
joblib.dump(pipeline, "model.pkl")

# mlflow tracking
mlflow.set_experiment("Credit Card Fraud Detection")

with mlflow.start_run():

    # log parameters
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("contamination", 0.0017)
    mlflow.log_param("threshold_percentile", 0.17)

    # log metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("avg_precision", ap_score)
    mlflow.log_metric("true_fraud_ratio", true_ratio)
    mlflow.log_metric("detected_anomaly_ratio", detected_ratio)

    # confusion matrix
    mlflow.log_metric("TN", cm[0][0])
    mlflow.log_metric("FP", cm[0][1])
    mlflow.log_metric("FN", cm[1][0])
    mlflow.log_metric("TP", cm[1][1])

    # save artifact
    mlflow.log_artifact("model.pkl")

    print("\nModel logged successfully in MLflow")

print("\nTraining & Evaluation Complete")