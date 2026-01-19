import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("💳 Credit Card Fraud Detector")

# Load trained model
model = pickle.load(open("isolation_forest_model.pkl", "rb"))

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.dataframe(df.head())

    # Predict
    preds = model.predict(df)
    preds = np.where(preds == 1, 0, 1)  # 0: normal, 1: fraud
    df['Predicted_Fraud'] = preds

    st.success(f"Detected {preds.sum()} potential frauds out of {len(preds)} transactions.")
    st.dataframe(df.head(10))

    # Download predictions
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv")
