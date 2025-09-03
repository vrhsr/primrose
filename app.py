import streamlit as st
import pandas as pd
import joblib
import pickle

# Load models
xgb_model = joblib.load("model/xgb_pipeline.joblib")
rf_model = joblib.load("model/rf_pipeline.joblib")
with open("model/labelencoder.pkl", "rb") as f:
    le_target = pickle.load(f)

st.title("📊 Claim Denial Prediction App")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Uploaded Data", df.head())

    # Predict
    preds = xgb_model.predict(df)
    preds_labels = le_target.inverse_transform(preds)

    df["Predicted Denial Reason"] = preds_labels

    st.write("### Predictions", df)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
