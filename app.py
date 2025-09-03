import streamlit as st
import pandas as pd
import joblib

# Load models
logreg_model = joblib.load("models/logreg_pipeline.joblib")
rf_model = joblib.load("models/rf_pipeline.joblib")
xgb_model = joblib.load("models/xgb_pipeline.joblib")
le_target = joblib.load("models/label_encoder.joblib")

st.title("📊 Denial Reason Prediction App")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file safely
    try:
        if uploaded_file.name.endswith(".csv"):
            df_new = pd.read_csv(uploaded_file)
        else:
            df_new = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    st.subheader("Preview of Uploaded Data")
    st.write(df_new.head())

    # Model choice
    model_choice = st.selectbox("Choose model", ["Logistic Regression", "Random Forest", "XGBoost"])

    if st.button("Predict"):
        try:
            if model_choice == "Logistic Regression":
                preds = logreg_model.predict(df_new)
            elif model_choice == "Random Forest":
                preds = rf_model.predict(df_new)
            else:
                preds = xgb_model.predict(df_new)

            # Decode labels
            preds_decoded = le_target.inverse_transform(preds)
            df_new["Predicted Denial Reason"] = preds_decoded

            st.subheader("Prediction Results")
            st.write(df_new)

            # Download predictions
            csv = df_new.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Prediction failed. Please check your file format.\n\nError: {e}")
