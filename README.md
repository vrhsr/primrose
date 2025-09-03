---
title: primrose
app_file: app.py
sdk: gradio
sdk_version: 5.44.1
---
📊 Claim Denial Prediction App

This project is a machine learning-powered Streamlit app that predicts insurance claim denial reasons based on uploaded claim data.

🚀 Features

Upload CSV/Excel claim files.

Automatically preprocesses categorical data.

Predicts denial reasons using XGBoost / Random Forest models.

Displays predictions in a clean table.

Download predictions as a CSV file.

🏗️ Project Structure
primrose/
│── app.py                  # Streamlit app
│── model/
│    ├── xgb_pipeline.joblib
│    ├── rf_pipeline.joblib
│    └── labelencoder.pkl
│── requirements.txt        # Dependencies
│── data/
│    └── sample.csv         # Example file (optional)
│── README.md               # Project documentation

📦 Installation (Run Locally)

Clone the repo:

git clone https://github.com/your-username/primrose-ml-app.git
cd primrose-ml-app


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


Open in browser:

http://localhost:8501