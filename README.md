📂 Project Structure
primrose/
│
├── app.py                     # Main Gradio app
├── requirements.txt           # Python dependencies
│
├── data/                      
│   ├── sample_claims.csv      
│
├── models/
├   ├──label_encoder.joblib
├   ├──logreg_pipeline.joblib
│   ├──rf_pipeline.joblib
│   ├──xgb_bundle.joblib
│   ├──xgb_pipeline.joblib
│
├── outputs/                   
│   ├── logs/                  
│   ├── reports/               
│   └── figures/               
│
└── README.md                  

⚙️ Installation

Clone this repo:

git clone https://github.com/your-username/primrose.git
cd primrose


Install dependencies:

pip install -r requirements.txt


(Optional) Train a new model:

jupyter notebook train_model.ipynb


This saves xgb_bundle.joblib inside the project.

▶️ Run the Application

Run the Gradio app:

python app.py


Console output:

Running on local URL: http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live


Open the public URL in browser

Upload a .csv or .xlsx file with the following columns:
