
### Project Structure
primrose/
│
├── app.py              # Main FastAPI application
├── requirements.txt
├── README.md
│
├── data/               # Contains sample data
│   └── sample_claims.csv
│
├── models/             # Pre-trained models (DO NOT delete)
│   ├── label_encoder.joblib
│   ├── logreg_pipeline.joblib
│   ├── rf_pipeline.joblib
│   ├── xgb_bundle.joblib
│   └── xgb_pipeline.joblib
│
├── outputs/            # Generated logs, reports, and figures
│   ├── logs/
│   ├── reports/
│   └── figures/

### Installation

git clone https://github.com/vrhsr/primrose.git

cd primrose


### Install the dependencies:

pip install -r requirements.txt

### Jupyter Notebook

jupyter notebook primrose.ipynb

### Start the application:

python app.py

### Console Output

Running on local URL:  http://127.0.0.1:7860

Running on public URL: https://b17747416539cc1828.gradio.live
