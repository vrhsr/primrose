
### Project Structure

primrose
│

├── app.py

├── requirements.txt

├── README.md
│

├── data
│   ── sample_claims.csv

├── models
│   ── label_encoder.joblib
│   ── logreg_pipeline.joblib
│   ── rf_pipeline.joblib
│   ── xgb_bundle.joblib
│   ── xgb_pipeline.joblib

├── outputs
│   ── logs/
│   ── reports/
│   ── figures/

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

Running on public URL: https://76281562845dceb3dd.gradio.live

 ### 📝⭕Recommendation:

Before uploading a file to the Gradio web app, remove any blank rows or extra notes above the header row.

Ensure the header row contains all required columns: CPT Code, Insurance Company, Physician Name, Payment Amount, Balance.

In contrast, the Jupyter Notebook version automatically detects the header row, so it works even with uncleaned files.


once it's running you'll be redirected to a website 
where you find dropbox to upload 
<img width="1904" height="394" alt="image" src="https://github.com/user-attachments/assets/9bb162e8-0362-4810-a0c6-d2d5593feb1a" />


once you uploaded the file you could see these tabs 

"Model Performance
Insights
Visualizations
Predictions Table" 

#### Model Performance

<img width="1801" height="925" alt="image" src="https://github.com/user-attachments/assets/fd66548e-8fc7-4a62-949c-0a51b3b5f7ea" />
<img width="1782" height="206" alt="image" src="https://github.com/user-attachments/assets/c84dd526-224c-4a6b-93d3-107c376c9976" />


### insights  

<img width="1816" height="708" alt="image" src="https://github.com/user-attachments/assets/7545fb7f-1c3c-42de-96ce-6ccf0c3855e1" />

### visualization

<img width="1838" height="934" alt="image" src="https://github.com/user-attachments/assets/22dd0a99-b52f-47ac-889a-c0a4f62996fc" />
<img width="1814" height="573" alt="image" src="https://github.com/user-attachments/assets/10ae72c5-5f77-4ae5-80db-2fab68f4617c" />

### Prediction table 
<img width="1775" height="899" alt="image" src="https://github.com/user-attachments/assets/9970a95d-9562-43d4-93ae-79063c0a4ad8" />

