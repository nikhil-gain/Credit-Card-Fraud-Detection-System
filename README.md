# Credit Card Fraud Detection System

A real-time credit card fraud detection system built with XGBoost, SHAP, FastAPI and Streamlit.
Trained on 284k+ real European credit card transactions with 97%+ ROC-AUC score.

## Live Demo
- **Dashboard:** https://real-time-credit-card-fraud-detection-system.streamlit.app/
- **API Docs:** https://fraud-detection-api-7g34.onrender.com/docs

> ⚠️ First request may take 30–60 seconds (Render free tier wakes up on demand)

## Tech Stack
- Python, XGBoost, SMOTE, SHAP
- FastAPI (REST API backend)
- Streamlit (Interactive dashboard)

## Features
- Trained on 284k+ real transactions with 0.17% fraud rate
- Handles class imbalance using SMOTE
- 97%+ ROC-AUC score with hyperparameter tuning via GridSearchCV
- SHAP explainability for each prediction
- REST API + Interactive real-time dashboard

## Dataset
Uses the Credit Card Fraud Detection dataset from Kaggle.
- URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Download Instructions
1. Download `creditcard.csv` from Kaggle
2. Place it at `data/creditcard.csv`

## Run Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Reproducing the results

After placing the dataset in the `data/` directory, run the preprocessing and training pipeline:
```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```
The preprocessing artifacts and trained models are generated automatically and do not need to be downloaded.

### Start FastAPI backend
```bash
cd app
uvicorn app:app --reload
```


### Start Streamlit dashboard
```bash
cd app
streamlit run dashboard.py
```

### Test API directly in VS Code in windows
```bash
curl.exe -X POST "http://localhost:8000/predict" `
-H "Content-Type: application/json" `
-d '{\"features\": [-16.5265, 8.5850, -18.6499, 9.5056, -13.7938, -2.8324, -16.7017, 7.5173, -8.5071, -14.1102, 5.2992, -10.8340, 1.6711, -9.3739, 0.3608, -9.8992, -19.2363, -8.3986, 3.1017, -1.5149, 1.1907, -1.1277, -2.3586, 0.6735, -1.4137, -0.4628, -2.0186, -1.0428, 1.79]}'
```
### OR

test in your browser at following website after activating the FastAPI
```bash
http://localhost:8000/docs
```

Expected response for fraud case:
```json
{"fraud": true, "probability": 0.98}
```

## Project Structure
```bash
fraud-detection/
├── data/
│   └── creditcard.csv        # download from Kaggle
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_shap.ipynb
├── models/
│   └── fraud_model.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   ├── app.py                # FastAPI backend
│   └── dashboard.py          # Streamlit frontend
├── requirements.txt
└── README.md
```

### Hope, you enjoyed the project. 


