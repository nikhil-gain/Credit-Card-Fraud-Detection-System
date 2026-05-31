# Credit-Card-Fraud-Detection-System
This is the new project of the credit card fraud detection system using the ml.

## Dataset

This project uses the Credit Card Fraud Detection dataset available on Kaggle.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Download Instructions

1. Download `creditcard.csv` from the Kaggle dataset page.
2. Create a `data/` directory in the project root if it does not already exist.
3. Place the downloaded file at:

```
data/creditcard.csv
```

### Reproducing Results

After placing the dataset in the `data/` directory, run the preprocessing and training pipeline:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

The preprocessing artifacts and trained models are generated automatically and do not need to be downloaded.

### To run the API
in ``` app/ ``` folder to run API

```bash 
uvicorn app:app --reload 
``` 
and in next PowerShell of VS Code:

```bash
curl.exe -X POST http://localhost:8000/predict `
-H "Content-Type: application/json" `
-d '{\"features\":[0.0,-1.35,-0.07,2.53,1.37,-0.33,0.46,0.23,0.09,0.36,0.09,-0.55,-0.61,-0.99,-0.31,1.46,-0.47,0.20,0.02,0.40,0.25,-0.01,0.27,-0.11,0.06,-0.20,0.17,0.12,0.50]}'
```
#### OR
 test in your browser at `http://localhost:8000/docs`