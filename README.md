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

<br>
`data/creditcard.csv`
<br>

### Reproducing Results

After placing the dataset in the `data/` directory, run the preprocessing and training pipeline:

<br>
<br>python src/preprocess.py
<br>python src/train.py
<br>python src/evaluate.py
<br>

The preprocessing artifacts and trained models are generated automatically and do not need to be downloaded.
