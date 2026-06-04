import pandas as pd 
import numpy as np 
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path):
    dataset = pd.read_csv(path)
    return dataset

def preprocess(dataset):
    dataset.drop(columns=['Time'], inplace=True)
    scaler = StandardScaler()
    dataset['Amount'] = scaler.fit_transform(dataset[['Amount']])
    x = dataset.iloc[:,:-1]
    y = dataset["Class"]
    return x, y, scaler

def split_and_resample(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)
    return x_train_sm, x_test, y_train_sm, y_test

def save_artifacts(scaler, x_train_sm, y_train_sm, x_test, y_test):
    joblib.dump(scaler, '../data/scaler.pkl')
    joblib.dump((x_train_sm, x_test, y_train_sm, y_test), '../data/preprocessed_splits.pkl')
    print("Artifacts saved successfully.")

if __name__ == "__main__":
    dataset = load_data('../data/creditcard.csv')
    x, y, scaler = preprocess(dataset)
    x_train_sm, x_test, y_train_sm, y_test = split_and_resample(x, y)
    save_artifacts(scaler, x_train_sm, y_train_sm, x_test, y_test)
    print("Preprocessing completed successfully.")
   