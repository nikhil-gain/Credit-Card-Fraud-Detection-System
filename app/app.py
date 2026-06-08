from fastapi import FastAPI
import joblib
import numpy as np 
from pydantic import BaseModel
from typing import List
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "../models/fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "../data/scaler.pkl"))

app = FastAPI()

# Define the input format -29 features ie. V1-V28 + Amount
class Transaction(BaseModel):
    features: List[float]

# Health check endpoint
@app.get('/')
def home():
    return{'status':"Fraud Detection API is running"}

# Prediciton endpoint
@app.post('/predict')
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    
    # It scales raw Amount from dashboard input
    data[0, -1]= scaler.transform([[data[0, -1]]])[0][0]
    # Get prediction and fraud probability
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return{
        'fraud': bool(prediction),
        'probability': round(float(probability), 4)
    }

