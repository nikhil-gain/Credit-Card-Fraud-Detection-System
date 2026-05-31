from fastapi import FastAPI
import joblib
import numpy as np 
from pydantic import BaseModel
from typing import List

model = joblib.load("../models/fraud_model.pkl")

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

    # Get prediction and fraud probability
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return{
        'fraud': bool(prediction),
        'probability': round(float(probability), 4)
    }

