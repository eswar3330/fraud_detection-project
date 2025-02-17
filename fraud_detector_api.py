import numpy as np
import joblib
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv
import pandas as pd
from huggingface_hub import InferenceClient  # Alternative: Hugging Face API

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Load Environment Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")  # For Hugging Face

# Check API Key Validity
if not api_key:
    logging.error("⚠️ Google API key is missing! Please set it in the .env file.")
    raise ValueError("Google API key is missing!")

logging.info("✅ Google API Key Loaded Successfully")

genai.configure(api_key=api_key)  # Corrected API Key Setup

# Load Pretrained Models with Exception Handling
try:
    iso_forest = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    logging.error(f"❌ Model file missing: {e}")
    raise
except Exception as e:
    logging.error(f"⚠️ Error loading model files: {e}")
    raise

# FastAPI App Setup
app = FastAPI()

# Request Body Model for Single Transaction
class Transaction(BaseModel):
    features: list

# Fraud Detection Function
def detect_fraud(features):
    try:
        # Convert features to DataFrame
        feature_names = [f"V{i+1}" for i in range(28)] + ["Amount"]
        features_df = pd.DataFrame([features], columns=feature_names)

        # Scale the features
        features_scaled = scaler.transform(features_df)

        # Isolation Forest Prediction
        iso_pred = iso_forest.predict(features_scaled)

        return iso_pred[0] == -1  # Returns True if fraud detected
    except Exception as e:
        logging.error(f"⚠️ Error in fraud detection: {e}")
        raise HTTPException(status_code=500, detail="Error in fraud detection.")

# Fraud Explanation Function (Google Gemini or Hugging Face)
def explain_fraud(features):
    prompt = f"""
    A bank transaction has been flagged as potentially fraudulent.
    Transaction details: {features}
    Explain why this transaction might be fraud.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response else "Could not generate explanation."
    except Exception as e:
        logging.error(f"⚠️ Google Gemini API error: {e}")
        logging.info("Switching to Hugging Face as a backup.")
        return explain_fraud_huggingface(features)

# Alternative: Hugging Face API for Fraud Explanation
def explain_fraud_huggingface(features):
    if not hf_api_key:
        return "Hugging Face API key is missing. Cannot generate explanation."
    
    client = InferenceClient(token=hf_api_key)
    MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    
    try:
        response = client.text_generation(prompt=f"Explain why this transaction is fraud: {features}", model=MODEL, max_new_tokens=100)
        return response if response else "Hugging Face response unavailable."
    except Exception as e:
        logging.error(f"⚠️ Hugging Face API error: {e}")
        return "Could not generate explanation from Hugging Face."

# API Endpoint for Single Transaction
@app.post("/detect")
def detect(transaction: Transaction):
    is_fraud = detect_fraud(transaction.features)
    explanation = explain_fraud(transaction.features) if is_fraud else "Transaction looks normal."
    return {
        "fraud_detected": bool(is_fraud),
        "explanation": explanation
    }

# API Endpoint for CSV File Upload
@app.post("/detect_csv")
async def detect_csv(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)
        
        # Ensure the file has the correct columns
        required_columns = [f"V{i+1}" for i in range(28)] + ["Amount"]
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="CSV file must contain columns: V1, V2, ..., V28, Amount")
        
        # Process each transaction
        results = []
        for _, row in df.iterrows():
            features = row.tolist()
            is_fraud = detect_fraud(features)
            explanation = explain_fraud(features) if is_fraud else "Transaction looks normal."
            results.append({
                "transaction": features,
                "fraud_detected": bool(is_fraud),
                "explanation": explanation
            })
        
        return {"results": results}
    except Exception as e:
        logging.error(f"⚠️ Error processing CSV file: {e}")
        raise HTTPException(status_code=500, detail="Error processing CSV file.")

# To run the server, use:
# `uvicorn fraud_detector_api:app --host 0.0.0.0 --port 8000 --reload`