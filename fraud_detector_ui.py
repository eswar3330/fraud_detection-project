import streamlit as st
import requests
import pandas as pd

st.title("🔍 Fraudulent Transaction Detector with AI")

# Option 1: Manual Input
st.header("Manual Input")
features = []
for i in range(28):  # V1 to V28
    features.append(st.number_input(f"V{i+1}", value=0.0))
features.append(st.number_input("Amount", value=0.0))  # Amount

if st.button("Detect Fraud (Manual)"):
    transaction_data = {"features": features}
    
    # Call FastAPI backend
    try:
        response = requests.post("http://127.0.0.1:8000/detect", json=transaction_data)
        result = response.json()
        
        st.write(f"### Fraud Detection Result: {'Fraud' if result['fraud_detected'] else 'Legit'}")
        st.write(f"### Explanation: {result['explanation']}")
    except Exception as e:
        st.error(f"Error calling API: {e}")

# Option 2: CSV File Upload
st.header("CSV File Upload")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Call FastAPI backend for CSV processing
    try:
        response = requests.post(
            "http://127.0.0.1:8000/detect_csv",
            files={"file": uploaded_file}
        )
        result = response.json()
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.write("### Fraud Detection Results")
            for res in result["results"]:
                st.write(f"Transaction: {res['transaction']}")
                st.write(f"Fraud Detected: {'Yes' if res['fraud_detected'] else 'No'}")
                st.write(f"Explanation: {res['explanation']}")
                st.write("---")
    except Exception as e:
        st.error(f"Error calling API: {e}")