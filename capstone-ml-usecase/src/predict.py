import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib
import preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

def predict_churn(input_data):
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return
        
    # Feature Engineering
    df_engineered = preprocess.feature_engineering(input_data)
    
    pipeline = joblib.load(MODEL_PATH)
    
    # Some models (like XGBoost) output 0/1, some output 'Yes'/'No'
    prediction = pipeline.predict(df_engineered)
    proba = pipeline.predict_proba(df_engineered)[:, 1]
    
    res = prediction[0]
    if isinstance(res, (np.integer, int, np.floating, float)):
        res = "Yes" if res == 1 else "No"
        
    print(f"\\n--- Prediction Result ---")
    print(f"Churn Risk: {res}")
    print(f"Probability of Churn: {proba[0]:.2%}")
    return res, proba[0]

if __name__ == "__main__":
    # Sample High-Risk Input
    sample_data = pd.DataFrame([{
        'CustomerID': 'NEW_001',
        'Gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Tenure': 2,
        'InternetService': 'Fiber optic',
        'TechSupport': 'No',
        'Contract': 'Month-to-month',
        'MonthlyCharges': 85.0,
        'TotalCharges': 170.0
    }])
    
    print("Input Customer Profile:")
    for col in sample_data.columns:
        print(f"{col}: {sample_data[col].iloc[0]}")
        
    predict_churn(sample_data)
