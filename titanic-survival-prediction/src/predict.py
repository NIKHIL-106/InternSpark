import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import utils
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_titanic_model.pkl')

def predict(input_df):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    # Feature engineering requires columns like Name, Ticket, Cabin to exist or be handled safely.
    # Our feature_engineering handles missing columns using errors='ignore', but expects 'Name' for Title.
    # If Name is missing, we must provide a dummy or handle it.
    if 'Name' not in input_df.columns:
        input_df['Name'] = 'Mr. Unknown' # Default dummy title
    if 'SibSp' not in input_df.columns:
        input_df['SibSp'] = 0
    if 'Parch' not in input_df.columns:
        input_df['Parch'] = 0
    if 'Cabin' not in input_df.columns:
        input_df['Cabin'] = np.nan
        
    engineered_df = utils.feature_engineering(input_df)
    
    # Load model pipeline
    model_pipeline = utils.load_model(MODEL_PATH)
    
    # Predict
    prediction = model_pipeline.predict(engineered_df)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    
    print(f"\\nPrediction: {result}")
    return prediction[0]

if __name__ == "__main__":
    # sample_passenger = [[3, 'male', 22, 1, 0, 7.25]]
    # Columns typically are: Pclass, Sex, Age, SibSp, Parch, Fare
    sample_data = {
        'Pclass': [3],
        'Sex': ['male'],
        'Age': [22],
        'SibSp': [1],
        'Parch': [0],
        'Fare': [7.25],
        'Name': ['Mr. Owen Harris Braund'],
        'Cabin': [np.nan],
        'Embarked': ['S']
    }
    sample_df = pd.DataFrame(sample_data)
    
    print("Input Passenger Data:")
    for col, val in sample_data.items():
        print(f"{col}: {val[0]}")
        
    predict(sample_df)
