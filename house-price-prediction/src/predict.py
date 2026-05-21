import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import utils
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_house_model.pkl')

def predict(input_data):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    # Load model pipeline
    model_pipeline = utils.load_model(MODEL_PATH)
    
    # Predict
    prediction = model_pipeline.predict(input_data)
    print(f"\\nPredicted House Price: ${prediction[0]:,.2f}")
    return prediction[0]

if __name__ == "__main__":
    # Create a dummy DataFrame with identical columns as training for the pipeline to work correctly
    # Real inputs would normally come from an API JSON request or arguments
    # Let's provide a static example mirroring the requested format
    
    # The requirement:
    # sample_house = [[3, 2, 1500, 1, 5]]
    # prediction = model.predict(sample_house)
    
    # Since our pipeline requires a DataFrame with specific column names (because of ColumnTransformer),
    # we'll build a sample dataframe that matches the structure.
    sample_dict = {
        'LotFrontage': [65.0],
        'LotArea': [8450],
        'OverallQual': [7],
        'YearBuilt': [2003],
        'TotalBsmtSF': [856],
        'GrLivArea': [1710],
        'FullBath': [2],
        'BedroomAbvGr': [3],
        'GarageCars': [2],
        'GarageArea': [548],
        'MSZoning': ['RL'],
        'Neighborhood': ['CollgCr'],
        'CentralAir': ['Y']
    }
    sample_df = pd.DataFrame(sample_dict)
    
    print(f"Input features:\\n{sample_df.iloc[0]}")
    predict(sample_df)
