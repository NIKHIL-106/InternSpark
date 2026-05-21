import os
import sys

# Add the current directory to sys.path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import numpy as np
import argparse

# Path to the saved model relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

def predict(sample):
    """Predict the Iris species for a given sample."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
        return
    
    print("Loading saved model pipeline...")
    pipeline = utils.load_model(MODEL_PATH)
    model = pipeline['model']
    scaler = pipeline['scaler']
    le = pipeline['le']
    
    print(f"Input features: {sample}")
    
    # Preprocess the sample
    sample_scaled = scaler.transform(sample)
    
    # Predict
    prediction_encoded = model.predict(sample_scaled)
    prediction_species = le.inverse_transform(prediction_encoded)
    
    print(f"\nPredicted Species: {prediction_species[0]}")
    return prediction_species[0]

if __name__ == "__main__":
    # Example sample from requirements
    # Format: [[sepal length, sepal width, petal length, petal width]]
    default_sample = [[5.1, 3.5, 1.4, 0.2]]
    
    parser = argparse.ArgumentParser(description='Predict Iris species')
    parser.add_argument('--features', type=float, nargs=4, 
                        help='Provide 4 feature values: sepal_length sepal_width petal_length petal_width',
                        default=default_sample[0])
    
    args = parser.parse_args()
    
    # Convert to 2D array
    sample = np.array([args.features])
    predict(sample)
