import requests
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define API endpoint
API_URL = "http://localhost:8000"  # Change to your deployed URL if different

def test_api_connection():
    """Test connection to the API"""
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API connection successful!")
            print(response.json())
        else:
            print(f"❌ API connection failed with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ API connection error: {e}")

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/model-info/")
        if response.status_code == 200:
            print("✅ Model info retrieved successfully!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def make_prediction(features, edges):
    """Make a prediction using the API"""
    try:
        payload = {
            "features": features,
            "edges": edges
        }
        response = requests.post(f"{API_URL}/predict/", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Prediction: {result['prediction']}")
            print(f"Probability: {result['probability']:.4f}")
            print(f"Class probabilities: {result['class_probabilities']}")
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {e}")

def prepare_sample_from_csv():
    """Prepare a sample from the CSV file"""
    try:
        # Load data
        df = pd.read_csv('balanced_dataset.csv')
        
        # Select features
        feature_cols = ['genotype', 'longitude', 'latitude', 'tick', 'cape', 'cattle', 'bio5']
        X = df[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Take one sample
        sample = X_scaled[0].tolist()
        
        # Create a simple edge structure (fully connected for one node)
        # For a single node, it can only connect to itself
        edges = [[0, 0]]
        
        print(f"Prepared sample features: {sample}")
        return sample, edges
    
    except Exception as e:
        print(f"❌ Error preparing sample: {e}")
        # Return dummy data if there's an error
        return [0, 0, 0, 0, 0, 0, 0], [[0, 0]]

if __name__ == "__main__":
    print("Testing ECF Prediction API Client...")
    
    # Test API connection
    test_api_connection()
    
    # Get model info
    get_model_info()
    
    # Prepare sample data
    features, edges = prepare_sample_from_csv()
    
    # Make prediction
    make_prediction(features, edges)
