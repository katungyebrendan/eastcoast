import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from torch_geometric.nn import GCNConv
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="ECF Prediction Model")

# Define request model
class PredictionRequest(BaseModel):
    features: List[float]
    edges: List[List[int]]

# Define Student Model
class StudentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(StudentGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Helper functions
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        # Load data
        df = pd.read_csv('balanced_dataset.csv')
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Select features (excluding name, target and spatial identifiers)
        feature_cols = ['genotype', 'longitude', 'latitude', 'tick', 'cape', 'cattle', 'bio5']
        X = df[feature_cols].values
        y = df['ECF'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, scaler, feature_cols
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_graph_data(X, y=None):
    """Create a graph from feature data"""
    # For this example, we'll create a fully connected graph
    num_nodes = X.shape[0]
    source_nodes = []
    target_nodes = []
    
    # Create fully connected graph (each node connects to every other node)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Don't connect node to itself
                source_nodes.append(i)
                target_nodes.append(j)
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)
    
    if y is not None:
        y = torch.tensor(y, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)
    else:
        return Data(x=x, edge_index=edge_index)

def train_model(model, data, device, epochs=100):
    """Train the GNN model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, data, device):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.y.size(0)
    return acc

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

input_dim = 7  # Number of features
hidden_dim = 32
output_dim = 2  # Binary classification

# Initialize model
student_model = StudentGNN(input_dim, hidden_dim, output_dim).to(device)

# Check if model already exists, if not, train and save
model_path = "student_model.pth"
scaler = None
feature_cols = []

@app.on_event("startup")
async def startup_event():
    global student_model, scaler, feature_cols
    
    try:
        # Check if the model file exists
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            # Load model state dict
            state_dict = torch.load(model_path, map_location=device)
            student_model.load_state_dict(state_dict)
            
            # We also need to load the scaler and feature columns
            if os.path.exists("model_metadata.pt"):
                metadata = torch.load("model_metadata.pt", map_location=device)
                scaler = metadata["scaler"]
                feature_cols = metadata["feature_cols"]
                logger.info(f"Loaded model metadata. Features: {feature_cols}")
        else:
            logger.info("Training new model...")
            # Load and preprocess data
            X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_preprocess_data()
            
            # Create graph data
            train_data = create_graph_data(X_train, y_train).to(device)
            test_data = create_graph_data(X_test, y_test).to(device)
            
            # Train model
            student_model = train_model(student_model, train_data, device)
            
            # Evaluate
            test_acc = evaluate_model(student_model, test_data, device)
            logger.info(f"Test accuracy: {test_acc:.4f}")
            
            # Save model and metadata
            torch.save(student_model.state_dict(), model_path)
            torch.save({
                "scaler": scaler,
                "feature_cols": feature_cols
            }, "model_metadata.pt")
            logger.info("Model and metadata saved")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue running the application even if model training fails
        # This allows us to manually fix issues and restart later

@app.get("/")
async def root():
    return {"message": "ECF Prediction API", "status": "active"}

@app.post("/predict/")
async def predict(data: PredictionRequest):
    try:
        # Convert input to tensor
        X = torch.tensor(data.features, dtype=torch.float).view(1, -1)
        edge_index = torch.tensor(data.edges, dtype=torch.long).t().contiguous()
        
        # Make prediction
        student_model.eval()
        with torch.no_grad():
            output = student_model(X, edge_index)
            probabilities = torch.exp(output).numpy()[0]
            prediction = output.argmax(dim=1).item()
        
        return {
            "prediction": int(prediction),  # 0 or 1
            "probability": float(probabilities[prediction]),  # Probability of predicted class
            "class_probabilities": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info/")
async def model_info():
    """Return information about the model"""
    return {
        "model_type": "Graph Convolutional Network (GCN)",
        "input_features": input_dim,
        "hidden_layers": [hidden_dim],
        "output_classes": output_dim,
        "trained": os.path.exists(model_path),
        "device": str(device)
    }

# Run locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
