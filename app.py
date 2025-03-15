import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="ECF Prediction Model")

# Define request model
class PredictionRequest(BaseModel):
    features: List[float]
    edges: List[List[int]]

# Define Teacher and Student Models
class TeacherGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TeacherGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class StudentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(StudentGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Knowledge Distillation Loss Function
def distillation_loss(student_output, teacher_output, labels, T=2.0, alpha=0.7):
    soft_targets = F.softmax(teacher_output / T, dim=1)
    student_probs = F.log_softmax(student_output / T, dim=1)
    distill_loss = F.kl_div(student_probs, soft_targets, reduction="batchmean") * (T**2)
    ce_loss = F.nll_loss(student_output, labels)
    return alpha * distill_loss + (1 - alpha) * ce_loss

# Helper functions
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        # Load data
        df = pd.read_csv('balanced_dataset.csv')
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Separate features and labels
        X = df[['tick', 'cape', 'cattle', 'bio5']].values  # Features
        y = df['ECF'].values  # Labels
        
        # Apply SMOTE to balance the dataset
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Cluster the data
        kmeans = KMeans(n_clusters=3, random_state=42)
        farm_clusters = kmeans.fit_predict(X_resampled)
        
        # Prepare Node Features (X) from the resampled dataset
        node_features = torch.tensor(X_resampled, dtype=torch.float)
        
        # Add clusters as node features
        node_features = torch.cat([node_features, torch.tensor(farm_clusters).unsqueeze(1)], dim=1)
        
        # Prepare Labels (y) for node classification
        labels = torch.tensor(y_resampled, dtype=torch.long)
        
        # Create Edge Index (Graph Structure) - Placeholder
        num_nodes = len(X_resampled)
        G = nx.erdos_renyi_graph(num_nodes, p=0.1)  # Placeholder: Replace with real adjacency structure
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data Object
        data = Data(x=node_features, edge_index=edge_index, y=labels)
        
        return data, labels.shape[1]
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Initialize models and parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

teacher_model = TeacherGNN(in_channels=6, hidden_channels=64, out_channels=2).to(device)
student_model = StudentGNN(in_channels=6, hidden_channels=32, out_channels=2).to(device)

teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.01)
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)

model_path = "student_model.pth"

@app.on_event("startup")
async def startup_event():
    global teacher_model, student_model, teacher_optimizer, student_optimizer
    
    try:
        # Load model if it exists
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            state_dict = torch.load(model_path, map_location=device)
            student_model.load_state_dict(state_dict)
        else:
            logger.info("Training new model...")

            # Load and preprocess data
            data, _ = load_and_preprocess_data()

            # Train Teacher Model
            for epoch in range(100):
                teacher_model.train()
                teacher_optimizer.zero_grad()
                out = teacher_model(data.x.to(device), data.edge_index.to(device))
                loss = F.nll_loss(out, data.y.to(device))
                loss.backward()
                teacher_optimizer.step()
                if epoch % 10 == 0:
                    logger.info(f"Teacher Epoch {epoch}: Loss = {loss.item():.4f}")

            # Train Student Model with Knowledge Distillation
            for epoch in range(100):
                student_model.train()
                student_optimizer.zero_grad()
                student_out = student_model(data.x.to(device), data.edge_index.to(device))
                with torch.no_grad():
                    teacher_out = teacher_model(data.x.to(device), data.edge_index.to(device))
                loss = distillation_loss(student_out, teacher_out, data.y.to(device))
                loss.backward()
                student_optimizer.step()
                if epoch % 10 == 0:
                    logger.info(f"Student Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # Save model
            torch.save(student_model.state_dict(), model_path)
            logger.info("Model saved.")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.post("/predict/")
async def predict(data: PredictionRequest):
    try:
        # Convert input to tensor
        X = torch.tensor(data.features, dtype=torch.float).view(1, -1)
        edge_index = torch.tensor(data.edges, dtype=torch.long).t().contiguous()
        
        # Make prediction using student model
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
    return {
        "model_type": "Graph Convolutional Network (GCN)",
        "input_features": 6,
        "hidden_layers": [64],
        "output_classes": 2,
        "trained": os.path.exists(model_path),
        "device": str(device)
    }

# Run locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
