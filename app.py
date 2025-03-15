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
from sklearn.cluster import KMeans
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

# Define Teacher Model (Larger GNN)
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

# Knowledge Distillation Loss
def distillation_loss(student_output, teacher_output, labels, T=2.0, alpha=0.7):
    soft_targets = F.softmax(teacher_output / T, dim=1)
    student_probs = F.log_softmax(student_output / T, dim=1)
    distill_loss = F.kl_div(student_probs, soft_targets, reduction="batchmean") * (T**2)
    ce_loss = F.nll_loss(student_output, labels)
    return alpha * distill_loss + (1 - alpha) * ce_loss

# Helper functions
def load_and_preprocess_data():
    """Load and preprocess the dataset with SMOTE and clustering"""
    try:
        # Load data
        df = pd.read_csv('balanced_dataset.csv')
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Select features (excluding name, target and spatial identifiers)
        feature_cols = ['genotype', 'longitude', 'latitude', 'tick', 'cape', 'cattle', 'bio5']
        X = df[feature_cols].values
        y = df['ECF'].values
        
        # Apply SMOTE to balance the dataset
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"After SMOTE: {len(X_resampled)} samples")
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_resampled)
        
        # Add cluster information as a feature
        X_with_clusters = np.column_stack((X_resampled, clusters))
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_with_clusters)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        # Update feature columns to include cluster
        feature_cols.append('cluster')
        
        return X_train, X_test, y_train, y_test, scaler, feature_cols
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_graph_data(X, y=None):
    """Create a graph from feature data using Erdős-Rényi model"""
    num_nodes = X.shape[0]
    
    # Create Erdős-Rényi graph with probability p=0.1
    G = nx.erdos_renyi_graph(num_nodes, p=0.1, seed=42)
    edge_list = list(G.edges())
    
    source_nodes = [e[0] for e in edge_list]
    target_nodes = [e[1] for e in edge_list]
    
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)
    
    if y is not None:
        y = torch.tensor(y, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)
    else:
        return Data(x=x, edge_index=edge_index)

def train_teacher_model(teacher_model, data, device, epochs=100):
    """Train the teacher GNN model"""
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.01)
    
    teacher_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = teacher_model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            logger.info(f'Teacher Model - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return teacher_model

def train_student_with_distillation(teacher_model, student_model, data, device, epochs=100):
    """Train the student GNN model with knowledge distillation"""
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)
    
    student_model.train()
    teacher_model.eval()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        student_out = student_model(data.x, data.edge_index)
        
        with torch.no_grad():
            teacher_out = teacher_model(data.x, data.edge_index)
        
        loss = distillation_loss(student_out, teacher_out, data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            logger.info(f'Student Model - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return student_model

def evaluate_model(model, data, device):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.y.size(0)
    return acc

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

input_dim = 8  # Number of features (7 original + 1 cluster)
teacher_hidden_dim = 64
student_hidden_dim = 32
output_dim = 2  # Binary classification

# Initialize models
teacher_model = TeacherGNN(input_dim, teacher_hidden_dim, output_dim).to(device)
student_model = StudentGNN(input_dim, student_hidden_dim, output_dim).to(device)

# Model paths
teacher_model_path = "teacher_model.pth"
student_model_path = "student_model.pth"
scaler = None
feature_cols = []

@app.on_event("startup")
async def startup_event():
    global teacher_model, student_model, scaler, feature_cols
    
    try:
        # Check if the student model file exists
        if os.path.exists(student_model_path) and os.path.exists("model_metadata.pt"):
            logger.info("Loading existing models...")
            # Load student model
            student_state_dict = torch.load(student_model_path, map_location=device)
            student_model.load_state_dict(student_state_dict)
            
            # Load metadata
            metadata = torch.load("model_metadata.pt", map_location=device)
            scaler = metadata["scaler"]
            feature_cols = metadata["feature_cols"]
            logger.info(f"Loaded model metadata. Features: {feature_cols}")
            
            # Also load teacher model if it exists (optional)
            if os.path.exists(teacher_model_path):
                teacher_state_dict = torch.load(teacher_model_path, map_location=device)
                teacher_model.load_state_dict(teacher_state_dict)
                logger.info("Loaded teacher model")
        else:
            logger.info("Training new models with knowledge distillation...")
            # Load and preprocess data
            X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_preprocess_data()
            
            # Create graph data
            train_data = create_graph_data(X_train, y_train).to(device)
            test_data = create_graph_data(X_test, y_test).to(device)
            
            # Train teacher model
            teacher_model = train_teacher_model(teacher_model, train_data, device)
            teacher_acc = evaluate_model(teacher_model, test_data, device)
            logger.info(f"Teacher model test accuracy: {teacher_acc:.4f}")
            
            # Train student model with knowledge distillation
            student_model = train_student_with_distillation(teacher_model, student_model, train_data, device)
            student_acc = evaluate_model(student_model, test_data, device)
            logger.info(f"Student model test accuracy: {student_acc:.4f}")
            
            # Save models and metadata
            torch.save(teacher_model.state_dict(), teacher_model_path)
            torch.save(student_model.state_dict(), student_model_path)
            torch.save({
                "scaler": scaler,
                "feature_cols": feature_cols
            }, "model_metadata.pt")
            logger.info("Models and metadata saved")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue running the application even if model training fails
        # This allows us to manually fix issues and restart later

@app.get("/")
async def root():
    return {"message": "ECF Prediction API with Knowledge Distillation", "status": "active"}

@app.post("/predict/")
async def predict(data: PredictionRequest):
    try:
        # Convert input to tensor
        X = torch.tensor(data.features, dtype=torch.float).view(1, -1)
        
        # Create self-loop for the single node since we need some edge structure
        # This fixes the "index out of bounds" error by ensuring we have valid edges
        edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
        
        # Make prediction with student model (smaller and faster)
        student_model.eval()
        with torch.no_grad():
            output = student_model(X, edge_index)
            probabilities = torch.exp(output).numpy()[0]
            prediction = output.argmax(dim=1).item()
        
        # Optionally also get teacher prediction for comparison
        teacher_model.eval()
        with torch.no_grad():
            teacher_output = teacher_model(X, edge_index)
            teacher_probabilities = torch.exp(teacher_output).numpy()[0]
            teacher_prediction = teacher_output.argmax(dim=1).item()
        
        return {
            "prediction": int(prediction),  # 0 or 1
            "probability": float(probabilities[prediction]),  # Probability of predicted class
            "class_probabilities": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            },
            "teacher_prediction": int(teacher_prediction),
            "teacher_probabilities": {
                "class_0": float(teacher_probabilities[0]),
                "class_1": float(teacher_probabilities[1])
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info/")
async def model_info():
    """Return information about the model"""
    return {
        "model_architecture": "Knowledge Distillation with GCN",
        "teacher_model": {
            "type": "Graph Convolutional Network (GCN)",
            "input_features": input_dim,
            "hidden_layers": [teacher_hidden_dim, teacher_hidden_dim],
            "output_classes": output_dim
        },
        "student_model": {
            "type": "Graph Convolutional Network (GCN)",
            "input_features": input_dim,
            "hidden_layers": [student_hidden_dim],
            "output_classes": output_dim
        },
        "trained": os.path.exists(student_model_path),
        "device": str(device),
        "features": feature_cols if feature_cols else ["Not loaded yet"]
    }

# Run locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
