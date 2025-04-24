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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
        
        # Select features based on the second implementation
        # Using the feature set from paste-2.txt
        feature_cols = ['tick', 'cape', 'cattle', 'bio5']
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
    
    # Evaluate teacher model
    teacher_model.eval()
    with torch.no_grad():
        teacher_out = teacher_model(data.x, data.edge_index)
        teacher_preds = teacher_out.argmax(dim=1)
        
        # Calculate metrics
        teacher_true_labels = data.y.cpu().numpy()
        teacher_pred_labels = teacher_preds.cpu().numpy()
        
        teacher_accuracy = accuracy_score(teacher_true_labels, teacher_pred_labels)
        teacher_precision = precision_score(teacher_true_labels, teacher_pred_labels, average='weighted')
        teacher_recall = recall_score(teacher_true_labels, teacher_pred_labels, average='weighted')
        teacher_f1 = f1_score(teacher_true_labels, teacher_pred_labels, average='weighted')
        
        logger.info(f"Teacher Model Metrics - Accuracy: {teacher_accuracy:.4f}, Precision: {teacher_precision:.4f}, Recall: {teacher_recall:.4f}, F1: {teacher_f1:.4f}")
    
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
    
    # Evaluate student model
    student_model.eval()
    with torch.no_grad():
        student_out = student_model(data.x, data.edge_index)
        student_preds = student_out.argmax(dim=1)
        
        # Calculate metrics
        student_true_labels = data.y.cpu().numpy()
        student_pred_labels = student_preds.cpu().numpy()
        
        student_accuracy = accuracy_score(student_true_labels, student_pred_labels)
        student_precision = precision_score(student_true_labels, student_pred_labels, average='weighted')
        student_recall = recall_score(student_true_labels, student_pred_labels, average='weighted')
        student_f1 = f1_score(student_true_labels, student_pred_labels, average='weighted')
        
        logger.info(f"Student Model Metrics - Accuracy: {student_accuracy:.4f}, Precision: {student_precision:.4f}, Recall: {student_recall:.4f}, F1: {student_f1:.4f}")
    
    return student_model

def evaluate_model(model, data, device):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.y.size(0)
        
        # Calculate additional metrics
        true_labels = data.y.cpu().numpy()
        pred_labels = pred.cpu().numpy()
        
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        
    return acc, precision, recall, f1

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# For the new feature set: ['tick', 'cape', 'cattle', 'bio5', 'cluster']
input_dim = 5  # Updated to match feature cols + cluster
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
            teacher_acc, teacher_prec, teacher_rec, teacher_f1 = evaluate_model(teacher_model, test_data, device)
            logger.info(f"Teacher model test metrics - Accuracy: {teacher_acc:.4f}, Precision: {teacher_prec:.4f}, Recall: {teacher_rec:.4f}, F1: {teacher_f1:.4f}")
            
            # Train student model with knowledge distillation
            student_model = train_student_with_distillation(teacher_model, student_model, train_data, device)
            student_acc, student_prec, student_rec, student_f1 = evaluate_model(student_model, test_data, device)
            logger.info(f"Student model test metrics - Accuracy: {student_acc:.4f}, Precision: {student_prec:.4f}, Recall: {student_rec:.4f}, F1: {student_f1:.4f}")
            
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
    logger.info("Prediction endpoint called")
    try:
        # Get the input features
        features = data.features
        
        # Handle the case where the input has fewer features than expected
        if len(features) < input_dim:
            logger.warning(f"Input has {len(features)} features, but model expects {input_dim}. Padding with zeros.")
            features_padded = features + [0] * (input_dim - len(features))
        else:
            features_padded = features[:input_dim]
        
        # Apply scaling if scaler is available
        if scaler is not None:
            features_padded = scaler.transform([features_padded])[0]
            
        # Convert input to tensor
        X = torch.tensor([features_padded], dtype=torch.float)
        
        # Create self-loop for the single node since we need some edge structure
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
        "features": feature_cols if feature_cols else ["tick", "cape", "cattle", "bio5", "cluster"],
        "trained": os.path.exists(student_model_path),
        "device": str(device)
    }

@app.get("/metrics/")
async def get_metrics():
    """Get performance metrics for the model if available"""
    if not os.path.exists(student_model_path):
        return {"status": "Model not trained yet"}
    
    try:
        # Load and preprocess data for evaluation
        X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data()
        
        # Create graph data for test set
        test_data = create_graph_data(X_test, y_test).to(device)
        
        # Evaluate teacher model
        teacher_acc, teacher_prec, teacher_rec, teacher_f1 = evaluate_model(teacher_model, test_data, device)
        
        # Evaluate student model
        student_acc, student_prec, student_rec, student_f1 = evaluate_model(student_model, test_data, device)
        
        return {
            "teacher_model": {
                "accuracy": teacher_acc,
                "precision": teacher_prec,
                "recall": teacher_rec,
                "f1_score": teacher_f1
            },
            "student_model": {
                "accuracy": student_acc,
                "precision": student_prec,
                "recall": student_rec,
                "f1_score": student_f1
            }
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"status": "Error calculating metrics", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
