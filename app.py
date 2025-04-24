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
import traceback
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
    edges: List[List[int]] = []  # Optional edges

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

# Define Student Model (Smaller GNN)
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
        
        # Select features (using only those specified in the second code sample)
        feature_cols = ['tick', 'cape', 'cattle', 'bio5']
        X = df[feature_cols].values
        y = df['ECF'].values
        
        # Apply SMOTE to balance the dataset
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"After SMOTE: {len(X_resampled)} samples")
        
        # Apply K-Means clustering with exactly 3 clusters
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
        
        return X_train, X_test, y_train, y_test, scaler, feature_cols, kmeans
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_graph_data(X, y=None):
    """Create a graph from feature data using Erdős-Rényi model with p=0.1"""
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

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model paths
teacher_model_path = "teacher_model.pth"
student_model_path = "student_model.pth"
scaler = None
feature_cols = []
kmeans_model = None
input_dim = 5  # 4 features + 1 cluster feature

# Initialize models with dimensions matching the second code
teacher_hidden_dim = 64
student_hidden_dim = 32
output_dim = 2  # Binary classification

# Initialize models with proper dimensions
teacher_model = TeacherGNN(input_dim, teacher_hidden_dim, output_dim).to(device)
student_model = StudentGNN(input_dim, student_hidden_dim, output_dim).to(device)

@app.on_event("startup")
async def startup_event():
    global teacher_model, student_model, scaler, feature_cols, kmeans_model
    
    try:
        logger.info("Starting up the ECF Prediction API...")
        
        # Check if the student model file exists
        if os.path.exists(student_model_path) and os.path.exists("model_metadata.pt"):
            logger.info("Found existing model files. Loading models...")
            try:
                # Load student model
                student_state_dict = torch.load(student_model_path, map_location=device)
                student_model.load_state_dict(student_state_dict)
                logger.info("Student model loaded successfully")
                
                # Load metadata
                metadata = torch.load("model_metadata.pt", map_location=device)
                scaler = metadata.get("scaler")
                feature_cols = metadata.get("feature_cols", [])
                kmeans_model = metadata.get("kmeans_model")
                
                if scaler is None:
                    logger.warning("Scaler not found in metadata")
                if not feature_cols:
                    logger.warning("Feature columns not found in metadata")
                if kmeans_model is None:
                    logger.warning("KMeans model not found in metadata")
                
                logger.info(f"Loaded model metadata. Features: {feature_cols}")
                
                # Also load teacher model if it exists
                if os.path.exists(teacher_model_path):
                    teacher_state_dict = torch.load(teacher_model_path, map_location=device)
                    teacher_model.load_state_dict(teacher_state_dict)
                    logger.info("Teacher model loaded successfully")
                else:
                    logger.warning("Teacher model file not found")
            except Exception as load_error:
                logger.error(f"Error loading models: {load_error}")
                # Fall back to training new models
                logger.info("Falling back to training new models...")
        else:
            logger.info("No existing models found. Training new models with knowledge distillation...")
        
        # If models weren't loaded successfully, train new ones
        if student_model is None or not hasattr(student_model, 'conv1'):
            # Check if dataset exists
            if not os.path.exists('balanced_dataset.csv'):
                logger.error("Dataset file 'balanced_dataset.csv' not found! Cannot train models.")
                # Continue with untrained models
                return
            
            try:
                # Load and preprocess data
                X_train, X_test, y_train, y_test, scaler, feature_cols, kmeans_model = load_and_preprocess_data()
                logger.info(f"Data loaded and preprocessed. Training set size: {X_train.shape[0]}")
                
                # Create graph data
                train_data = create_graph_data(X_train, y_train).to(device)
                test_data = create_graph_data(X_test, y_test).to(device)
                logger.info(f"Graph data created. Number of nodes: {train_data.num_nodes}, Number of edges: {train_data.num_edges}")
                
                # Train teacher model
                teacher_model = train_teacher_model(teacher_model, train_data, device, epochs=100)
                
                # Train student model with knowledge distillation
                student_model = train_student_with_distillation(teacher_model, student_model, train_data, device, epochs=100)
                
                # Save models and metadata
                torch.save(teacher_model.state_dict(), teacher_model_path)
                torch.save(student_model.state_dict(), student_model_path)
                torch.save({
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    "kmeans_model": kmeans_model
                }, "model_metadata.pt")
                logger.info("Models and metadata saved successfully")
            except Exception as train_error:
                logger.error(f"Error during model training: {train_error}")
                # Continue with untrained models
        
        # Validate that models are loaded and ready
        if student_model is not None and hasattr(student_model, 'conv1'):
            logger.info("API startup completed successfully - model ready for predictions")
        else:
            logger.warning("API startup completed but models may not be initialized correctly")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue running the application even if model training fails
@app.get("/")
async def root():
    return {"message": "ECF Prediction API with Knowledge Distillation", "status": "active"}

@app.head("/")
async def head_root():
    """Handle HEAD requests to the root endpoint."""
    # HEAD requests should return the same headers as GET but with no body
    return {"message": "ECF Prediction API with Knowledge Distillation", "status": "active"}

@app.post("/")
async def post_root():
    """Handle POST requests to the root endpoint."""
    return {
        "message": "ECF Prediction API with Knowledge Distillation", 
        "status": "active",
        "note": "For predictions, use the /predict/ endpoint with the required data format"
    }

@app.options("/")
async def options_root():
    """Handle OPTIONS requests to the root endpoint."""
    from fastapi.responses import Response
    response = Response()
    response.headers["Allow"] = "GET, POST, HEAD, OPTIONS"
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

@app.post("/predict/")
async def predict(data: PredictionRequest):
    try:
        # Log the received data for debugging
        logger.info(f"Received prediction request with features: {data.features}")
        if len(data.edges) > 0:
            logger.info(f"Custom edges provided: {data.edges}")
        
        # Get the input features - expect exactly 4 features (tick, cape, cattle, bio5)
        features = data.features
        
        if len(features) != 4:
            logger.warning(f"Expected 4 features but received {len(features)}. Please provide [tick, cape, cattle, bio5]")
            raise HTTPException(status_code=400, detail="Expected 4 features: [tick, cape, cattle, bio5]")
        
        # Check if models are loaded correctly
        if student_model is None:
            logger.error("Student model not loaded. Please check if training completed successfully.")
            raise HTTPException(status_code=500, detail="Model not loaded. Server needs restarting.")
        
        # Apply clustering to get the 5th feature (cluster)
        try:
            if kmeans_model is not None:
                cluster = kmeans_model.predict([features])[0]
                logger.info(f"KMeans assigned cluster: {cluster}")
                features_with_cluster = features + [cluster]
            else:
                # If no kmeans model, use a default cluster (0)
                logger.warning("KMeans model not available. Using default cluster 0.")
                features_with_cluster = features + [0]
        except Exception as cluster_error:
            logger.error(f"Error during clustering: {cluster_error}")
            # Continue with default cluster as fallback
            features_with_cluster = features + [0]
        
        # Apply scaling if scaler is available
        try:
            if scaler is not None:
                features_scaled = scaler.transform([features_with_cluster])[0]
                logger.info(f"Features after scaling: {features_scaled}")
            else:
                features_scaled = features_with_cluster
                logger.warning("Scaler not available. Using unscaled features.")
        except Exception as scale_error:
            logger.error(f"Error during feature scaling: {scale_error}")
            # Continue with unscaled features as fallback
            features_scaled = features_with_cluster
            
        # Convert input to tensor
        X = torch.tensor([features_scaled], dtype=torch.float).to(device)
        logger.info(f"Input tensor shape: {X.shape}")
        
        # Create edge structure - either use provided edges or create a self-loop
        if data.edges and len(data.edges) > 0:
            # Use provided edges
            edges = data.edges
            edge_list = torch.tensor(edges, dtype=torch.long).t().to(device)
        else:
            # Create self-loop for the single node
            edge_list = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        
        logger.info(f"Edge list shape: {edge_list.shape}")
        
        # Make prediction with student model (smaller and faster)
        student_model.eval()
        try:
            with torch.no_grad():
                student_output = student_model(X, edge_list)
                student_probabilities = torch.exp(student_output).cpu().numpy()[0]
                student_prediction = student_output.argmax(dim=1).item()
            
            logger.info(f"Student model prediction: {student_prediction}")
            logger.info(f"Student model probabilities: {student_probabilities}")
        except Exception as student_error:
            logger.error(f"Error during student model prediction: {student_error}")
            raise HTTPException(status_code=500, detail=f"Student model prediction failed: {str(student_error)}")
        
        # Also get teacher prediction for comparison
        teacher_prediction = None
        teacher_probabilities = [0, 0]
        
        if teacher_model is not None:
            try:
                teacher_model.eval()
                with torch.no_grad():
                    teacher_output = teacher_model(X, edge_list)
                    teacher_probabilities = torch.exp(teacher_output).cpu().numpy()[0]
                    teacher_prediction = teacher_output.argmax(dim=1).item()
                logger.info(f"Teacher model prediction: {teacher_prediction}")
            except Exception as teacher_error:
                logger.error(f"Error during teacher model prediction: {teacher_error}")
                # Continue without teacher prediction
        else:
            logger.warning("Teacher model not available")
        
        # Construct response
        response = {
            "prediction": int(student_prediction),  # 0 or 1
            "probability": float(student_probabilities[student_prediction]),
            "risk_level": "High" if student_prediction == 1 else "Low",
            "class_probabilities": {
                "low_risk": float(student_probabilities[0]),
                "high_risk": float(student_probabilities[1])
            },
            "cluster": int(features_with_cluster[4]),
        }
        
        # Add teacher predictions if available
        if teacher_prediction is not None:
            response.update({
                "teacher_prediction": int(teacher_prediction),
                "teacher_probabilities": {
                    "low_risk": float(teacher_probabilities[0]),
                    "high_risk": float(teacher_probabilities[1])
                }
            })
        
        logger.info(f"Returning prediction response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return more descriptive error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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
        "features": feature_cols if feature_cols else ["tick", "cape", "cattle", "bio5", "cluster"]
    }

# Run locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
