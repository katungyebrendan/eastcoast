# ECF Prediction API

This application uses a Graph Convolutional Network (GCN) to predict East Coast Fever (ECF) based on geospatial and biological features.

## Features

- FastAPI-based REST API
- PyTorch Geometric implementation of Graph Convolutional Networks
- Automatic model training on first startup
- Docker support for easy deployment
- Example client code for testing

## Directory Structure

```
.
├── app.py                  # Main application file
├── balanced_dataset.csv    # Training dataset
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── client.py               # Example client code
├── README.md               # This file
```

## Setup and Installation

### Option 1: Using Docker (Recommended)

1. Make sure Docker and Docker Compose are installed on your system
2. Clone this repository
3. Run the application:

```bash
docker-compose up -d
```

### Option 2: Local Installation

1. Make sure Python 3.9+ is installed
2. Clone this repository
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /`

Health check endpoint. Returns a simple message indicating the API is active.

### `GET /model-info/`

Returns information about the trained model.

### `POST /predict/`

Make a prediction using the trained model.

#### Request Format

```json
{
  "features": [0.5, -0.2, 0.3, 0.1, -0.5, 0.7, 0.2],
  "edges": [[0, 0]]
}
```

- `features`: List of 7 standardized features in the order: [genotype, longitude, latitude, tick, cape, cattle, bio5]
- `edges`: List of edge connections for the graph structure

#### Response Format

```json
{
  "prediction": 0,
  "probability": 0.85,
  "class_probabilities": {
    "class_0": 0.85,
    "class_1": 0.15
  }
}
```

## Testing

You can use the included `client.py` script to test the API:

```bash
python client.py
```

## Deployment on Render

To deploy this application on Render:

1. Create a new Web Service
2. Connect your GitHub repository
3. Choose "Docker" as the runtime
4. Set the environment variables (if needed)
5. Deploy!

## Troubleshooting

If you encounter any issues:

1. Check the logs using `docker-compose logs` if using Docker
2. Verify that the `balanced_dataset.csv` file is in the correct location
3. Ensure all dependencies are correctly installed

## Model Details

- **Model Type**: Graph Convolutional Network (GCN)
- **Input Features**: 7 (genotype, longitude, latitude, tick, cape, cattle, bio5)
- **Hidden Layers**: 1 layer with 32 neurons
- **Output Classes**: 2 (Binary classification for ECF)
