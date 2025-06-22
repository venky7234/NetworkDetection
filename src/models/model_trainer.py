import argparse
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(input_path, model_type):
    """
    Train a machine learning model for anomaly detection.
    
    Args:
        input_path (str): Path to the input feature CSV file
        model_type (str): Type of model to train ('kmeans', 'isolation_forest', or 'ocsvm')
    """
    logger.info(f"Loading features from: {input_path}")
    df = pd.read_csv(input_path)

    # Select only numeric columns
    df_numeric = df.select_dtypes(include=["number"])
    
    if df_numeric.empty:
        raise ValueError("No numeric columns found in the input data")

    # Initialize the appropriate model
    if model_type == "kmeans":
        model = KMeans(n_clusters=1, random_state=42)
    elif model_type == "isolation_forest":
        model = IsolationForest(contamination=0.1, random_state=42)
    elif model_type == "ocsvm":
        model = OneClassSVM(gamma='auto')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Training {model_type} model...")
    model.fit(df_numeric)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_type}_model.pkl"
    
    # Save the trained model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--input", required=True, help="Path to input feature CSV file")
    parser.add_argument("--model", required=True, 
                       choices=['kmeans', 'isolation_forest', 'ocsvm'],
                       help="Model type: kmeans | isolation_forest | ocsvm")
    args = parser.parse_args()

    try:
        model_path = train_model(args.input, args.model)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise 