#!/usr/bin/env python3
"""
Base Model Module

This module defines the base class for all anomaly detection models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAnomalyDetector(ABC):
    def __init__(self, model_dir="models"):
        """Initialize the base anomaly detector.
        
        Args:
            model_dir (str): Directory to save/load models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.scaler = None

    @abstractmethod
    def preprocess(self, data):
        """Preprocess the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Preprocessed data
        """
        pass

    @abstractmethod
    def train(self, data):
        """Train the anomaly detection model.
        
        Args:
            data (pd.DataFrame): Training data
        """
        pass

    @abstractmethod
    def predict(self, data):
        """Predict anomalies in the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Anomaly scores or labels
        """
        pass

    def save_model(self, model_name):
        """Save the trained model to disk.
        
        Args:
            model_name (str): Name of the model file
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_name):
        """Load a trained model from disk.
        
        Args:
            model_name (str): Name of the model file
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

    def evaluate(self, data, labels):
        """Evaluate the model's performance.
        
        Args:
            data (pd.DataFrame): Test data
            labels (np.ndarray): True labels (1 for anomaly, 0 for normal)
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        predictions = self.predict(data)
        
        # Convert predictions to binary labels if they're scores
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        } 