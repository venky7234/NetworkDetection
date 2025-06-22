#!/usr/bin/env python3
"""
K-Means Anomaly Detector

This module implements anomaly detection using K-Means clustering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .base_model import BaseAnomalyDetector
import logging

logger = logging.getLogger(__name__)

class KMeansAnomalyDetector(BaseAnomalyDetector):
    def __init__(self, n_clusters=5, contamination=0.1, model_dir="models"):
        """Initialize the K-Means anomaly detector.
        
        Args:
            n_clusters (int): Number of clusters
            contamination (float): Expected proportion of anomalies
            model_dir (str): Directory to save/load models
        """
        super().__init__(model_dir)
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.threshold = None

    def preprocess(self, data):
        """Preprocess the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Preprocessed data
        """
        # Select numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numerical_cols].values
        
        # Scale the features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled

    def train(self, data):
        """Train the K-Means model.
        
        Args:
            data (pd.DataFrame): Training data
        """
        X_scaled = self.preprocess(data)
        
        # Train K-Means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        self.model.fit(X_scaled)
        
        # Calculate distances to cluster centers
        distances = np.min(self.model.transform(X_scaled), axis=1)
        
        # Set threshold based on contamination
        self.threshold = np.percentile(distances, (1 - self.contamination) * 100)
        logger.info(f"K-Means model trained with {self.n_clusters} clusters")

    def predict(self, data):
        """Predict anomalies in the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Anomaly scores (1 for anomaly, 0 for normal)
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        X_scaled = self.preprocess(data)
        
        # Calculate distances to cluster centers
        distances = np.min(self.model.transform(X_scaled), axis=1)
        
        # Predict anomalies based on threshold
        predictions = (distances > self.threshold).astype(int)
        
        return predictions

    def get_anomaly_scores(self, data):
        """Get anomaly scores for the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Anomaly scores (higher values indicate more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained")
            
        X_scaled = self.preprocess(data)
        distances = np.min(self.model.transform(X_scaled), axis=1)
        
        # Normalize scores to [0, 1]
        scores = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        
        return scores 