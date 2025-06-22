#!/usr/bin/env python3
"""
One-Class SVM Anomaly Detector

This module implements anomaly detection using One-Class SVM.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from src.models.base_model import BaseAnomalyDetector
import logging

logger = logging.getLogger(__name__)

class OCSVMAnomalyDetector(BaseAnomalyDetector):
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale', model_dir="models"):
        """Initialize the One-Class SVM anomaly detector.
        
        Args:
            nu (float): An upper bound on the fraction of training errors
            kernel (str): Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma (str or float): Kernel coefficient
            model_dir (str): Directory to save/load models
        """
        super().__init__(model_dir)
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = StandardScaler()

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
        """Train the One-Class SVM model.
        
        Args:
            data (pd.DataFrame): Training data
        """
        X_scaled = self.preprocess(data)
        
        # Train One-Class SVM
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.model.fit(X_scaled)
        logger.info(f"One-Class SVM model trained with {self.kernel} kernel")

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
        
        # One-Class SVM returns -1 for anomalies and 1 for normal
        predictions = (self.model.predict(X_scaled) == -1).astype(int)
        
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
        
        # Get decision function values
        scores = -self.model.decision_function(X_scaled)
        
        # Normalize scores to [0, 1]
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        return scores 