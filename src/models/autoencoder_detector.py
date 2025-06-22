#!/usr/bin/env python3
"""
Autoencoder Anomaly Detector

This module implements anomaly detection using a deep autoencoder.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
from .base_model import BaseAnomalyDetector
import logging
import os

logger = logging.getLogger(__name__)

class AutoencoderAnomalyDetector(BaseAnomalyDetector):
    def __init__(self, encoding_dim=32, contamination=0.1, model_dir="models"):
        """Initialize the Autoencoder anomaly detector.
        
        Args:
            encoding_dim (int): Dimension of the encoded representation
            contamination (float): Expected proportion of anomalies
            model_dir (str): Directory to save/load models
        """
        super().__init__(model_dir)
        self.encoding_dim = encoding_dim
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.threshold = None

    def _build_autoencoder(self, input_dim):
        """Build the autoencoder model.
        
        Args:
            input_dim (int): Dimension of the input data
            
        Returns:
            tuple: (encoder, decoder, autoencoder) models
        """
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        
        # Compile
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return encoder, autoencoder

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

    def train(self, data, epochs=50, batch_size=32):
        """Train the autoencoder model.
        
        Args:
            data (pd.DataFrame): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        X_scaled = self.preprocess(data)
        
        # Build and train autoencoder
        _, self.model = self._build_autoencoder(X_scaled.shape[1])
        
        # Train the model
        self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            verbose=1
        )
        
        # Calculate reconstruction error on training data
        X_pred = self.model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        
        # Set threshold based on contamination
        self.threshold = np.percentile(mse, (1 - self.contamination) * 100)
        logger.info(f"Autoencoder model trained with encoding dimension {self.encoding_dim}")

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
        
        # Calculate reconstruction error
        X_pred = self.model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        
        # Predict anomalies based on threshold
        predictions = (mse > self.threshold).astype(int)
        
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
        
        # Calculate reconstruction error
        X_pred = self.model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        
        # Normalize scores to [0, 1]
        scores = (mse - np.min(mse)) / (np.max(mse) - np.min(mse))
        
        return scores

    def save_model(self, model_name):
        """Save the trained model to disk.
        
        Args:
            model_name (str): Name of the model file
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        # Save the model
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        self.model.save(model_path)
        
        # Save the threshold
        threshold_path = os.path.join(self.model_dir, f"{model_name}_threshold.npy")
        np.save(threshold_path, self.threshold)
        
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_name):
        """Load a trained model from disk.
        
        Args:
            model_name (str): Name of the model file
        """
        # Load the model
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the threshold
        threshold_path = os.path.join(self.model_dir, f"{model_name}_threshold.npy")
        if os.path.exists(threshold_path):
            self.threshold = np.load(threshold_path)
        
        logger.info(f"Model loaded from {model_path}") 