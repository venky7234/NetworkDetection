import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, model_type='random_forest', model_dir='models'):
        """
        Initialize the anomaly detector with specified model type
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'isolation_forest', or 'autoencoder')
            model_dir (str): Directory to save/load models
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate model based on model_type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
        elif self.model_type == 'autoencoder':
            self.model = self._build_autoencoder()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _build_autoencoder(self):
        """Build and return an autoencoder model"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(None,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(None, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def preprocess_data(self, data):
        """
        Preprocess the input data
        
        Args:
            data (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Preprocessed data
        """
        # Scale the features
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data
    
    def train(self, X, y=None, validation_split=0.2):
        """
        Train the model on the provided data
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series, optional): Training labels (not needed for isolation forest)
            validation_split (float): Proportion of data to use for validation
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Preprocess the data
        X_scaled = self.preprocess_data(X)
        
        if self.model_type == 'autoencoder':
            # For autoencoder, we don't need labels
            self.model.fit(
                X_scaled,
                X_scaled,
                epochs=50,
                batch_size=32,
                validation_split=validation_split,
                verbose=1
            )
        elif self.model_type == 'isolation_forest':
            # Isolation Forest doesn't need labels
            self.model.fit(X_scaled)
        else:
            # For supervised models (Random Forest)
            if y is None:
                raise ValueError("Labels (y) are required for supervised models")
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            logger.info("\nValidation Results:")
            logger.info(classification_report(y_val, y_pred))
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predictions (1 for normal, -1 for anomaly)
        """
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'autoencoder':
            # For autoencoder, calculate reconstruction error
            predictions = self.model.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
            # Convert to binary predictions (1 for normal, -1 for anomaly)
            return np.where(mse < np.percentile(mse, 90), 1, -1)
        elif self.model_type == 'isolation_forest':
            # Isolation Forest returns -1 for anomalies, 1 for normal
            return self.model.predict(X_scaled)
        else:
            # For supervised models
            return self.model.predict(X_scaled)
    
    def save_model(self, filename=None):
        """
        Save the trained model to disk
        
        Args:
            filename (str, optional): Name of the file to save the model
        """
        if filename is None:
            filename = f"{self.model_type}_model"
        
        model_path = os.path.join(self.model_dir, filename)
        
        if self.model_type == 'autoencoder':
            self.model.save(f"{model_path}.h5")
        else:
            joblib.dump(self.model, f"{model_path}.joblib")
        
        # Save the scaler
        joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename=None):
        """
        Load a trained model from disk
        
        Args:
            filename (str, optional): Name of the file to load the model from
        """
        if filename is None:
            filename = f"{self.model_type}_model"
        
        model_path = os.path.join(self.model_dir, filename)
        
        if self.model_type == 'autoencoder':
            self.model = load_model(f"{model_path}.h5")
        else:
            self.model = joblib.load(f"{model_path}.joblib")
        
        # Load the scaler
        self.scaler = joblib.load(f"{model_path}_scaler.joblib")
        logger.info(f"Model loaded from {model_path}")
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
        Args:
            X (pd.DataFrame): Test features
            y (pd.Series): True labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        X_scaled = self.scaler.transform(X)
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred)
        }
        
        return metrics 