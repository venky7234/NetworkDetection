import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data.data_ingestion import PacketCapture
from src.features.feature_extraction import FeatureExtractor
from src.models.model import AnomalyDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize packet capture
    packet_capture = PacketCapture(interface='Wi-Fi', output_file='captured_packets.pcap')
    
    # Start capturing packets for training
    logger.info("Starting packet capture for training data...")
    packet_capture.start_capture(duration=300)  # Capture for 5 minutes
    
    # Extract features from captured packets
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract_features('captured_packets.pcap')
    
    # Split data into training and testing sets
    train_size = int(0.8 * len(features_df))
    train_data = features_df[:train_size]
    test_data = features_df[train_size:]
    
    # Train different types of models
    model_types = ['random_forest', 'isolation_forest', 'autoencoder']
    
    for model_type in model_types:
        logger.info(f"\nTraining {model_type} model...")
        
        # Initialize model
        model = AnomalyDetector(model_type=model_type)
        
        # For supervised models (random_forest), we need labels
        # In a real scenario, you would have labeled data
        # For this example, we'll create synthetic labels
        if model_type == 'random_forest':
            # Create synthetic labels (1 for normal, 0 for anomaly)
            # In reality, you would have real labels
            y_train = pd.Series(1, index=train_data.index)  # All normal for this example
            y_test = pd.Series(1, index=test_data.index)
            
            # Train the model
            model.train(train_data, y_train)
            
            # Evaluate the model
            metrics = model.evaluate(test_data, y_test)
            logger.info(f"\nEvaluation metrics for {model_type}:")
            logger.info(metrics['classification_report'])
        else:
            # For unsupervised models, we don't need labels
            model.train(train_data)
        
        # Save the trained model
        model.save_model()
        
        # Make predictions on test data
        predictions = model.predict(test_data)
        logger.info(f"\nPredictions for {model_type}:")
        logger.info(f"Number of anomalies detected: {sum(predictions == -1)}")
        logger.info(f"Number of normal packets: {sum(predictions == 1)}")

if __name__ == "__main__":
    main() 