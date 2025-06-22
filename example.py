#!/usr/bin/env python3
"""
Example Script

This script demonstrates how to use the network traffic anomaly detection system.
"""

import os
import pandas as pd
from scapy.all import get_if_list
from src.data_ingestion.capture import TrafficCapture
from src.feature_extraction.extract import FeatureExtractor
from src.models.kmeans_detector import KMeansAnomalyDetector
from src.models.ocsvm_detector import OCSVMAnomalyDetector
from src.models.autoencoder_detector import AutoencoderAnomalyDetector
from src.visualization.visualizer import AnomalyVisualizer
import numpy as np

def get_available_interfaces():
    """Get list of available network interfaces."""
    try:
        interfaces = get_if_list()
        if not interfaces:
            print("No network interfaces found!")
            return None
        return interfaces
    except Exception as e:
        print(f"Error getting network interfaces: {str(e)}")
        return None

def main():
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Get available interfaces
    interfaces = get_available_interfaces()
    if not interfaces:
        print("Please check your network interfaces and try again.")
        return
    
    # Print available interfaces
    print("\nAvailable network interfaces:")
    for i, iface in enumerate(interfaces):
        print(f"{i+1}. {iface}")
    
    # Let user select interface
    while True:
        try:
            choice = int(input("\nSelect interface number (1-{}): ".format(len(interfaces))))
            if 1 <= choice <= len(interfaces):
                interface = interfaces[choice-1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nSelected interface: {interface}")
    
    # Initialize components
    capture = TrafficCapture()
    extractor = FeatureExtractor()
    visualizer = AnomalyVisualizer()
    
    # Example 1: Live capture and analysis
    print("\nExample 1: Live capture and analysis")
    print("-----------------------------------")
    
    try:
        # Capture live traffic
        pcap_file = capture.live_capture(interface, duration=60)
        print(f"Captured traffic saved to {pcap_file}")
        
        # Extract features
        packets = capture.read_pcap(pcap_file)
        features_df = extractor.extract_features(packets)
        print(f"Extracted {len(features_df)} flows")
        
        # Train and evaluate different models
        models = {
            "K-Means": KMeansAnomalyDetector(),
            "One-Class SVM": OCSVMAnomalyDetector(),
            "Autoencoder": AutoencoderAnomalyDetector()
        }
        
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            model.train(features_df)
            
            # Get predictions and scores
            predictions = model.predict(features_df)
            scores = model.get_anomaly_scores(features_df)
            
            # Save model
            model.save_model(f"{name.lower().replace('-', '_')}")
            
            # Create visualizations
            print(f"Creating visualizations for {name}...")
            
            # Plot anomaly scores
            fig = visualizer.plot_anomaly_scores(scores)
            visualizer.save_plot(fig, f"results/{name.lower().replace('-', '_')}_scores")
            
            # Plot PCA visualization
            fig = visualizer.plot_pca_visualization(features_df, predictions)
            visualizer.save_plot(fig, f"results/{name.lower().replace('-', '_')}_pca")
            
            # Print statistics
            print(f"Anomalies detected: {np.sum(predictions)}")
            print(f"Anomaly rate: {np.mean(predictions):.2%}")
        
        # Example 2: Load and use a trained model
        print("\nExample 2: Load and use a trained model")
        print("---------------------------------------")
        
        # Load the K-Means model
        model = KMeansAnomalyDetector()
        model.load_model("kmeans")
        
        # Make predictions on new data
        new_pcap_file = capture.live_capture(interface, duration=30)
        new_packets = capture.read_pcap(new_pcap_file)
        new_features = extractor.extract_features(new_packets)
        
        predictions = model.predict(new_features)
        scores = model.get_anomaly_scores(new_features)
        
        print(f"New data analysis:")
        print(f"Total flows: {len(new_features)}")
        print(f"Anomalies detected: {np.sum(predictions)}")
        print(f"Anomaly rate: {np.mean(predictions):.2%}")
        
        # Create visualization for new data
        fig = visualizer.plot_anomaly_scores(scores)
        visualizer.save_plot(fig, "results/new_data_scores")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")

if __name__ == "__main__":
    main() 