#!/usr/bin/env python3
"""
Streamlit Dashboard

This module provides a web-based dashboard for monitoring network traffic anomalies.
"""

import sys
import os

# Add project root folder to sys.path dynamically so imports from src work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scapy.all import get_if_list

from src.data_ingestion.capture import TrafficCapture
from src.feature_extraction.extract import FeatureExtractor
from src.models.kmeans_detector import KMeansAnomalyDetector
from src.models.detector import OCSVMAnomalyDetector
from src.models.autoencoder_detector import AutoencoderAnomalyDetector
from src.visualization.visualizer import AnomalyVisualizer


def get_available_interfaces():
    """Get list of available network interfaces."""
    try:
        interfaces = get_if_list()
        if not interfaces:
            st.error("No network interfaces found!")
            return ["No interfaces available"]
        return interfaces
    except Exception as e:
        st.error(f"Error getting network interfaces: {str(e)}")
        return ["Error getting interfaces"]


def main():
    st.set_page_config(
        page_title="Network Traffic Anomaly Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Network Traffic Anomaly Detection Dashboard")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["K-Means", "One-Class SVM", "Autoencoder"]
    )
    
    # Live capture options
    st.sidebar.header("Live Capture")
    
    # Get available interfaces and create dropdown
    available_interfaces = get_available_interfaces()
    interface = st.sidebar.selectbox(
        "Network Interface",
        available_interfaces,
        index=0 if available_interfaces else None
    )
    
    duration = st.sidebar.number_input("Capture Duration (seconds)", 60, 3600, 300)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Live Monitoring", "Analysis", "Settings"])
    
    with tab1:
        st.header("Live Traffic Monitoring")
        
        if st.button("Start Capture"):
            if interface == "No interfaces available" or interface == "Error getting interfaces":
                st.error("Please check your network interfaces and try again.")
            else:
                with st.spinner("Capturing network traffic..."):
                    try:
                        # Initialize components
                        capture = TrafficCapture()
                        extractor = FeatureExtractor()
                        visualizer = AnomalyVisualizer()
                        
                        # Select model
                        if model_type == "K-Means":
                            model = KMeansAnomalyDetector()
                        elif model_type == "One-Class SVM":
                            model = OCSVMAnomalyDetector()
                        else:
                            model = AutoencoderAnomalyDetector()
                        
                        # Capture traffic
                        pcap_file = capture.live_capture(interface, duration=duration)
                        
                        # Extract features
                        packets = capture.read_pcap(pcap_file)
                        features_df = extractor.extract_features(packets)
                        
                        # Display feature distributions
                        st.subheader("Feature Distributions")
                        fig = visualizer.plot_feature_distributions(features_df)
                        st.plotly_chart(fig)
                        
                        # Train model and detect anomalies
                        model.train(features_df)
                        predictions = model.predict(features_df)
                        scores = model.get_anomaly_scores(features_df)
                        
                        # Display anomaly scores
                        st.subheader("Anomaly Detection Results")
                        fig = visualizer.plot_anomaly_scores(scores)
                        st.plotly_chart(fig)
                        
                        # Display PCA visualization
                        st.subheader("PCA Visualization")
                        fig = visualizer.plot_pca_visualization(features_df, predictions)
                        st.plotly_chart(fig)
                        
                        # Display statistics
                        st.subheader("Detection Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Flows", len(features_df))
                        with col2:
                            st.metric("Anomalies Detected", np.sum(predictions))
                        with col3:
                            st.metric("Anomaly Rate", f"{np.mean(predictions):.2%}")
                            
                    except Exception as e:
                        st.error(f"Error during capture: {str(e)}")
    
    with tab2:
        st.header("Traffic Analysis")
        
        # File uploader for PCAP files
        uploaded_file = st.file_uploader("Upload PCAP file", type=['pcap'])
        
        if uploaded_file is not None:
            try:
                # Save uploaded file
                pcap_path = f"data/uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap"
                with open(pcap_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Process the file
                capture = TrafficCapture()
                extractor = FeatureExtractor()
                visualizer = AnomalyVisualizer()
                
                # Select model
                if model_type == "K-Means":
                    model = KMeansAnomalyDetector()
                elif model_type == "One-Class SVM":
                    model = OCSVMAnomalyDetector()
                else:
                    model = AutoencoderAnomalyDetector()
                
                # Extract features
                packets = capture.read_pcap(pcap_path)
                features_df = extractor.extract_features(packets)
                
                # Display feature analysis
                st.subheader("Feature Analysis")
                fig = visualizer.plot_feature_distributions(features_df)
                st.plotly_chart(fig)
                
                # Train model and detect anomalies
                model.train(features_df)
                predictions = model.predict(features_df)
                scores = model.get_anomaly_scores(features_df)
                
                # Display results
                st.subheader("Anomaly Detection Results")
                fig = visualizer.plot_anomaly_scores(scores)
                st.plotly_chart(fig)
                
                # Display PCA visualization
                st.subheader("PCA Visualization")
                fig = visualizer.plot_pca_visualization(features_df, predictions)
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("Model Settings")
        
        if model_type == "K-Means":
            n_clusters = st.number_input("Number of Clusters", 2, 20, 5)
            contamination = st.slider("Expected Anomaly Rate", 0.01, 0.5, 0.1)
            
        elif model_type == "One-Class SVM":
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            nu = st.slider("Nu (Expected Anomaly Rate)", 0.01, 0.5, 0.1)
            gamma = st.selectbox("Gamma", ["scale", "auto"])
            
        else:  # Autoencoder
            encoding_dim = st.number_input("Encoding Dimension", 8, 128, 32)
            contamination = st.slider("Expected Anomaly Rate", 0.01, 0.5, 0.1)
            epochs = st.number_input("Training Epochs", 10, 200, 50)
            batch_size = st.number_input("Batch Size", 16, 256, 32)


if __name__ == "__main__":
    main()
