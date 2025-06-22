#!/usr/bin/env python3
"""
Feature Extraction Module

This module extracts relevant features from network packets for anomaly detection.
Features include flow duration, packet counts, protocol types, and more.
"""

import os
import pandas as pd
import numpy as np
from scapy.all import rdpcap
from collections import defaultdict
import logging
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        """Initialize the FeatureExtractor class."""
        self.flow_features = defaultdict(lambda: {
            'start_time': None,
            'end_time': None,
            'packet_count': 0,
            'byte_count': 0,
            'protocols': set(),
            'inter_arrival_times': [],
            'last_packet_time': None
        })

    def _get_flow_key(self, packet):
        """Generate a unique key for a network flow.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            tuple: Flow key (src_ip, dst_ip, src_port, dst_port, protocol)
        """
        if 'IP' in packet:
            src_ip = packet['IP'].src
            dst_ip = packet['IP'].dst
            protocol = packet['IP'].proto
            
            # Get ports if available
            src_port = packet['TCP'].sport if 'TCP' in packet else packet['UDP'].sport if 'UDP' in packet else 0
            dst_port = packet['TCP'].dport if 'TCP' in packet else packet['UDP'].dport if 'UDP' in packet else 0
            
            return (src_ip, dst_ip, src_port, dst_port, protocol)
        return None

    def _update_flow_features(self, packet, flow_key):
        """Update flow features with information from a packet.
        
        Args:
            packet: Scapy packet object
            flow_key: Flow identifier
        """
        flow = self.flow_features[flow_key]
        packet_time = datetime.fromtimestamp(float(packet.time))

        
        # Update timestamps
        if flow['start_time'] is None:
            flow['start_time'] = packet_time
        flow['end_time'] = packet_time
        
        # Update packet and byte counts
        flow['packet_count'] += 1
        flow['byte_count'] += len(packet)
        
        # Update protocols
        for layer in packet.layers():
            flow['protocols'].add(layer.name)
        
        # Update inter-arrival times
        if flow['last_packet_time'] is not None:
            inter_arrival = (packet_time - flow['last_packet_time']).total_seconds()
            flow['inter_arrival_times'].append(inter_arrival)
        flow['last_packet_time'] = packet_time

    def extract_features(self, packets):
        """Extract features from a list of packets.
        
        Args:
            packets: List of Scapy packet objects
            
        Returns:
            pd.DataFrame: DataFrame containing extracted features
        """
        logger.info("Starting feature extraction")
        
        # Process packets
        for packet in packets:
            flow_key = self._get_flow_key(packet)
            if flow_key:
                self._update_flow_features(packet, flow_key)
        
        # Convert flow features to DataFrame
        features = []
        for flow_key, flow in self.flow_features.items():
            src_ip, dst_ip, src_port, dst_port, protocol = flow_key
            
            # Calculate flow duration
            duration = (flow['end_time'] - flow['start_time']).total_seconds()
            
            # Calculate inter-arrival time statistics
            inter_arrival_times = flow['inter_arrival_times']
            mean_iat = np.mean(inter_arrival_times) if inter_arrival_times else 0
            std_iat = np.std(inter_arrival_times) if inter_arrival_times else 0
            
            feature_dict = {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'duration': duration,
                'packet_count': flow['packet_count'],
                'byte_count': flow['byte_count'],
                'mean_inter_arrival_time': mean_iat,
                'std_inter_arrival_time': std_iat,
                'protocol_count': len(flow['protocols'])
            }
            features.append(feature_dict)
        
        df = pd.DataFrame(features)
        logger.info(f"Extracted features for {len(df)} flows")
        return df

    def save_features(self, df, output_file):
        """Save extracted features to a CSV file.
        
        Args:
            df: DataFrame containing features
            output_file: Path to output CSV file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Features saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Network Traffic Feature Extraction")
    parser.add_argument("--input", required=True, help="Input PCAP file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor()
    packets = rdpcap(args.input)
    features_df = extractor.extract_features(packets)
    extractor.save_features(features_df, args.output)

if __name__ == "__main__":
    main() 