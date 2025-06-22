#!/usr/bin/env python3
"""
Network Traffic Capture Module

This module provides functionality to capture network traffic either from PCAP files
or through live capture using Scapy.
"""

import os
import argparse
from datetime import datetime
from scapy.all import sniff, wrpcap, rdpcap
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrafficCapture:
    def __init__(self, output_dir="data"):
        """Initialize the TrafficCapture class.
        
        Args:
            output_dir (str): Directory to save captured packets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def live_capture(self, interface, duration=60, packet_count=None):
        """Capture live network traffic.
        
        Args:
            interface (str): Network interface to capture from
            duration (int): Duration of capture in seconds
            packet_count (int): Number of packets to capture (None for unlimited)
            
        Returns:
            str: Path to the saved PCAP file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"capture_{timestamp}.pcap")
        
        logger.info(f"Starting live capture on interface {interface}")
        
        try:
            packets = sniff(
                iface=interface,
                timeout=duration,
                count=packet_count,
                store=True
            )
            
            wrpcap(output_file, packets)
            logger.info(f"Captured {len(packets)} packets. Saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error during capture: {str(e)}")
            raise

    def read_pcap(self, pcap_file):
        """Read packets from a PCAP file.
        
        Args:
            pcap_file (str): Path to the PCAP file
            
        Returns:
            list: List of Scapy packet objects
        """
        try:
            logger.info(f"Reading packets from {pcap_file}")
            packets = rdpcap(pcap_file)
            logger.info(f"Successfully read {len(packets)} packets")
            return packets
            
        except Exception as e:
            logger.error(f"Error reading PCAP file: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Network Traffic Capture Tool")
    parser.add_argument("--interface", help="Network interface to capture from")
    parser.add_argument("--duration", type=int, default=60,
                      help="Duration of capture in seconds")
    parser.add_argument("--count", type=int,
                      help="Number of packets to capture")
    parser.add_argument("--output", default="data/capture.pcap",
                      help="Output PCAP file path")
    parser.add_argument("--read", help="Read packets from PCAP file")
    
    args = parser.parse_args()
    
    capture = TrafficCapture()
    
    if args.read:
        packets = capture.read_pcap(args.read)
        print(f"Read {len(packets)} packets from {args.read}")
    elif args.interface:
        output_file = capture.live_capture(
            args.interface,
            duration=args.duration,
            packet_count=args.count
        )
        print(f"Captured packets saved to {output_file}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 