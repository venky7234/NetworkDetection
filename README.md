<<<<<<< HEAD
# Network Traffic Anomaly Detection using Machine Learning

This project implements a machine learning-based system for detecting anomalies in network traffic. It uses various ML models including K-Means clustering, One-Class SVM, and Autoencoders to identify unusual patterns in network data.

## Features

- Network traffic data collection from PCAP files or live capture
- Feature extraction and preprocessing
- Multiple ML models for anomaly detection
- Visualization of results
- Real-time monitoring capabilities (optional)

## Project Structure

```
├── data/                  # Directory for storing PCAP files and processed data
├── src/                   # Source code
│   ├── data_ingestion/    # Data collection and parsing
│   ├── feature_extraction/# Feature engineering
│   ├── models/           # ML model implementations
│   ├── evaluation/       # Model evaluation scripts
│   └── visualization/    # Visualization tools
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd network-traffic-anomaly-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On
 Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Collection:
```bash
python src/data_ingestion/capture.py --interface eth0 --output data/capture.pcap
```

2. Feature Extraction:
```bash
python src/feature_extraction/extract.py --input data/capture.pcap --output data/features.csv
```

3. Model Training:
```bash
python src/models/train.py --input data/features.csv --model kmeans
```

4. Anomaly Detection:
```bash
python src/models/detect.py --input data/test.pcap --model kmeans
```
"""python -m src.models.detector --input data/test.pcap --model kmeans"""

5. Visualization (Optional):
```bash
streamlit run src/visualization/dashboard.py
```

## Models

The system implements three different anomaly detection approaches:

1. K-Means Clustering
2. One-Class SVM
3. Deep Autoencoder

Each model can be trained and evaluated independently.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
=======
# NetworkDetection
>>>>>>> 889be53ba236cf0acb9f539c03cf1bd78a63c46f
