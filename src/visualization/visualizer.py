#!/usr/bin/env python3
"""
Visualization Module

This module provides functions for visualizing network traffic data and anomaly detection results.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- plotly
- kaleido (for saving plotly figures as images)

Make sure to install these before running:
pip install numpy pandas matplotlib seaborn scikit-learn plotly kaleido
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class AnomalyVisualizer:
    def __init__(self) -> None:
        """Initialize the AnomalyVisualizer class and set plot styles."""
        # Use a valid matplotlib style for seaborn (adjust if version differs)
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            logger.warning("Seaborn style not found in matplotlib; using default style.")
            plt.style.use('default')
        
        # Set seaborn color palette
        sns.set_palette("husl")

    def plot_feature_distributions(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_cols: int = 3,
        bins: Optional[int] = None,
        kde: bool = False
    ) -> plt.Figure:
        """
        Plot distributions of selected features.

        Args:
            data (pd.DataFrame): Input data
            features (list, optional): List of features to plot (None for all numerical)
            n_cols (int): Number of columns in the plot grid
            bins (int, optional): Number of bins for histogram
            kde (bool): Whether to plot KDE curve on histogram

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if features is None:
            features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not features:
            raise ValueError("No numerical features found to plot.")

        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            sns.histplot(data=data, x=feature, bins=bins, kde=kde, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {feature}')
        
        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig

    def plot_anomaly_scores(
        self,
        scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> go.Figure:
        """
        Plot anomaly scores with optional threshold.

        Args:
            scores (np.ndarray): Anomaly scores
            threshold (float, optional): Threshold for anomaly detection

        Returns:
            plotly.graph_objs._figure.Figure: Interactive plotly figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines',
            name='Anomaly Score'
        ))

        if threshold is not None:
            fig.add_trace(go.Scatter(
                y=[threshold] * len(scores),
                mode='lines',
                name='Threshold',
                line=dict(dash='dash', color='red')
            ))

        fig.update_layout(
            title='Anomaly Scores Over Time',
            xaxis_title='Sample Index',
            yaxis_title='Anomaly Score',
            showlegend=True
        )
        return fig

    def plot_pca_visualization(
        self,
        data: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        n_components: int = 2
    ) -> go.Figure:
        """
        Create PCA visualization of the data.

        Args:
            data (pd.DataFrame): Input data
            labels (np.ndarray, optional): Labels for coloring points
            n_components (int): Number of PCA components to compute (only 2 supported for plot)

        Returns:
            plotly.graph_objs._figure.Figure: Interactive PCA scatter plot
        """
        if n_components != 2:
            raise ValueError("Currently only n_components=2 is supported for plotting.")

        numerical_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numerical_cols].values
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        if labels is not None:
            pca_df['label'] = labels

        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='label' if labels is not None else None,
            title='PCA Visualization of Network Traffic',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
            }
        )
        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> plt.Figure:
        """
        Plot confusion matrix using seaborn heatmap.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            matplotlib.figure.Figure: The confusion matrix figure
        """
        cm = pd.crosstab(
            pd.Series(y_true, name='Actual'),
            pd.Series(y_pred, name='Predicted')
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        return plt.gcf()

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float
    ) -> go.Figure:
        """
        Plot ROC curve.

        Args:
            fpr (np.ndarray): False positive rates
            tpr (np.ndarray): True positive rates
            auc (float): Area Under the Curve (AUC) score

        Returns:
            plotly.graph_objs._figure.Figure: Interactive ROC curve plot
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {auc:.3f})'
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        return fig

    def plot_anomaly_timeline(
        self,
        timestamps: List[Union[datetime, str]],
        scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> go.Figure:
        """
        Plot anomaly scores over a timeline.

        Args:
            timestamps (list): List of timestamps (datetime or str)
            scores (np.ndarray): Anomaly scores
            threshold (float, optional): Threshold for anomaly detection

        Returns:
            plotly.graph_objs._figure.Figure: Interactive timeline plot
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode='lines',
            name='Anomaly Score'
        ))

        if threshold is not None:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[threshold] * len(scores),
                mode='lines',
                name='Threshold',
                line=dict(dash='dash', color='red')
            ))

        fig.update_layout(
            title='Anomaly Scores Timeline',
            xaxis_title='Time',
            yaxis_title='Anomaly Score',
            showlegend=True
        )
        return fig

    def save_plot(
        self,
        fig: Union[plt.Figure, go.Figure],
        filename: str,
        format: str = 'png'
    ) -> None:
        """
        Save a plot to a file.

        Args:
            fig (matplotlib.figure.Figure or plotly.graph_objs._figure.Figure): Plot figure to save
            filename (str): Output filename without extension
            format (str): File format (e.g., 'png', 'pdf', 'jpeg')

        Raises:
            ImportError: If Plotly image saving dependencies are missing
        """
        try:
            if isinstance(fig, go.Figure):
                # Requires kaleido installed: pip install kaleido
                fig.write_image(f"{filename}.{format}")
            else:
                fig.savefig(f"{filename}.{format}", dpi=300)
            logger.info(f"Plot saved to {filename}.{format}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            raise

