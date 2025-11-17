"""
Visualization utilities for financial data and model outputs, including Risk dashboards, 
PnL plots, charts and graphs for risk metrics and regime detection.

Dependencies :
    - matplotlib
    - seaborn
    - plotly
    - pandas
    - numpy
    - pathlib
"""
# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame()

    def plot_time_series(self, df: pd.DataFrame(), column: str, title: str = "Time Series Plot"):
        """Plot a time series graph for a specified column in the DataFrame."""
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column], label=column)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_candlestick(self, df: pd.DataFrame, title: str = "Candlestick Chart"):
        """Plot a candlestick chart using Plotly."""
        fig = px.candlestick(
            df,
            x=df.index,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            title=title
        )
        fig.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame, title: str = "Correlation Heatmap"):
        """Plot a correlation heatmap for the DataFrame."""
        plt.figure(figsize=(10, 8))
        corr = self.df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(title)
        plt.show()
    
    def plot_risk_metrics(self, df: pd.DataFrame, metrics: list, title: str = "Risk Metrics Over Time"):
        """Plot multiple risk metrics over time."""
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(df.index, df[metric], label=metric)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_regime_detection(self, df: pd.DataFrame, regime_column: str, title: str = "Regime Detection"):
        """Plot regime detection results."""
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[regime_column], label='Regime', color='orange')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Regime")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_plot(self, fig, filename: str, directory: str = "plots"):
        """
        Save a plot to a specified directory.

        Args:
            fig: The figure object to save
            filename: The name of the file (without extension)
            directory: The directory to save the plot in (default: plots)
        """
        # Create output directory if it doesn't exist
        output_path = Path(directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the figure
        filepath = output_path / f"{filename}.png"
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def save_interactive_plot(self, fig, filename: str, directory: str = "plots"):
        """
        Save an interactive plotly plot to a specified directory.

        Args:
            fig: The plotly figure object to save
            filename: The name of the file (without extension)
            directory: The directory to save the plot in (default: plots)
        """
        # Create output directory if it doesn't exist
        output_path = Path(directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the figure
        filepath = output_path / f"{filename}.html"
        fig.write_html(filepath)
        print(f"Interactive plot saved to {filepath}")

