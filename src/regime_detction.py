"""
ML/stochastic regime classification using Hidden Markov Models (HMM)

This module implements regime detection using Hidden Markov Models (HMM) to classify 
market regimes based on historical price data. It provides functionality to fit an HMM 
to the data, predict regimes, and visualize the results.

Dependencies:
    - pandas
    - numpy
    - scikit-learn
    - hmmlearn
    - matplotlib
    - seaborn
"""

# Standard library imports
from typing import Tuple, Optional
import warnings

# Third party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class RegimeDetector:
    """
    Hidden Markov Model based regime detection for financial time series.
    
    This class fits a Gaussian HMM to identify distinct market regimes based on
    features such as returns, volatility, and volume.
    """
    
    def __init__(self, n_regimes: int = 3, n_iter: int = 100, random_state: int = 42):
        """
        Initialize the RegimeDetector.
        
        Args:
            n_regimes: Number of hidden states (market regimes) to detect
            n_iter: Number of iterations for the EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.regimes = None
        self.regime_probs = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for regime detection.
        
        Args:
            df: DataFrame with price data (must have 'Close', 'Volume' columns)
            
        Returns:
            DataFrame with computed features
        """
        features = pd.DataFrame(index=df.index)
        
        # Find Close and Volume columns
        close_col = [col for col in df.columns if 'Close' in col][0]
        volume_col = [col for col in df.columns if 'Volume' in col][0]
        
        # Calculate returns
        features['returns'] = df[close_col].pct_change()
        
        # Calculate volatility (rolling standard deviation of returns)
        features['volatility'] = features['returns'].rolling(window=21).std()
        
        # Calculate volume changes
        features['volume_change'] = df[volume_col].pct_change()
        
        # Calculate momentum indicators
        features['momentum_5'] = df[close_col].pct_change(5)
        features['momentum_21'] = df[close_col].pct_change(21)
        
        # Calculate moving average ratios
        ma_short = df[close_col].rolling(window=20).mean()
        ma_long = df[close_col].rolling(window=50).mean()
        features['ma_ratio'] = ma_short / ma_long - 1
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def fit(self, features: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit the HMM model to the features.
        
        Args:
            features: DataFrame with features for regime detection
            
        Returns:
            Self for method chaining
        """
        # Scale features
        X = self.scaler.fit_transform(features.values)
        
        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(X)
        
        return self
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes for the given features.
        
        Args:
            features: DataFrame with features for regime detection
            
        Returns:
            Tuple of (regime labels, regime probabilities)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale features
        X = self.scaler.transform(features.values)
        
        # Predict regimes
        self.regimes = self.model.predict(X)
        self.regime_probs = self.model.predict_proba(X)
        
        return self.regimes, self.regime_probs
    
    def fit_predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model and predict regimes in one step.
        
        Args:
            features: DataFrame with features for regime detection
            
        Returns:
            Tuple of (regime labels, regime probabilities)
        """
        self.fit(features)
        return self.predict(features)
    
    def get_regime_statistics(self, features: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """
        Calculate statistics for each regime.
        
        Args:
            features: DataFrame with features
            regimes: Array of regime labels
            
        Returns:
            DataFrame with regime statistics
        """
        stats = []
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_features = features[mask]
            
            stats.append({
                'Regime': regime,
                'Count': mask.sum(),
                'Percentage': mask.sum() / len(regimes) * 100,
                'Mean_Return': regime_features['returns'].mean(),
                'Std_Return': regime_features['returns'].std(),
                'Mean_Volatility': regime_features['volatility'].mean(),
            })
        
        return pd.DataFrame(stats)
    
    def plot_regimes(self, df: pd.DataFrame, features: pd.DataFrame, 
                     regimes: np.ndarray, save_path: Optional[str] = None):
        """
        Plot price series with regime overlay.
        
        Args:
            df: Original DataFrame with price data
            features: DataFrame with features (for matching indices)
            regimes: Array of regime labels
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Find Close column
        close_col = [col for col in df.columns if 'Close' in col][0]
        
        # Align data
        aligned_df = df.loc[features.index]
        
        # Plot price with regime colors
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for regime in range(self.n_regimes):
            mask = regimes == regime
            ax1.scatter(features.index[mask], aligned_df[close_col][mask], 
                       c=colors[regime], label=f'Regime {regime}', alpha=0.6, s=10)
        
        ax1.plot(aligned_df.index, aligned_df[close_col], 'k-', alpha=0.3, linewidth=0.5)
        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.set_title('Price Series with Market Regimes', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot regime timeline
        for regime in range(self.n_regimes):
            mask = regimes == regime
            ax2.fill_between(features.index, 0, 1, where=mask, 
                            color=colors[regime], alpha=0.5, label=f'Regime {regime}')
        
        ax2.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_title('Regime Timeline', fontsize=14, fontweight='bold')
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_regime_characteristics(self, features: pd.DataFrame, regimes: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Plot characteristics of each regime.
        
        Args:
            features: DataFrame with features
            regimes: Array of regime labels
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # Plot returns distribution by regime
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = features['returns'][mask]
            axes[0].hist(regime_returns, bins=50, alpha=0.5, 
                        label=f'Regime {regime}', color=colors[regime], density=True)
        
        axes[0].set_xlabel('Returns', fontsize=11)
        axes[0].set_ylabel('Density', fontsize=11)
        axes[0].set_title('Returns Distribution by Regime', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot volatility distribution by regime
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_vol = features['volatility'][mask]
            axes[1].hist(regime_vol, bins=50, alpha=0.5, 
                        label=f'Regime {regime}', color=colors[regime], density=True)
        
        axes[1].set_xlabel('Volatility', fontsize=11)
        axes[1].set_ylabel('Density', fontsize=11)
        axes[1].set_title('Volatility Distribution by Regime', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot scatter: returns vs volatility
        for regime in range(self.n_regimes):
            mask = regimes == regime
            axes[2].scatter(features['volatility'][mask], features['returns'][mask],
                          alpha=0.5, label=f'Regime {regime}', color=colors[regime], s=20)
        
        axes[2].set_xlabel('Volatility', fontsize=11)
        axes[2].set_ylabel('Returns', fontsize=11)
        axes[2].set_title('Returns vs Volatility by Regime', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot regime duration
        regime_changes = np.diff(regimes, prepend=regimes[0])
        regime_starts = np.where(regime_changes != 0)[0]
        durations = {regime: [] for regime in range(self.n_regimes)}
        
        for i in range(len(regime_starts) - 1):
            start = regime_starts[i]
            end = regime_starts[i + 1]
            regime = regimes[start]
            durations[regime].append(end - start)
        
        regime_labels = [f'Regime {i}' for i in range(self.n_regimes)]
        duration_data = [durations[i] if durations[i] else [0] for i in range(self.n_regimes)]
        
        bp = axes[3].boxplot(duration_data, labels=regime_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:self.n_regimes]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        axes[3].set_ylabel('Duration (days)', fontsize=11)
        axes[3].set_title('Regime Duration Distribution', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the regime transition probability matrix.
        
        Returns:
            DataFrame with transition probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        trans_matrix = pd.DataFrame(
            self.model.transmat_,
            columns=[f'To Regime {i}' for i in range(self.n_regimes)],
            index=[f'From Regime {i}' for i in range(self.n_regimes)]
        )
        
        return trans_matrix
    
    def summarize(self, features: pd.DataFrame, regimes: np.ndarray):
        """
        Print a comprehensive summary of the regime detection results.
        
        Args:
            features: DataFrame with features
            regimes: Array of regime labels
        """
        print("=" * 80)
        print("REGIME DETECTION SUMMARY")
        print("=" * 80)
        
        print(f"\nNumber of Regimes: {self.n_regimes}")
        print(f"Total Observations: {len(regimes)}")
        
        print("\nRegime Statistics:")
        print("-" * 80)
        stats = self.get_regime_statistics(features, regimes)
        print(stats.to_string(index=False))
        
        print("\n\nTransition Probability Matrix:")
        print("-" * 80)
        trans_matrix = self.get_transition_matrix()
        print(trans_matrix.round(4))
        
        print("\n" + "=" * 80)