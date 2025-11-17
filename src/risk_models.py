""" 
Modules for various risk models used in retail options trading analytics.

Such as: Volatility Modeling, Value at Risk (VaR), Expected Shortfall (ES),
GARCH models.

Dependencies:
    - pandas
    - numpy
    - scipy
    - statsmodels
    - arch
    - matplotlib
    - seaborn
"""
# Standard library imports
from typing import Optional

# Third party imports
import pandas as pd
import numpy as np
from scipy import stats
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns

class RiskModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) using historical simulation."""
        returns = self.df['Close'].pct_change().dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def calculate_es(self, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (ES) using historical simulation."""
        returns = self.df['Close'].pct_change().dropna()
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        es = returns[returns <= var_threshold].mean()
        return es

    def fit_garch_model(self, p: int = 1, q: int = 1) -> arch_model:
        """Fit a GARCH model to the returns data."""
        returns = self.df['Close'].pct_change().dropna() * 100  # Convert to percentage
        model = arch_model(returns, vol='Garch', p=p, q=q)
        garch_fit = model.fit(disp='off')
        return garch_fit

    def plot_volatility(self, garch_fit: arch_model):
        """Plot the conditional volatility from the GARCH model."""
        plt.figure(figsize=(10, 6))
        plt.plot(garch_fit.conditional_volatility, label='Conditional Volatility')
        plt.title('GARCH Model Conditional Volatility')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.show()

    def plot_return_distribution(self):
        """Plot the distribution of returns with a fitted normal distribution."""
        returns = self.df['Close'].pct_change().dropna()
        sns.histplot(returns, bins=50, kde=True, stat="density")
        
        mu, std = stats.norm.fit(returns)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        
        plt.title('Return Distribution with Fitted Normal Distribution')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.show()

    def summary(self, garch_fit: Optional[arch_model] = None):
        """Print a summary of risk metrics and GARCH model fit if provided."""
        var_95 = self.calculate_var(0.95)
        es_95 = self.calculate_es(0.95)
        
        print(f"Value at Risk (95%): {var_95:.4f}")
        print(f"Expected Shortfall (95%): {es_95:.4f}")
        
        if garch_fit:
            print("\nGARCH Model Summary:")
            print(garch_fit.summary())
            self.plot_volatility(garch_fit)
            self.plot_return_distribution()

    def run_full_analysis(self, garch_p: int = 1, garch_q: int = 1):
        """Run full risk analysis including VaR, ES, GARCH fitting, and plotting."""
        garch_fit = self.fit_garch_model(p=garch_p, q=garch_q)
        self.summary(garch_fit=garch_fit)
