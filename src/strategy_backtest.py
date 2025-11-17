"""
Strategy backtesting framework for options trading strategies.

This module implements a comprehensive backtesting framework for evaluating 
options trading strategies with risk-adjusted performance metrics.

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - seaborn
"""

# Standard library imports
from typing import Dict, List, Optional
import warnings

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class StrategyBacktest:
    """
    Backtesting framework for options trading strategies.
    
    This class provides functionality to backtest trading strategies, calculate
    performance metrics, and visualize results with risk-adjusted returns.
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        """
        Initialize the backtesting framework.
        
        Args:
            initial_capital: Starting capital for the backtest
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.portfolio_values = None
        self.positions = None
        self.metrics = None
        
    def generate_signals(self, df: pd.DataFrame, strategy: str = 'moving_average',
                        short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
        """
        Generate trading signals based on the specified strategy.
        
        Args:
            df: DataFrame with price data
            strategy: Strategy type ('moving_average', 'momentum', 'mean_reversion')
            short_window: Short-term window for indicators
            long_window: Long-term window for indicators
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=df.index)
        
        # Find Close column
        close_col = [col for col in df.columns if 'Close' in col][0]
        signals['price'] = df[close_col]
        
        if strategy == 'moving_average':
            # Moving Average Crossover Strategy
            signals['short_ma'] = signals['price'].rolling(window=short_window).mean()
            signals['long_ma'] = signals['price'].rolling(window=long_window).mean()
            
            # Generate signals: 1 (buy) when short MA crosses above long MA, -1 (sell) when below
            signals['signal'] = 0
            signals.loc[signals['short_ma'] > signals['long_ma'], 'signal'] = 1
            signals.loc[signals['short_ma'] < signals['long_ma'], 'signal'] = -1
            
        elif strategy == 'momentum':
            # Momentum Strategy
            signals['returns'] = signals['price'].pct_change()
            signals['momentum'] = signals['returns'].rolling(window=long_window).mean()
            
            # Generate signals: buy when momentum is positive, sell when negative
            signals['signal'] = 0
            signals.loc[signals['momentum'] > 0, 'signal'] = 1
            signals.loc[signals['momentum'] < 0, 'signal'] = -1
            
        elif strategy == 'mean_reversion':
            # Mean Reversion Strategy
            signals['sma'] = signals['price'].rolling(window=long_window).mean()
            signals['std'] = signals['price'].rolling(window=long_window).std()
            signals['upper_band'] = signals['sma'] + 2 * signals['std']
            signals['lower_band'] = signals['sma'] - 2 * signals['std']
            
            # Generate signals: sell when price above upper band, buy when below lower band
            signals['signal'] = 0
            signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1
            signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Generate position changes (entries and exits)
        signals['position'] = signals['signal'].diff()
        
        return signals.dropna()
    
    def run_backtest(self, df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the backtest based on generated signals.
        
        Args:
            df: DataFrame with price data
            signals: DataFrame with trading signals
            
        Returns:
            DataFrame with portfolio values and positions
        """
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['price'] = signals['price']
        portfolio['signal'] = signals['signal']
        portfolio['position'] = signals['signal']  # Position: 1 (long), -1 (short), 0 (neutral)
        
        # Calculate holdings
        portfolio['holdings'] = portfolio['position'] * portfolio['price']
        
        # Calculate cash after each trade
        portfolio['cash'] = self.initial_capital
        position_changes = signals['position'] != 0
        
        for i in range(1, len(portfolio)):
            if position_changes.iloc[i]:
                # Calculate trade cost with commission
                trade_value = abs(portfolio['position'].iloc[i] - portfolio['position'].iloc[i-1]) * portfolio['price'].iloc[i]
                trade_cost = trade_value * self.commission
                
                # Update cash
                portfolio.loc[portfolio.index[i], 'cash'] = (
                    portfolio['cash'].iloc[i-1] - 
                    (portfolio['position'].iloc[i] - portfolio['position'].iloc[i-1]) * portfolio['price'].iloc[i] -
                    trade_cost
                )
                
                # Record trade
                self.trades.append({
                    'date': portfolio.index[i],
                    'position': portfolio['position'].iloc[i],
                    'price': portfolio['price'].iloc[i],
                    'cost': trade_cost
                })
            else:
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
        
        # Calculate total portfolio value
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        
        # Calculate returns
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
        
        self.portfolio_values = portfolio
        return portfolio
    
    def calculate_metrics(self, portfolio: pd.DataFrame, 
                         benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            portfolio: DataFrame with portfolio values
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary with performance metrics
        """
        returns = portfolio['returns'].dropna()
        
        # Basic metrics
        total_return = portfolio['cumulative_returns'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Downside deviation (for Sortino ratio)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_deviation if downside_deviation != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = len([t for t in self.trades if t['position'] > 0])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
            'Total Trades': total_trades,
            'Final Value': portfolio['total'].iloc[-1]
        }
        
        # Benchmark comparison
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
            excess_returns = returns - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
            
            metrics['Tracking Error'] = tracking_error
            metrics['Information Ratio'] = information_ratio
            
            # Beta calculation
            covariance = returns.cov(aligned_benchmark)
            benchmark_variance = aligned_benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            metrics['Beta'] = beta
        
        self.metrics = metrics
        return metrics
    
    def plot_performance(self, portfolio: pd.DataFrame, 
                        benchmark: Optional[pd.Series] = None,
                        save_path: Optional[str] = None):
        """
        Plot portfolio performance with multiple visualizations.
        
        Args:
            portfolio: DataFrame with portfolio values
            benchmark: Optional benchmark series for comparison
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(portfolio.index, portfolio['cumulative_returns'] * 100, 
                label='Strategy', linewidth=2, color='blue')
        
        if benchmark is not None:
            benchmark_aligned = benchmark.reindex(portfolio.index).fillna(method='ffill')
            benchmark_cumret = (1 + benchmark_aligned).cumprod() - 1
            ax1.plot(portfolio.index, benchmark_cumret * 100, 
                    label='Benchmark', linewidth=2, color='orange', alpha=0.7)
        
        ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(portfolio.index, portfolio['total'], linewidth=2, color='green')
        ax2.set_title('Portfolio Value', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Value ($)', fontsize=11)
        ax2.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital', alpha=0.5)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        returns = portfolio['returns'].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax3.fill_between(drawdown.index, drawdown * 100, 0, 
                        color='red', alpha=0.3)
        ax3.plot(drawdown.index, drawdown * 100, linewidth=2, color='red')
        ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Returns distribution
        ax4 = fig.add_subplot(gs[2, 0])
        returns_pct = returns * 100
        ax4.hist(returns_pct, bins=50, color='purple', alpha=0.6, edgecolor='black')
        ax4.axvline(x=returns_pct.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {returns_pct.mean():.3f}%')
        ax4.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Daily Returns (%)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Rolling Sharpe ratio
        ax5 = fig.add_subplot(gs[2, 1])
        rolling_returns = returns.rolling(window=63)  # Quarterly
        rolling_sharpe = (rolling_returns.mean() * 252) / (rolling_returns.std() * np.sqrt(252))
        
        ax5.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='teal')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_title('Rolling Sharpe Ratio (63-day)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Date', fontsize=11)
        ax5.set_ylabel('Sharpe Ratio', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Strategy Backtest Performance', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_trades(self, portfolio: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot price chart with buy/sell signals.
        
        Args:
            portfolio: DataFrame with portfolio values and positions
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot price
        ax.plot(portfolio.index, portfolio['price'], linewidth=2, 
               color='black', label='Price', alpha=0.7)
        
        # Plot buy signals
        buy_signals = portfolio[portfolio['position'].diff() > 0]
        ax.scatter(buy_signals.index, buy_signals['price'], 
                  marker='^', color='green', s=100, label='Buy', zorder=5)
        
        # Plot sell signals
        sell_signals = portfolio[portfolio['position'].diff() < 0]
        ax.scatter(sell_signals.index, sell_signals['price'], 
                  marker='v', color='red', s=100, label='Sell', zorder=5)
        
        ax.set_title('Trading Signals', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def compare_strategies(self, df: pd.DataFrame, strategies: List[str],
                          save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple strategies and return performance comparison.
        
        Args:
            df: DataFrame with price data
            strategies: List of strategy names to compare
            save_path: Optional path to save comparison plot
            
        Returns:
            DataFrame with strategy comparison metrics
        """
        results = []
        fig, ax = plt.subplots(figsize=(16, 8))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        
        for i, strategy in enumerate(strategies):
            # Reset trades for each strategy
            self.trades = []
            
            # Generate signals and run backtest
            signals = self.generate_signals(df, strategy=strategy)
            portfolio = self.run_backtest(df, signals)
            metrics = self.calculate_metrics(portfolio)
            
            # Store results
            metrics['Strategy'] = strategy
            results.append(metrics)
            
            # Plot cumulative returns
            ax.plot(portfolio.index, portfolio['cumulative_returns'] * 100,
                   label=strategy.replace('_', ' ').title(), 
                   linewidth=2, color=colors[i % len(colors)])
        
        ax.set_title('Strategy Comparison: Cumulative Returns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('Strategy')
        
        return comparison_df
    
    def summarize(self):
        """
        Print a comprehensive summary of backtest results.
        """
        if self.metrics is None:
            print("No backtest results available. Run backtest first.")
            return
        
        print("=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        
        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.metrics['Final Value']:,.2f}")
        print(f"Total Return: {self.metrics['Total Return']:.2%}")
        
        print("\nRisk-Adjusted Performance:")
        print("-" * 80)
        print(f"Annual Return: {self.metrics['Annual Return']:.2%}")
        print(f"Annual Volatility: {self.metrics['Annual Volatility']:.2%}")
        print(f"Sharpe Ratio: {self.metrics['Sharpe Ratio']:.4f}")
        print(f"Sortino Ratio: {self.metrics['Sortino Ratio']:.4f}")
        print(f"Calmar Ratio: {self.metrics['Calmar Ratio']:.4f}")
        
        print("\nRisk Metrics:")
        print("-" * 80)
        print(f"Maximum Drawdown: {self.metrics['Max Drawdown']:.2%}")
        
        print("\nTrading Statistics:")
        print("-" * 80)
        print(f"Total Trades: {self.metrics['Total Trades']}")
        print(f"Win Rate: {self.metrics['Win Rate']:.2%}")
        
        if 'Information Ratio' in self.metrics:
            print("\nBenchmark Comparison:")
            print("-" * 80)
            print(f"Information Ratio: {self.metrics['Information Ratio']:.4f}")
            print(f"Tracking Error: {self.metrics['Tracking Error']:.2%}")
            print(f"Beta: {self.metrics['Beta']:.4f}")
        
        print("\n" + "=" * 80)
