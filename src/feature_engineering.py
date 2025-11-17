"""
Feature engineering utilities for retail options trading risk analytics.
This module includes functions for creating technical indicators,
transforming features, and preparing data for modeling.

Dependencies:  
    - pandas
    - numpy
    - ta
"""
# Standard library imports
import numpy as np
import pandas as pd
import ta

class FeatureEngineer:
    def __init__(self, long, short):
        self.long = long
        self.short = short

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the DataFrame."""
        df = df.copy()
        df[f'SMA_{self.long}'] = ta.trend.sma_indicator(df['Close'], window=self.long)
        df[f'SMA_{self.short}'] = ta.trend.sma_indicator(df['Close'], window=self.short)

        df[f'EMA_{self.long}'] = ta.trend.ema_indicator(df['Close'], window=self.long)
        df[f'EMA_{self.short}'] = ta.trend.ema_indicator(df['Close'], window=self.short)

        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd(df['Close'])
        df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
        df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
        return df

    def normalize_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Normalize specified feature columns using Min-Max scaling."""
        df = df.copy()
        for col in feature_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
        return df

    def create_lagged_features(self, df: pd.DataFrame, feature_cols: list, lags: int) -> pd.DataFrame:
        """Create lagged features for specified columns."""
        df = df.copy()
        for col in feature_cols:
            for lag in range(1, lags + 1):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def prepare_for_modeling(self, df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:
        """Prepare features and target variable for modeling."""
        df = df.dropna().copy()
        X = df[feature_cols]
        y = df[target_col]
        return X, y
    
    def display_feature_summary(self, df: pd.DataFrame):
        """Display summary statistics of the features."""
        print("Feature Summary Statistics:")
        print(df.describe())

        print("\nFeature Correlation Matrix:")
        print(df.corr())

        print("\nMissing Values in Features:")
        print(df.isnull().sum())

    def create_weekly_aggregation_table(
        self, collection_name: str, symbol: str = "data"
    ) -> pd.DataFrame:
        """Create a weekly aggregation table with open, high, low, close, and volume."""
        df = self.read_data(collection_name, symbol)

        # Resample to weekly frequency
        weekly_table = pd.DataFrame()

        for col in df.columns:
            if "Open" in col:
                weekly_table[f"{col}_Weekly"] = df[col].resample("W").first()
            elif "High" in col:
                weekly_table[f"{col}_Weekly"] = df[col].resample("W").max()
            elif "Low" in col:
                weekly_table[f"{col}_Weekly"] = df[col].resample("W").min()
            elif "Close" in col:
                weekly_table[f"{col}_Weekly"] = df[col].resample("W").last()
            elif "Volume" in col:
                weekly_table[f"{col}_Weekly"] = df[col].resample("W").sum()

        return weekly_table

    def read_data(self, collection_name: str, symbol: str = "data") -> pd.DataFrame:
        """Read data from the specified collection and symbol."""
        try:
            collection = self.db[collection_name]
            data = collection.read(symbol)
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error reading data from collection '{collection_name}': {e}")
            return pd.DataFrame()

    def pipeline(self, df: pd.DataFrame, feature_cols: list, target_col: str, lags: int) -> tuple:
        """Complete feature engineering pipeline."""
        df = self.add_technical_indicators(df)
        df = self.normalize_features(df, feature_cols)
        df = self.create_lagged_features(df, feature_cols, lags)
        X, y = self.prepare_for_modeling(df, target_col, feature_cols)
        return X, y

def main():
    # Example usage
    fe = FeatureEngineer(long=50, short=20)
    # sample dataframe
    data = {
        'Close': np.random.rand(100) * 100,
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Volume': np.random.randint(1000, 10000, size=100)
    }
    df = pd.DataFrame(data)
    feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    target_col = 'Close'
    X, y = fe.pipeline(df, feature_cols, target_col, lags=3)
    fe.display_feature_summary(df)

if __name__ == "__main__":
    main()