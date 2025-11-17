"""
Data ingestion module for loading and preprocessing datasets.

This module provides functions to read data from various sources, clean it,
and prepare it for analysis or modeling.

Dependencies:
    - pandas
    - numpy
    - yfinance
    - arcticdb
"""

# Standard library imports
from pathlib import Path
from datetime import datetime

# Third party imports
import pandas as pd
import yfinance as yf
import arcticdb as adb


class DataIngestor:
    def __init__(self, db_uri: str):
        self.db = adb.Arctic(db_uri)

    def fetch_stock_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance."""
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        # Flatten MultiIndex columns if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [
                "_".join(col).strip() if isinstance(col, tuple) else col
                for col in stock_data.columns.values
            ]
        return stock_data

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values and duplicates."""
        df = df.drop_duplicates()
        df = df.ffill().bfill()
        return df

    def store_data(self, collection_name: str, df: pd.DataFrame):
        """Store the cleaned data into ArcticDB."""
        lib = self.db.get_library(collection_name, create_if_missing=True)
        lib.write("data", df)

    def read_data(self, collection_name: str, symbol: str = "data") -> pd.DataFrame:
        """Read data from ArcticDB collection."""
        lib = self.db.get_library(collection_name, create_if_missing=False)
        df = lib.read(symbol).data
        return df

    def display_data(self, collection_name: str, symbol: str = "data", head: int = 10):
        """Display data from ArcticDB collection with summary statistics."""
        try:
            df = self.read_data(collection_name, symbol)

            print(f"\n{'=' * 60}")
            print(f"Data from collection: '{collection_name}' (symbol: '{symbol}')")
            print(f"{'=' * 60}\n")

            print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"Date range: {df.index.min()} to {df.index.max()}\n")

            print(f"First {head} rows:")
            print(df.head(head))

            print(f"\nLast {head} rows:")
            print(df.tail(head))

            print("\nSummary Statistics:")
            print(df.describe())

            print("\nColumn Names:")
            print(df.columns.tolist())

            print("\nData Types:")
            print(df.dtypes)

            print(f"\n{'=' * 60}\n")

        except Exception as e:
            print(f"Error reading data from collection '{collection_name}': {e}")

    def list_collections(self):
        """List all available collections (libraries) in the database."""
        libraries = self.db.list_libraries()
        print(f"\nAvailable collections: {libraries}")
        return libraries

    def create_summary_table(
        self, collection_name: str, symbol: str = "data"
    ) -> pd.DataFrame:
        """Create a summary statistics table from the data."""
        df = self.read_data(collection_name, symbol)
        summary_table = df.describe()
        return summary_table

    def create_monthly_aggregation_table(
        self, collection_name: str, symbol: str = "data"
    ) -> pd.DataFrame:
        """Create a monthly aggregation table with open, high, low, close, and volume."""
        df = self.read_data(collection_name, symbol)

        # Resample to monthly frequency
        monthly_table = pd.DataFrame()

        for col in df.columns:
            if "Open" in col:
                monthly_table[f"{col}_Monthly"] = df[col].resample("ME").first()
            elif "High" in col:
                monthly_table[f"{col}_Monthly"] = df[col].resample("ME").max()
            elif "Low" in col:
                monthly_table[f"{col}_Monthly"] = df[col].resample("ME").min()
            elif "Close" in col:
                monthly_table[f"{col}_Monthly"] = df[col].resample("ME").last()
            elif "Volume" in col:
                monthly_table[f"{col}_Monthly"] = df[col].resample("ME").sum()

        return monthly_table

    def create_returns_table(
        self, collection_name: str, symbol: str = "data"
    ) -> pd.DataFrame:
        """Create a table with daily, weekly, and monthly returns."""
        df = self.read_data(collection_name, symbol)

        returns_table = pd.DataFrame()

        # Find close price columns
        close_cols = [col for col in df.columns if "Close" in col]

        for col in close_cols:
            ticker = col.replace("Close_", "")
            returns_table[f"{ticker}_Daily_Return"] = df[col].pct_change()
            returns_table[f"{ticker}_Weekly_Return"] = df[col].pct_change(5)
            returns_table[f"{ticker}_Monthly_Return"] = df[col].pct_change(21)

        return returns_table

    def save_table(
        self, table: pd.DataFrame, filename: str, output_dir: str = "results/tables"
    ):
        """
        Save a table to Excel format.

        Args:
            table: DataFrame to save
            filename: Base filename (without extension)
            output_dir: Directory to save tables to (default: results/tables)
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename}_{timestamp}"

        # Save as Excel
        excel_path = output_path / f"{base_filename}.xlsx"
        table.to_excel(excel_path, engine="openpyxl")
        print(f"Table saved to: {excel_path}")

        return str(excel_path)

    def create_and_save_all_tables(
        self,
        collection_name: str,
        symbol: str = "data",
        output_dir: str = "results/tables",
    ):
        """
        Create and save all available tables for a given collection.

        Args:
            collection_name: Name of the collection to process
            symbol: Symbol name in the collection (default: "data")
            output_dir: Directory to save tables to (default: results/tables)
        """
        print(f"\nCreating and saving tables for collection '{collection_name}'...\n")

        # Create and save summary statistics table
        print("1. Creating summary statistics table...")
        summary_table = self.create_summary_table(collection_name, symbol)
        self.save_table(summary_table, f"{collection_name}_summary_stats", output_dir)

        # Create and save monthly aggregation table
        print("\n2. Creating monthly aggregation table...")
        monthly_table = self.create_monthly_aggregation_table(collection_name, symbol)
        self.save_table(monthly_table, f"{collection_name}_monthly_agg", output_dir)

        # Create and save returns table
        print("\n3. Creating returns table...")
        returns_table = self.create_returns_table(collection_name, symbol)
        self.save_table(returns_table, f"{collection_name}_returns", output_dir)

        # Create and save raw data table
        print("\n4. Saving raw data table...")
        raw_data = self.read_data(collection_name, symbol)
        self.save_table(raw_data, f"{collection_name}_raw_data", output_dir)

        print(f"\n✓ All tables saved successfully to '{output_dir}' folder!\n")

    def ingest(self, ticker: str, start_date: str, end_date: str, collection_name: str):
        """Complete ingestion process from fetching to storing data."""
        raw_data = self.fetch_stock_data(ticker, start_date, end_date)
        cleaned_data = self.clean_data(raw_data)
        self.store_data(collection_name, cleaned_data)

    def pipeline_summary(
        self, ticker: str, start_date: str, end_date: str, collection_name: str
    ):
        """Print a summary of the ingestion pipeline."""
        print(
            f"Ingesting data for {ticker} from {start_date} to {end_date} into collection '{collection_name}'"
        )
        raw_data = self.fetch_stock_data(ticker, start_date, end_date)
        print(f"Fetched {len(raw_data)} rows of raw data.")
        cleaned_data = self.clean_data(raw_data)
        print(
            f"Cleaned data has {len(cleaned_data)} rows after removing duplicates and handling missing values."
        )
        self.store_data(collection_name, cleaned_data)
        print(f"Data stored in collection '{collection_name}' successfully.")


def main():
    ingestor = DataIngestor(db_uri="lmdb://data/arcticdb")

    # Ingest data
    ingestor.pipeline_summary(
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2021-01-01",
        collection_name="apple_stock_data",
    )

    # List all collections
    ingestor.list_collections()

    # Display the stored data
    ingestor.display_data(collection_name="apple_stock_data", head=5)

    # Create and save all tables
    ingestor.create_and_save_all_tables(
        collection_name="apple_stock_data", output_dir="results/tables"
    )


if __name__ == "__main__":
    main()
