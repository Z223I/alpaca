import os
import glob
import sys
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, time, date
import pytz
import pandas as pd
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi   # pip3 install alpaca-trade-api -U

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.utils.read_csv import read_csv  # noqa: E402
from atoms.api.init_alpaca_client import init_alpaca_client  # noqa: E402
from atoms.display.plot_candle_chart import plot_candle_chart  # noqa: E402
from atoms.utils.extract_symbol_data import extract_symbol_data  # noqa: E402
from atoms.utils.calculate_orb_levels import calculate_orb_levels  # noqa: E402
from atoms.utils.calculate_ema import calculate_ema  # noqa: E402
from atoms.utils.calculate_vwap import calculate_vwap_typical  # noqa: E402
from atoms.utils.calculate_vector_angle import (  # noqa: E402
    calculate_vector_angle)


class ORB:
    """
    Open Range Breakout (ORB) class for analyzing trading opportunities
    based on breakout patterns from opening range data.
    """

    def __init__(self):
        """Initialize the ORB class."""

        # Get api key and secret from environment variables

        # Set portfolio risk from environment variable or use default
        self.PORTFOLIO_RISK = float(os.getenv('PORTFOLIO_RISK', '0.10'))

        # Initialize Alpaca API client using atom
        self.api = init_alpaca_client()

        # Other initializations
        self.data_directory = 'data'
        self.csv_data: Optional[List[Dict[str, Any]]] = None
        self.current_file: Optional[str] = None
        self.csv_date: Optional[date] = None
        self.pca_data: Optional[pd.DataFrame] = None

    def _get_most_recent_csv(self) -> Optional[str]:
        """
        Get the most recent CSV file from the data directory.

        Returns:
            Path to the most recent CSV file, or None if no CSV files found
        """
        try:
            csv_pattern = os.path.join(self.data_directory, '*.csv')
            csv_files = glob.glob(csv_pattern)

            if not csv_files:
                return None

            # Sort by modification time, most recent first
            csv_files.sort(key=os.path.getmtime, reverse=True)
            return csv_files[0]

        except Exception as e:
            print(f"Error finding CSV files: {e}")
            return None

    def _prompt_user_for_file(self, filename: str) -> bool:
        """
        Prompt user to confirm if they want to use the suggested file.

        Args:
            filename: Name of the file to confirm

        Returns:
            True if user confirms, False otherwise
        """
        try:
            response = input(f"Use file '{filename}'? (Y/n): ").strip().lower()
            return response in ['y', 'yes', '']
        except (EOFError, KeyboardInterrupt):
            return False

    def _load_and_process_csv_data(self) -> bool:
        """
        Load and process CSV data from the most recent file.

        This method:
        1. Finds the most current CSV file in the data directory
        2. Prompts user for confirmation to use that file
        3. Reads the file if confirmed, exits if not

        Returns:
            True if successful, False otherwise
        """
        print("ORB - Open Range Breakout Analysis")
        print("=" * 40)

        # Find the most recent CSV file
        most_recent_file = self._get_most_recent_csv()

        if not most_recent_file:
            print("No CSV files found in the data directory.")
            return False

        print(f"Most recent CSV file: {most_recent_file}")

        # Prompt user for confirmation
        if not self._prompt_user_for_file(most_recent_file):
            print("Operation cancelled by user.")
            return False

        # Read the CSV file
        try:
            print(f"Reading file: {most_recent_file}")
            self.csv_data = read_csv(most_recent_file)
            self.current_file = most_recent_file

            # Extract date from filename (YYYYMMDD.csv format)
            filename = os.path.basename(most_recent_file)
            date_str = filename.replace('.csv', '')
            try:
                self.csv_date = datetime.strptime(date_str, '%Y%m%d').date()
                print(f"Extracted date from filename: {self.csv_date}")
            except ValueError:
                print(f"Warning: Could not parse date from filename "
                      f"'{filename}'. Expected YYYYMMDD.csv format.")
                self.csv_date = None

            if not self.csv_data:
                print("Warning: CSV file is empty or contains no data.")
                return False

            print(f"Successfully loaded {len(self.csv_data)} rows of data.")

            # Display basic file information
            if self.csv_data:
                columns = list(self.csv_data[0].keys())
                print(f"Columns: {', '.join(columns)}")
                print(f"Sample data from first row: {self.csv_data[0]}")

            return True

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

    def _get_orb_market_data(self) -> bool:
        """
        Get market data for symbols from CSV for normal market hours
        (9:30 AM to 4:00 PM ET).

        Returns:
            True if successful, False otherwise
        """
        if not self.csv_data:
            print("No CSV data loaded.")
            return False

        try:
            # Extract symbols from CSV data
            symbols = []
            for row in self.csv_data:
                if 'symbol' in row and row['symbol']:
                    symbols.append(row['symbol'])
                elif 'Symbol' in row and row['Symbol']:
                    symbols.append(row['Symbol'])

            if not symbols:
                print("No symbols found in CSV data.")
                return False

            print(f"Getting market data for {len(symbols)} symbols: "
                  f"{symbols[:5]}{'...' if len(symbols) > 5 else ''}")

            # Use date from CSV filename with normal market hours
            # (9:30 AM to 4:00 PM ET)
            et_tz = pytz.timezone('America/New_York')
            target_date = (self.csv_date if self.csv_date
                           else datetime.now(et_tz).date())

            start_time = datetime.combine(target_date, time(9, 30),
                                          tzinfo=et_tz)
            end_time = datetime.combine(target_date, time(16, 0),
                                        tzinfo=et_tz)

            print(f"Fetching 1-minute data from {start_time} to {end_time}")

            # Get stock data using 1-minute timeframe with legacy API
            market_data = {}
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(
                        symbol,
                        tradeapi.TimeFrame.Minute,  # type: ignore
                        start=start_time.isoformat(),
                        end=end_time.isoformat()
                    )
                    market_data[symbol] = bars
                except Exception as e:
                    print(f"Error getting data for {symbol}: {e}")
                    continue

            if market_data:
                print("Successfully retrieved market data for ORB analysis.")
                # Store the market data for further analysis
                self.market_data = market_data

                # Save market data to stock_data directory
                self._save_market_data()

                return True
            else:
                print("No market data retrieved.")
                return False

        except Exception as e:
            print(f"Error getting market data: {e}")
            return False

    def _save_market_data(self) -> bool:
        """
        Save market data to JSON file in stock_data directory using the
        same CSV filename.

        Returns:
            True if successful, False otherwise
        """
        if not self.market_data or not self.current_file:
            print("No market data or current file to save.")
            return False

        try:
            # Create stock_data directory if it doesn't exist
            stock_data_dir = 'stock_data'
            os.makedirs(stock_data_dir, exist_ok=True)

            # Get filename without extension and create JSON filename
            filename = os.path.basename(self.current_file)
            json_filename = filename.replace('.csv', '.json')
            json_filepath = os.path.join(stock_data_dir, json_filename)

            # Convert market data to serializable format
            serializable_data = {}
            for symbol, bars in self.market_data.items():
                serializable_data[symbol] = []
                for bar in bars:
                    bar_data = {
                        'timestamp': bar.t.isoformat(),
                        'open': float(bar.o),
                        'high': float(bar.h),
                        'low': float(bar.l),
                        'close': float(bar.c),
                        'volume': int(bar.v)
                    }
                    serializable_data[symbol].append(bar_data)

            # Save to JSON file
            with open(json_filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)

            print(f"Market data saved to: {json_filepath}")
            return True

        except Exception as e:
            print(f"Error saving market data: {e}")
            return False

    def _pca_data_prep(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Prepare data for PCA analysis by filtering to 9:30-10:15 ET and
        collecting metrics.

        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low',
                'close', 'volume']
            symbol: Stock symbol name

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract symbol data using the dedicated atom
            symbol_data = extract_symbol_data(df, symbol)

            if symbol_data is None:
                print(f"No data found for symbol: {symbol}")
                return False

            # Filter the data from 9:30 to 10:15 ET
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])

            # Create time masks for 9:30 to 10:15 ET
            et_tz = pytz.timezone('America/New_York')
            start_time = time(9, 30)
            end_time = time(10, 15)

            # Convert timestamps to ET timezone and filter
            symbol_data_et = symbol_data.copy()
            symbol_data_et['timestamp_et'] = symbol_data_et[
                'timestamp'].dt.tz_convert(et_tz)
            symbol_data_et['time_only'] = symbol_data_et[
                'timestamp_et'].dt.time

            filtered_data = symbol_data_et[
                (symbol_data_et['time_only'] >= start_time) &
                (symbol_data_et['time_only'] <= end_time)
            ].copy()

            # Verify there are 45 lines of data and return if not
            if len(filtered_data) != 45:
                print(f"Expected 45 lines of data for {symbol}, "
                      f"got {len(filtered_data)}")
                return False

            # Calculate ORB levels
            orb_high, orb_low = calculate_orb_levels(filtered_data)

            # Calculate EMA (9-period) for close prices
            ema_success, ema_values = calculate_ema(
                filtered_data, price_column='close', period=9)

            # Calculate VWAP using typical price (HLC/3)
            vwap_success, vwap_values = calculate_vwap_typical(filtered_data)

            # Calculate vector angle using the first 15 candlesticks
            vector_angle = calculate_vector_angle(
                filtered_data, price_column='close', num_candles=15)

            # Create a dataframe with all collected data
            pca_row_data = []
            for idx, row in filtered_data.iterrows():
                pca_row = {
                    'symbol': symbol,
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'orb_high': orb_high,
                    'orb_low': orb_low,
                    'vector_angle': vector_angle  # Repeat for all 45 lines
                }

                # Add EMA values if available
                if (ema_success and
                        idx - filtered_data.index[0] < len(ema_values)):
                    pca_row['ema_9'] = ema_values[idx - filtered_data.index[0]]
                else:
                    pca_row['ema_9'] = None

                # Add VWAP values if available
                if (vwap_success and
                        idx - filtered_data.index[0] < len(vwap_values)):
                    pca_row['vwap'] = vwap_values[idx - filtered_data.index[0]]
                else:
                    pca_row['vwap'] = None

                pca_row_data.append(pca_row)

            # Create or append to the class variable dataframe
            new_pca_df = pd.DataFrame(pca_row_data)

            if self.pca_data is None:
                self.pca_data = new_pca_df
            else:
                self.pca_data = pd.concat(
                    [self.pca_data, new_pca_df], ignore_index=True)

            print(f"PCA data prepared for {symbol}: "
                  f"{len(new_pca_df)} rows added")
            return True

        except Exception as e:
            print(f"Error preparing PCA data for {symbol}: {e}")
            return False

    def _load_market_dataframe(self) -> bool:
        """
        Load market data from saved JSON file and convert to pandas DataFrame.

        Returns:
            True if successful, False otherwise
        """
        if not self.current_file:
            print("No current file to process.")
            return False

        try:
            # Construct JSON filepath
            stock_data_dir = 'stock_data'
            filename = os.path.basename(self.current_file)
            json_filename = filename.replace('.csv', '.json')
            json_filepath = os.path.join(stock_data_dir, json_filename)

            # Check if JSON file exists
            if not os.path.exists(json_filepath):
                print(f"JSON file not found: {json_filepath}")
                return False

            # Read JSON data
            with open(json_filepath, 'r') as f:
                market_data = json.load(f)

            # Convert to DataFrame
            all_data = []
            for symbol, bars in market_data.items():
                for bar in bars:
                    bar_data = bar.copy()
                    bar_data['symbol'] = symbol
                    all_data.append(bar_data)

            if not all_data:
                print("No market data found in JSON file.")
                return False

            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            print(f"\nMarket Data DataFrame ({len(df)} rows):")
            print("=" * 50)
            print(df.head())

            # Store DataFrame for further analysis
            self.market_df = df

            return True

        except Exception as e:
            print(f"Error processing market data: {e}")
            return False

    def _generate_candle_charts(self) -> bool:
        """
        Generate candlestick charts with volume for each stock symbol.

        Returns:
            True if successful, False otherwise
        """
        if not hasattr(self, 'market_df') or self.market_df is None:
            print("No market DataFrame available for charting.")
            return False

        try:
            # Get unique symbols and sort them
            symbols = sorted(self.market_df['symbol'].unique())

            print(f"\nGenerating candlestick charts for "
                  f"{len(symbols)} symbols...")

            # Create plots directory if it doesn't exist
            plots_dir = 'plots'
            os.makedirs(plots_dir, exist_ok=True)

            # Generate chart for each symbol
            success_count = 0
            for symbol in symbols:
                if plot_candle_chart(self.market_df, symbol, plots_dir):
                    success_count += 1

            print(f"Successfully generated {success_count}/{len(symbols)} "
                  f"charts in '{plots_dir}' directory.")

            return success_count > 0

        except Exception as e:
            print(f"Error generating charts: {e}")
            return False

    def Exec(self) -> bool:
        """
        Execute the ORB analysis process.

        Returns:
            True if successful, False otherwise
        """
        success = self._load_and_process_csv_data()

        if not success:
            print("Failed to load and process CSV data.")
            return False

        # Get market data for ORB analysis
        market_data_success = self._get_orb_market_data()

        if not market_data_success:
            print("Failed to get market data for ORB analysis.")
            return False

        # Process market data from saved JSON file
        process_success = self._load_market_dataframe()

        if not process_success:
            print("Failed to process market data.")
            return False

        # Generate candlestick charts for all symbols
        chart_success = self._generate_candle_charts()

        if not chart_success:
            print("Failed to generate charts.")
            return False

        return True


def main():
    """
    Main function to setup and run the ORB class.
    Handles keyboard interrupts gracefully.
    """
    try:
        # Create and run ORB analysis
        orb = ORB()
        success = orb.Exec()

        if success:
            print("\nORB analysis completed successfully.")
        else:
            print("\nORB analysis failed or was cancelled.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
