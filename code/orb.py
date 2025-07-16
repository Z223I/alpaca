import os
import glob
import sys
import json
import argparse
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

    def __init__(self, plot_super_alerts: bool = True):
        """Initialize the ORB class.
        
        Args:
            plot_super_alerts: If True, plot super alerts. If False, plot regular alerts.
        """

        # Get api key and secret from environment variables

        # Set portfolio risk from environment variable or use default
        self.PORTFOLIO_RISK = float(os.getenv('PORTFOLIO_RISK', '0.10'))
        
        # Set debugging flag from environment variable or use default
        self.isDebugging = os.getenv('ORB_DEBUG', 'false').lower() in ['true', '1', 'yes']

        # Initialize Alpaca API client using atom
        self.api = init_alpaca_client()

        # Other initializations
        self.data_directory = 'data'
        self.csv_data: Optional[List[Dict[str, Any]]] = None
        self.current_file: Optional[str] = None
        self.csv_date: Optional[date] = None
        self.pca_data: Optional[pd.DataFrame] = None
        self.plot_super_alerts = plot_super_alerts

    def _is_valid_date_csv(self, filename: str) -> bool:
        """
        Check if filename matches YYYYMMDD.csv format.
        
        Args:
            filename: The filename to validate
            
        Returns:
            True if filename matches YYYYMMDD.csv format, False otherwise
        """
        try:
            # Remove .csv extension
            if not filename.lower().endswith('.csv'):
                return False
                
            date_str = filename[:-4]  # Remove .csv
            
            # Check if it's exactly 8 digits
            if len(date_str) != 8 or not date_str.isdigit():
                return False
                
            # Try to parse as date to ensure it's a valid date
            datetime.strptime(date_str, '%Y%m%d')
            return True
            
        except ValueError:
            return False

    def _load_alerts_for_symbol(self, symbol: str, target_date: date) -> List[Dict[str, Any]]:
        """
        Load alerts for a specific symbol and date. Loads super alerts or regular alerts based on plot_super_alerts setting.
        
        Args:
            symbol: Stock symbol to load alerts for
            target_date: Date to load alerts for
            
        Returns:
            List of alert dictionaries containing timestamp, type, and alert data
        """
        if self.plot_super_alerts:
            return self._load_super_alerts_for_symbol(symbol, target_date)
        else:
            return self._load_regular_alerts_for_symbol(symbol, target_date)

    def _load_super_alerts_for_symbol(self, symbol: str, target_date: date) -> List[Dict[str, Any]]:
        """
        Load super alerts for a specific symbol and date from historical_data/super_alerts.
        
        Args:
            symbol: Stock symbol to load super alerts for
            target_date: Date to load super alerts for
            
        Returns:
            List of super alert dictionaries containing timestamp, type, and alert data
        """
        alerts = []
        
        try:
            # Format date as YYYY-MM-DD for directory structure
            date_str = target_date.strftime('%Y-%m-%d')
            alerts_base_dir = os.path.join('historical_data', date_str, 'super_alerts')
            
            if not os.path.exists(alerts_base_dir):
                return alerts
            
            # Check both bullish and bearish super alert directories
            for alert_type in ['bullish', 'bearish']:
                alert_dir = os.path.join(alerts_base_dir, alert_type)
                
                if not os.path.exists(alert_dir):
                    continue
                    
                # Look for super alert files matching the symbol
                alert_pattern = f"super_alert_{symbol}_*.json"
                alert_files = glob.glob(os.path.join(alert_dir, alert_pattern))
                
                for alert_file in alert_files:
                    try:
                        with open(alert_file, 'r') as f:
                            alert_data = json.load(f)
                            
                        # Add alert type and parse timestamp
                        alert_data['alert_type'] = alert_type
                        
                        # Parse timestamp to datetime object and handle timezone
                        if 'timestamp' in alert_data:
                            timestamp_str = alert_data['timestamp']
                            try:
                                # Handle timezone format: convert -0400 to -04:00 for Python compatibility
                                if timestamp_str.endswith(('-0400', '-0500')):
                                    # Insert colon in timezone offset for proper ISO format
                                    timestamp_str = timestamp_str[:-2] + ':' + timestamp_str[-2:]
                                
                                # Parse the timestamp - super alerts are in ET timezone with offset
                                alert_dt = datetime.fromisoformat(timestamp_str)
                                
                            except ValueError as ve1:
                                # Fallback: try parsing without timezone, then localize to ET
                                try:
                                    if '+' in timestamp_str or timestamp_str.count('-') > 2:
                                        # Remove timezone if present for fallback parsing
                                        timestamp_base = timestamp_str.split('+')[0].split('T')[0] + 'T' + timestamp_str.split('T')[1].split('-')[0].split('+')[0]
                                    else:
                                        timestamp_base = timestamp_str
                                    
                                    alert_dt = datetime.fromisoformat(timestamp_base)
                                    # If timezone-naive, assume it's in ET timezone
                                    if alert_dt.tzinfo is None:
                                        et_tz = pytz.timezone('America/New_York')
                                        alert_dt = et_tz.localize(alert_dt)
                                except (ValueError, OSError) as ve2:
                                    # More specific error reporting
                                    continue
                            except OSError as oe:
                                # Handle "date value out of range" errors
                                continue
                            
                            alert_data['timestamp_dt'] = alert_dt
                        
                        alerts.append(alert_data)
                            
                    except Exception as e:
                        print(f"Warning: Error loading super alert file {alert_file}: {e}")
                        continue
            
            # Sort super alerts by timestamp 
            def safe_sort_key(alert):
                timestamp_dt = alert.get('timestamp_dt')
                if timestamp_dt is None:
                    # Return a timezone-aware max datetime for alerts without timestamps
                    return datetime.max.replace(tzinfo=pytz.UTC)
                elif timestamp_dt.tzinfo is None:
                    # If timestamp is timezone-naive, make it timezone-aware (assume ET)
                    et_tz = pytz.timezone('America/New_York')
                    return et_tz.localize(timestamp_dt)
                else:
                    # Already timezone-aware
                    return timestamp_dt
            
            alerts.sort(key=safe_sort_key)
            
            return alerts
            
        except Exception as e:
            print(f"Error loading super alerts for {symbol} on {target_date}: {e}")
            return alerts

    def _load_regular_alerts_for_symbol(self, symbol: str, target_date: date) -> List[Dict[str, Any]]:
        """
        Load regular alerts for a specific symbol and date from historical_data/alerts.
        
        Args:
            symbol: Stock symbol to load regular alerts for
            target_date: Date to load regular alerts for
            
        Returns:
            List of regular alert dictionaries containing timestamp, type, and alert data
        """
        alerts = []
        
        try:
            # Format date as YYYY-MM-DD for directory structure
            date_str = target_date.strftime('%Y-%m-%d')
            
            # Check bullish alerts directory
            bullish_dir = os.path.join('historical_data', date_str, 'alerts', 'bullish')
            if os.path.exists(bullish_dir):
                # Look for alert files matching the symbol
                alert_pattern = os.path.join(bullish_dir, f"alert_{symbol}_*.json")
                alert_files = glob.glob(alert_pattern)
                
                for alert_file in alert_files:
                    try:
                        with open(alert_file, 'r') as f:
                            alert_data = json.load(f)
                            
                        # Add alert type for display
                        alert_data['alert_type'] = 'bullish_alert'
                        
                        # Parse timestamp for proper sorting
                        timestamp_str = alert_data.get('timestamp', '')
                        if timestamp_str:
                            try:
                                # Handle timezone format: convert -0400 to -04:00 for Python compatibility
                                if timestamp_str.endswith(('-0400', '-0500')):
                                    # Insert colon in timezone offset for proper ISO format
                                    timestamp_str = timestamp_str[:-2] + ':' + timestamp_str[-2:]
                                
                                # Parse the timestamp - regular alerts are in ET timezone with offset
                                alert_dt = datetime.fromisoformat(timestamp_str)
                                
                            except ValueError:
                                # Fallback: try parsing without timezone, then localize to ET
                                try:
                                    if '+' in timestamp_str or timestamp_str.count('-') > 2:
                                        # Remove timezone if present for fallback parsing
                                        timestamp_base = timestamp_str.split('+')[0].split('T')[0] + 'T' + timestamp_str.split('T')[1].split('-')[0].split('+')[0]
                                    else:
                                        timestamp_base = timestamp_str
                                    
                                    alert_dt = datetime.fromisoformat(timestamp_base)
                                    # If timezone-naive, assume it's in ET timezone
                                    if alert_dt.tzinfo is None:
                                        et_tz = pytz.timezone('America/New_York')
                                        alert_dt = et_tz.localize(alert_dt)
                                except (ValueError, OSError):
                                    # More specific error reporting
                                    continue
                            except OSError:
                                # Handle "date value out of range" errors
                                continue
                            
                            alert_data['timestamp_dt'] = alert_dt
                        
                        alerts.append(alert_data)
                            
                    except Exception as e:
                        print(f"Warning: Error loading regular alert file {alert_file}: {e}")
                        continue
            
            # Check bearish alerts directory
            bearish_dir = os.path.join('historical_data', date_str, 'alerts', 'bearish')
            if os.path.exists(bearish_dir):
                # Look for alert files matching the symbol
                alert_pattern = os.path.join(bearish_dir, f"alert_{symbol}_*.json")
                alert_files = glob.glob(alert_pattern)
                
                for alert_file in alert_files:
                    try:
                        with open(alert_file, 'r') as f:
                            alert_data = json.load(f)
                            
                        # Add alert type for display
                        alert_data['alert_type'] = 'bearish_alert'
                        
                        # Parse timestamp for proper sorting
                        timestamp_str = alert_data.get('timestamp', '')
                        if timestamp_str:
                            try:
                                # Handle timezone format: convert -0400 to -04:00 for Python compatibility
                                if timestamp_str.endswith(('-0400', '-0500')):
                                    # Insert colon in timezone offset for proper ISO format
                                    timestamp_str = timestamp_str[:-2] + ':' + timestamp_str[-2:]
                                
                                # Parse the timestamp - regular alerts are in ET timezone with offset
                                alert_dt = datetime.fromisoformat(timestamp_str)
                                
                            except ValueError:
                                # Fallback: try parsing without timezone, then localize to ET
                                try:
                                    if '+' in timestamp_str or timestamp_str.count('-') > 2:
                                        # Remove timezone if present for fallback parsing
                                        timestamp_base = timestamp_str.split('+')[0].split('T')[0] + 'T' + timestamp_str.split('T')[1].split('-')[0].split('+')[0]
                                    else:
                                        timestamp_base = timestamp_str
                                    
                                    alert_dt = datetime.fromisoformat(timestamp_base)
                                    # If timezone-naive, assume it's in ET timezone
                                    if alert_dt.tzinfo is None:
                                        et_tz = pytz.timezone('America/New_York')
                                        alert_dt = et_tz.localize(alert_dt)
                                except (ValueError, OSError):
                                    # More specific error reporting
                                    continue
                            except OSError:
                                # Handle "date value out of range" errors
                                continue
                            
                            alert_data['timestamp_dt'] = alert_dt
                        
                        alerts.append(alert_data)
                            
                    except Exception as e:
                        print(f"Warning: Error loading regular alert file {alert_file}: {e}")
                        continue
            
            # Sort regular alerts by timestamp 
            def safe_sort_key(alert):
                timestamp_dt = alert.get('timestamp_dt')
                if timestamp_dt is None:
                    # Return a timezone-aware max datetime for alerts without timestamps
                    return datetime.max.replace(tzinfo=pytz.UTC)
                elif timestamp_dt.tzinfo is None:
                    # If timestamp is timezone-naive, make it timezone-aware (assume ET)
                    et_tz = pytz.timezone('America/New_York')
                    return et_tz.localize(timestamp_dt)
                else:
                    # Already timezone-aware
                    return timestamp_dt
            
            alerts.sort(key=safe_sort_key)
            
            return alerts
            
        except Exception as e:
            print(f"Error loading regular alerts for {symbol} on {target_date}: {e}")
            return alerts

    def _select_csv_file(self) -> Optional[str]:
        """
        List all CSV files in the data directory and allow user to select one.
        The most recent file is the default selection.

        Returns:
            Path to selected CSV file, or None if cancelled
        """
        try:
            csv_pattern = os.path.join(self.data_directory, '*.csv')
            all_csv_files = glob.glob(csv_pattern)

            # Filter to only include files matching YYYYMMDD.csv format
            csv_files = []
            for filepath in all_csv_files:
                filename = os.path.basename(filepath)
                if self._is_valid_date_csv(filename):
                    csv_files.append(filepath)

            if not csv_files:
                print("No CSV files with YYYYMMDD.csv format found in the data directory.")
                if all_csv_files:
                    print(f"Found {len(all_csv_files)} CSV files, but none match YYYYMMDD.csv format.")
                return None

            # Sort by modification time, most recent first
            csv_files.sort(key=os.path.getmtime, reverse=True)
            
            print("\nAvailable CSV files (YYYYMMDD.csv format):")
            print("=" * 50)
            
            for i, filepath in enumerate(csv_files):
                filename = os.path.basename(filepath)
                mod_time = os.path.getmtime(filepath)
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                default_marker = " (default)" if i == 0 else ""
                print(f"{i + 1:2d}. {filename} - {mod_date}{default_marker}")
            
            print("=" * 50)
            
            while True:
                try:
                    response = input(f"Select file (1-{len(csv_files)}) or press Enter for default: ").strip()
                    
                    if response == "":
                        # Use default (most recent)
                        selected_file = csv_files[0]
                        print(f"Using default: {os.path.basename(selected_file)}")
                        return selected_file
                    
                    selection = int(response)
                    if 1 <= selection <= len(csv_files):
                        selected_file = csv_files[selection - 1]
                        print(f"Selected: {os.path.basename(selected_file)}")
                        return selected_file
                    else:
                        print(f"Please enter a number between 1 and {len(csv_files)}")
                        
                except ValueError:
                    print("Please enter a valid number or press Enter for default")
                except (EOFError, KeyboardInterrupt):
                    print("\nOperation cancelled by user.")
                    return None
                    
        except Exception as e:
            print(f"Error selecting CSV file: {e}")
            return None

    def _load_and_process_csv_data(self) -> bool:
        """
        Load and process CSV data from user-selected file.

        This method:
        1. Lists all CSV files in the data directory
        2. Allows user to select a file (with most recent as default)
        3. Reads the selected file

        Returns:
            True if successful, False otherwise
        """
        print("ORB - Open Range Breakout Analysis")
        print("=" * 40)

        # Let user select CSV file
        selected_file = self._select_csv_file()

        if not selected_file:
            print("No file selected or operation cancelled.")
            return False

        # Read the CSV file
        try:
            print(f"Reading file: {selected_file}")
            self.csv_data = read_csv(selected_file)
            self.current_file = selected_file

            # Extract date from filename (YYYYMMDD.csv format)
            filename = os.path.basename(selected_file)
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
            # (9:30 AM to 4:00 PM ET) - start from premarket to get opening range data
            et_tz = pytz.timezone('America/New_York')
            target_date = (self.csv_date if self.csv_date
                           else datetime.now(et_tz).date())

            start_time = datetime.combine(target_date, time(4, 0),
                                          tzinfo=et_tz)  # Start from premarket
            end_time = datetime.combine(target_date, time(16, 0),
                                        tzinfo=et_tz)

            print(f"Fetching 1-minute data from {start_time} to {end_time}")
            print(f"  (Starting from premarket to ensure opening range data is included)")

            # Determine which feed to use based on configuration
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            feed = 'iex' if "paper" in base_url else 'sip'
            print(f"Using {feed.upper()} data feed for {'paper' if 'paper' in base_url else 'live'} trading")

            # Get stock data using 1-minute timeframe with legacy API
            market_data = {}
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(
                        symbol,
                        tradeapi.TimeFrame.Minute,  # type: ignore
                        start=start_time.isoformat(),
                        end=end_time.isoformat(),
                        feed=feed  # Use IEX for paper trading, SIP for live trading
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

    def _filter_stock_data_by_time(self, symbol_data: pd.DataFrame,
                                   start_time: time,
                                   end_time: time) -> Optional[pd.DataFrame]:
        """
        Filter stock data by time range in ET timezone.

        Args:
            symbol_data: DataFrame with stock data including timestamp column
            start_time: Start time for filtering (e.g., time(9, 30))
            end_time: End time for filtering (e.g., time(10, 15))

        Returns:
            Filtered DataFrame or None if no data or error
        """
        try:
            if symbol_data is None or symbol_data.empty:
                return None

            # Ensure timestamp is datetime
            symbol_data_copy = symbol_data.copy()
            symbol_data_copy['timestamp'] = pd.to_datetime(
                symbol_data_copy['timestamp'])

            # Convert timestamps to ET timezone and filter
            et_tz = pytz.timezone('America/New_York')
            symbol_data_copy['timestamp_et'] = symbol_data_copy[
                'timestamp'].dt.tz_convert(et_tz)
            symbol_data_copy['time_only'] = symbol_data_copy[
                'timestamp_et'].dt.time

            # Filter by time range
            filtered_data = symbol_data_copy[
                (symbol_data_copy['time_only'] >= start_time) &
                (symbol_data_copy['time_only'] <= end_time)
            ].copy()

            # Clean up temporary columns
            filtered_data = filtered_data.drop(
                columns=['timestamp_et', 'time_only'])

            return filtered_data if not filtered_data.empty else None

        except Exception as e:
            print(f"Error filtering stock data by time: {e}")
            return None

    def _pca_data_prep(self, df: pd.DataFrame, symbol: str,
                       data_samples: int = 90) -> bool:
        """
        Prepare data for PCA analysis by taking the first N samples of data
        and collecting metrics.

        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low',
                'close', 'volume']
            symbol: Stock symbol name
            data_samples: Number of data samples to take from the beginning
                (default: 90)

        Returns:
            True if successful, False otherwise
        """
        # Local debugging flag - set to True for detailed debug output
        isDebugging = True
        
        try:
            if isDebugging:
                print(f"\nDEBUG: Starting PCA data prep for symbol: {symbol}")
                print(f"DEBUG: Input DataFrame shape: {df.shape}")
                print(f"DEBUG: Input DataFrame columns: {list(df.columns)}")
            
            # Extract symbol data using the dedicated atom
            symbol_data = extract_symbol_data(df, symbol)

            if symbol_data is None:
                print(f"No data found for symbol: {symbol}")
                return False

            if isDebugging:
                print(f"DEBUG: Extracted symbol data shape: {symbol_data.shape}")
                print(f"DEBUG: Date range in symbol data: {symbol_data['timestamp'].min()} to {symbol_data['timestamp'].max()}")

            # Take the first N lines of data instead of time filtering
            if isDebugging:
                print(f"DEBUG: Taking first {data_samples} lines of data")
                
            filtered_data = symbol_data.head(data_samples).copy()

            if filtered_data is None or filtered_data.empty:
                print(f"No data available for symbol: {symbol}")
                return False

            if isDebugging:
                print(f"DEBUG: First {data_samples} lines data shape: {filtered_data.shape}")
                print(f"DEBUG: Time range in first {data_samples} lines: {filtered_data['timestamp'].min()} to {filtered_data['timestamp'].max()}")

            # Verify there are the expected number of lines of data and return if not
            if len(filtered_data) != data_samples:
                print(f"Expected {data_samples} lines of data for {symbol}, "
                      f"got {len(filtered_data)}")
                if isDebugging:
                    print(f"DEBUG: Data length validation failed - expected {data_samples}, got {len(filtered_data)}")
                return False

            if isDebugging:
                print(f"DEBUG: Data length validation passed - found {data_samples} rows")

            # Calculate ORB levels
            if isDebugging:
                print(f"DEBUG: Calculating ORB levels...")
                
            orb_high, orb_low = calculate_orb_levels(filtered_data)
            
            if isDebugging:
                print(f"DEBUG: ORB levels calculated - High: {orb_high}, Low: {orb_low}")

            # Calculate EMA (9-period) for close prices
            if isDebugging:
                print(f"DEBUG: Calculating EMA (9-period)...")
                
            ema_success, ema_values = calculate_ema(
                filtered_data, price_column='close', period=9)
                
            if isDebugging:
                print(f"DEBUG: EMA calculation - Success: {ema_success}")
                print(f"DEBUG: EMA values type: {type(ema_values)}")
                if ema_values is not None:
                    print(f"DEBUG: EMA values count: {len(ema_values)}")
                    # Convert to list if it's a pandas Series
                    if hasattr(ema_values, 'tolist'):
                        ema_values = ema_values.tolist()
                        print(f"DEBUG: Converted EMA values to list")

            # Calculate EMA (20-period) for close prices
            if isDebugging:
                print(f"DEBUG: Calculating EMA (20-period)...")
                
            ema20_success, ema20_values = calculate_ema(
                filtered_data, price_column='close', period=20)
                
            if isDebugging:
                print(f"DEBUG: EMA20 calculation - Success: {ema20_success}")
                print(f"DEBUG: EMA20 values type: {type(ema20_values)}")
                if ema20_values is not None:
                    print(f"DEBUG: EMA20 values count: {len(ema20_values)}")
                    # Convert to list if it's a pandas Series
                    if hasattr(ema20_values, 'tolist'):
                        ema20_values = ema20_values.tolist()
                        print(f"DEBUG: Converted EMA20 values to list")

            # Calculate VWAP using typical price (HLC/3)
            if isDebugging:
                print(f"DEBUG: Calculating VWAP...")
                
            vwap_success, vwap_values = calculate_vwap_typical(filtered_data)
            
            if isDebugging:
                print(f"DEBUG: VWAP calculation - Success: {vwap_success}")
                print(f"DEBUG: VWAP values type: {type(vwap_values)}")
                if vwap_values is not None:
                    print(f"DEBUG: VWAP values count: {len(vwap_values)}")
                    # Convert to list if it's a pandas Series
                    if hasattr(vwap_values, 'tolist'):
                        vwap_values = vwap_values.tolist()
                        print(f"DEBUG: Converted VWAP values to list")

            # Calculate vector angle using all candlesticks
            if isDebugging:
                print(f"DEBUG: Calculating vector angle (all {data_samples} candles)...")
                
            vector_angle = calculate_vector_angle(
                filtered_data, price_column='close', num_candles=data_samples)
                
            if isDebugging:
                print(f"DEBUG: Vector angle calculated: {vector_angle}")

            # Create independent features for PCA analysis
            if isDebugging:
                print(f"DEBUG: Creating independent features for PCA analysis...")
            
            # Calculate independent features
            pca_row_data = []
            
            # Create a copy with reset index for easier processing
            data_reset = filtered_data.reset_index(drop=True)
            
            for i in range(len(data_reset)):
                row = data_reset.iloc[i]
                
                # Independent feature calculations
                pca_row = {
                    'symbol': symbol,
                    'timestamp': row['timestamp']
                }
                
                # 1. Return features (price-independent)
                if i > 0:
                    prev_close = data_reset.iloc[i-1]['close']
                    pca_row['return_open_close'] = (row['close'] - row['open']) / row['open']
                    pca_row['return_prev_close'] = (row['open'] - prev_close) / prev_close
                else:
                    pca_row['return_open_close'] = 0.0
                    pca_row['return_prev_close'] = 0.0
                
                # 2. Volatility features
                pca_row['intraday_range'] = (row['high'] - row['low']) / row['open']
                pca_row['upper_wick'] = (row['high'] - max(row['open'], row['close'])) / row['open']
                pca_row['lower_wick'] = (min(row['open'], row['close']) - row['low']) / row['open']
                
                # 3. Volume features
                if i >= 5:  # Need at least 5 periods for rolling average
                    avg_volume = data_reset.iloc[max(0, i-4):i+1]['volume'].mean()
                    pca_row['volume_ratio'] = row['volume'] / avg_volume if avg_volume > 0 else 1.0
                else:
                    pca_row['volume_ratio'] = 1.0
                
                # 4. Momentum features
                if i >= 4:  # 5-period momentum
                    prev_close_5 = data_reset.iloc[i-4]['close']
                    pca_row['momentum_5'] = (row['close'] - prev_close_5) / prev_close_5
                else:
                    pca_row['momentum_5'] = 0.0
                
                # 5. ORB-specific features
                if orb_high is not None and orb_low is not None:
                    orb_range = orb_high - orb_low
                    if orb_range > 0:
                        pca_row['close_vs_orb_high'] = (row['close'] - orb_high) / orb_range
                        pca_row['close_vs_orb_low'] = (row['close'] - orb_low) / orb_range
                        pca_row['high_vs_orb_high'] = (row['high'] - orb_high) / orb_range
                        pca_row['low_vs_orb_low'] = (row['low'] - orb_low) / orb_range
                    else:
                        pca_row['close_vs_orb_high'] = 0.0
                        pca_row['close_vs_orb_low'] = 0.0
                        pca_row['high_vs_orb_high'] = 0.0
                        pca_row['low_vs_orb_low'] = 0.0
                else:
                    # Handle case where ORB levels couldn't be calculated
                    pca_row['close_vs_orb_high'] = 0.0
                    pca_row['close_vs_orb_low'] = 0.0
                    pca_row['high_vs_orb_high'] = 0.0
                    pca_row['low_vs_orb_low'] = 0.0
                
                # 6. Technical indicator deviations
                row_index = i
                if (ema_success and ema_values is not None and 
                        row_index < len(ema_values) and ema_values[row_index] is not None):
                    pca_row['close_vs_ema'] = (row['close'] - ema_values[row_index]) / row['close']
                else:
                    pca_row['close_vs_ema'] = 0.0
                
                if (ema20_success and ema20_values is not None and 
                        row_index < len(ema20_values) and ema20_values[row_index] is not None):
                    pca_row['close_vs_ema20'] = (row['close'] - ema20_values[row_index]) / row['close']
                else:
                    pca_row['close_vs_ema20'] = 0.0
                
                if (vwap_success and vwap_values is not None and
                        row_index < len(vwap_values) and vwap_values[row_index] is not None):
                    pca_row['close_vs_vwap'] = (row['close'] - vwap_values[row_index]) / row['close']
                else:
                    pca_row['close_vs_vwap'] = 0.0
                
                # 7. Vector angle (directional momentum)
                pca_row['vector_angle'] = vector_angle
                
                # 8. Time-based features
                minute_of_session = i  # 0-based minute from start
                pca_row['session_progress'] = minute_of_session / (data_samples - 1)
                
                pca_row_data.append(pca_row)

            if isDebugging:
                print(f"DEBUG: Created {len(pca_row_data)} PCA data rows")

            # Create or append to the class variable dataframe
            new_pca_df = pd.DataFrame(pca_row_data)

            if isDebugging:
                print(f"DEBUG: New PCA DataFrame created with shape: {new_pca_df.shape}")
                print(f"DEBUG: Columns in new DataFrame: {list(new_pca_df.columns)}")

            if self.pca_data is None:
                self.pca_data = new_pca_df
                if isDebugging:
                    print(f"DEBUG: Initialized class pca_data with new DataFrame")
            else:
                prev_shape = self.pca_data.shape
                self.pca_data = pd.concat(
                    [self.pca_data, new_pca_df], ignore_index=True)
                if isDebugging:
                    print(f"DEBUG: Appended to existing pca_data - Previous shape: {prev_shape}, New shape: {self.pca_data.shape}")

            print(f"PCA data prepared for {symbol}: "
                  f"{len(new_pca_df)} rows added")
            
            # Print prepared DataFrame if debugging is enabled (using local isDebugging)
            if isDebugging:
                print(f"\nDEBUG: Prepared PCA DataFrame for {symbol}:")
                print("=" * 60)
                print(new_pca_df.to_string(index=False))
                print("=" * 60)
                print(f"DataFrame shape: {new_pca_df.shape}")
                print(f"DataFrame columns: {list(new_pca_df.columns)}")
                print()
            
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

    def _perform_pca_analysis(self) -> bool:
        """
        Perform PCA analysis on all symbols in the market DataFrame.
        
        This method iterates through all unique symbols in the market_df
        and calls _pca_data_prep for each symbol to collect PCA data.
        
        Returns:
            True if successful for at least one symbol, False otherwise
        """
        # Local debugging flag - set to True for detailed debug output
        is_debugging = True
        
        if not hasattr(self, 'market_df') or self.market_df is None:
            print("No market DataFrame available for PCA analysis.")
            return False
            
        try:
            # Get unique symbols from the DataFrame
            symbols = self.market_df['symbol'].unique()
            
            if len(symbols) == 0:
                print("No symbols found in market DataFrame.")
                return False
                
            print(f"\nStarting PCA analysis for {len(symbols)} symbols...")
            print("=" * 50)
            
            # Initialize success counter
            success_count = 0
            
            # Process each symbol
            for symbol in symbols:
                print(f"\nProcessing symbol: {symbol}")
                success = self._pca_data_prep(self.market_df, symbol)
                
                if success:
                    success_count += 1
                    print(f"✓ PCA data preparation successful for {symbol}")
                else:
                    print(f"✗ PCA data preparation failed for {symbol}")
            
            print(f"\nPCA Analysis Summary:")
            print(f"Successful: {success_count}/{len(symbols)} symbols")
            
            if hasattr(self, 'pca_data') and self.pca_data is not None:
                print(f"Total PCA data rows collected: {len(self.pca_data)}")
                print("PCA data columns:", list(self.pca_data.columns))
            
            # Print PCA data if debugging is enabled
            if is_debugging and hasattr(self, 'pca_data') and self.pca_data is not None:
                print(f"\nDEBUG: Complete PCA Data:")
                print("=" * 80)
                print(self.pca_data.to_string(index=False))
                print("=" * 80)
                print(f"Total rows: {len(self.pca_data)}")
                print(f"Columns: {list(self.pca_data.columns)}")
                print()
            
            # Standardize the PCA data
            if success_count > 0:
                print("\nStarting PCA data standardization...")
                standardize_success = self._standardize_pca_data()
                if standardize_success:
                    print("PCA data standardization completed successfully.")
                    
                    # Perform PCA analysis on standardized data
                    print("\nPerforming PCA analysis...")
                    pca_analysis_success = self._perform_pca_computation()
                    if pca_analysis_success:
                        print("PCA computation completed successfully.")
                    else:
                        print("PCA computation failed.")
                else:
                    print("PCA data standardization failed.")
            
            return success_count > 0
            
        except Exception as e:
            print(f"Error performing PCA analysis: {e}")
            return False

    def _standardize_pca_data(self) -> bool:
        """
        Standardize PCA data by removing symbol and timestamp columns,
        extracting volume and vector_angle, and performing statistical
        standardization on price columns.
        
        Returns:
            True if successful, False otherwise
        """
        # Local debugging flag - set to True for detailed debug output
        is_debugging = True
        
        if not hasattr(self, 'pca_data') or self.pca_data is None:
            print("No PCA data available for standardization.")
            return False
            
        try:
            if is_debugging:
                print(f"\nDEBUG: Starting PCA data standardization...")
                print(f"DEBUG: Original PCA data shape: {self.pca_data.shape}")
                print(f"DEBUG: Original columns: {list(self.pca_data.columns)}")
            
            # Step 1: Remove symbol and timestamp, extract volume and vector_angle
            working_df = self.pca_data.copy()
            
            # Remove symbol and timestamp columns (non-feature columns)
            columns_to_remove = ['symbol', 'timestamp']
            for col in columns_to_remove:
                if col in working_df.columns:
                    working_df = working_df.drop(columns=[col])
                    if is_debugging:
                        print(f"DEBUG: Removed column: {col}")
            
            # All remaining columns are independent features - standardize them all together
            if is_debugging:
                print(f"DEBUG: All features will be standardized together")
                print(f"DEBUG: Feature columns: {list(working_df.columns)}")
            
            # No need to separate features - they are all independent now
            
            if is_debugging:
                print(f"DEBUG: Independent feature columns: "
                      f"{list(working_df.columns)}")
                print(f"DEBUG: Feature data shape: {working_df.shape}")
            
            # Step 2: Perform statistical standardization on all features
            try:
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                print("Error: scikit-learn not installed. "
                      "Install with: pip install scikit-learn")
                return False
            
            # Standardize all independent features together
            scaler = StandardScaler()
            feature_columns = working_df.columns.tolist()
            standardized_features = scaler.fit_transform(working_df)
            standardized_df = pd.DataFrame(
                standardized_features, 
                columns=feature_columns,
                index=working_df.index
            )
            
            if is_debugging:
                print(f"DEBUG: Standardized features shape: "
                      f"{standardized_df.shape}")
                print("DEBUG: All features standardization completed")
            
            # Store the standardized data
            self.pca_data_standardized = standardized_df
            
            if is_debugging:
                print(f"DEBUG: Final standardized data shape: "
                      f"{standardized_df.shape}")
                print(f"DEBUG: Final columns: {list(standardized_df.columns)}")
                
                # Step 3: Print the resulting dataframe if debugging
                print("\nDEBUG: Standardized PCA Data:")
                print("=" * 80)
                print(standardized_df.to_string(index=False))
                print("=" * 80)
                print(f"Total rows: {len(standardized_df)}")
                print(f"Columns: {list(standardized_df.columns)}")
                print()
            
            print("PCA data standardization completed successfully.")
            return True
            
        except Exception as e:
            print(f"Error standardizing PCA data: {e}")
            return False

    def _perform_pca_computation(self) -> bool:
        """
        Perform Principal Component Analysis on standardized data.
        
        Returns:
            True if successful, False otherwise
        """
        # Local debugging flag - set to True for detailed debug output
        is_debugging = True
        
        if not hasattr(self, 'pca_data_standardized') or self.pca_data_standardized is None:
            print("No standardized PCA data available for computation.")
            return False
            
        try:
            if is_debugging:
                print("\nDEBUG: Starting PCA computation...")
                print(f"DEBUG: Standardized data shape: {self.pca_data_standardized.shape}")
                print(f"DEBUG: Standardized data columns: {list(self.pca_data_standardized.columns)}")
            
            # Import PCA from sklearn
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                print("Error: scikit-learn not installed. "
                      "Install with: pip install scikit-learn")
                return False
            
            # Prepare data for PCA (remove any NaN values)
            pca_input_data = self.pca_data_standardized.dropna()
            
            if pca_input_data.empty:
                print("No valid data remaining after removing NaN values.")
                return False
                
            if is_debugging:
                print(f"DEBUG: Data shape after NaN removal: {pca_input_data.shape}")
                print(f"DEBUG: Input data ready for PCA")
            
            # Determine number of components (use min of features or samples)
            n_features = pca_input_data.shape[1]
            n_samples = pca_input_data.shape[0]
            n_components = min(n_features, n_samples, 10)  # Limit to 10 components max
            
            if is_debugging:
                print(f"DEBUG: Number of features: {n_features}")
                print(f"DEBUG: Number of samples: {n_samples}")
                print(f"DEBUG: Using {n_components} components for PCA")
            
            # Create and fit PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(pca_input_data)
            
            # Store PCA results
            self.pca_components = pca.components_
            self.pca_explained_variance = pca.explained_variance_
            self.pca_explained_variance_ratio = pca.explained_variance_ratio_
            self.pca_transformed_data = pca_result
            self.pca_feature_names = list(pca_input_data.columns)
            
            if is_debugging:
                print(f"DEBUG: PCA transformation completed")
                print(f"DEBUG: Transformed data shape: {pca_result.shape}")
                print(f"DEBUG: Explained variance ratio: {pca.explained_variance_ratio_}")
                print(f"DEBUG: Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")
                
                # Print detailed results
                print("\nDEBUG: PCA Analysis Results:")
                print("=" * 60)
                print("Component\tExplained Variance\tExplained Variance Ratio")
                print("-" * 60)
                actual_n_components = len(pca.explained_variance_)
                for i in range(actual_n_components):
                    print(f"PC{i+1}\t\t{pca.explained_variance_[i]:.4f}\t\t"
                          f"{pca.explained_variance_ratio_[i]:.4f}")
                print("=" * 60)
                
                # Print feature loadings for first few components
                print("\nDEBUG: Feature Loadings (First 3 Components):")
                print("=" * 80)
                feature_names = list(pca_input_data.columns)
                actual_components = min(3, len(pca.explained_variance_ratio_))
                for i in range(actual_components):
                    print(f"\nPrincipal Component {i+1} "
                          f"(explains {pca.explained_variance_ratio_[i]:.2%} variance):")
                    loadings = [(feature_names[j], pca.components_[i, j]) 
                               for j in range(len(feature_names))]
                    # Sort by absolute loading value
                    loadings.sort(key=lambda x: abs(x[1]), reverse=True)
                    for feature, loading in loadings:
                        print(f"  {feature}: {loading:.4f}")
                print("=" * 80)
                print()
            
            # Summary for user
            total_variance_explained = pca.explained_variance_ratio_.sum()
            print(f"PCA completed with {n_components} components")
            print(f"Total variance explained: {total_variance_explained:.2%}")
            print(f"First component explains: {pca.explained_variance_ratio_[0]:.2%}")
            
            return True
            
        except Exception as e:
            print(f"Error performing PCA computation: {e}")
            return False

    def _generate_candle_charts(self) -> bool:
        """
        Generate candlestick charts with volume for each stock symbol.
        Always fetches fresh, complete market data from 9:30 AM to 4:00 PM ET 
        for accurate charting instead of using potentially incomplete stored data.

        Returns:
            True if successful, False otherwise
        """
        if not self.csv_data:
            print("No CSV data available for charting.")
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

            # Remove duplicates and sort
            symbols = sorted(list(set(symbols)))

            print(f"\nGenerating candlestick charts for {len(symbols)} symbols...")
            print("Fetching fresh market data from 9:30 AM to 4:00 PM ET for complete charts...")

            # Create plots directory if it doesn't exist
            plots_dir = 'plots'
            os.makedirs(plots_dir, exist_ok=True)

            # Set up time range for complete market data (9:30 AM to 4:00 PM ET)
            # Start from premarket (4:00 AM) to ensure we get opening range data
            et_tz = pytz.timezone('America/New_York')
            target_date = (self.csv_date if self.csv_date 
                           else datetime.now(et_tz).date())

            start_time = datetime.combine(target_date, time(4, 0), tzinfo=et_tz)  # Start from premarket
            end_time = datetime.combine(target_date, time(16, 0), tzinfo=et_tz)

            # Determine which feed to use based on configuration
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            feed = 'iex' if "paper" in base_url else 'sip'
            
            print(f"Using {feed.upper()} data feed for {'paper' if 'paper' in base_url else 'live'} trading")
            print(f"Fetching complete market data from {start_time} to {end_time}")
            print(f"  (Starting from premarket to ensure opening range data is included)")

            # Generate chart for each symbol
            success_count = 0
            alert_type_name = "super alerts" if self.plot_super_alerts else "regular alerts"
            
            for symbol in symbols:
                try:
                    # Fetch fresh, complete market data for this symbol
                    print(f"Fetching fresh market data for {symbol}...")
                    print(f"  Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    bars = self.api.get_bars(
                        symbol,
                        tradeapi.TimeFrame.Minute,  # type: ignore
                        start=start_time.isoformat(),
                        end=end_time.isoformat(),
                        limit=10000,  # Ensure we get all data (6.5 hours * 60 minutes = 390 minutes max)
                        feed=feed  # Use IEX for paper trading, SIP for live trading
                    )
                    
                    if not bars:
                        print(f"No market data available for {symbol}")
                        continue
                    
                    # Convert bars to DataFrame format expected by plot_candle_chart
                    symbol_data = []
                    for bar in bars:
                        bar_data = {
                            'timestamp': bar.t.isoformat(),
                            'open': float(bar.o),
                            'high': float(bar.h),
                            'low': float(bar.l),
                            'close': float(bar.c),
                            'volume': int(bar.v),
                            'symbol': symbol
                        }
                        symbol_data.append(bar_data)
                    
                    # Create DataFrame for this symbol
                    symbol_df = pd.DataFrame(symbol_data)
                    symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
                    
                    print(f"Retrieved {len(symbol_df)} minutes of data for {symbol}")
                    if len(symbol_df) > 0:
                        min_time = symbol_df['timestamp'].min()
                        max_time = symbol_df['timestamp'].max()
                        print(f"Data range: {min_time} to {max_time}")
                        
                        # Check if we got the full trading session
                        expected_start = start_time.replace(tzinfo=None)
                        expected_end = end_time.replace(tzinfo=None)
                        
                        if min_time.replace(tzinfo=None) > expected_start:
                            print(f"⚠️  Data starts later than expected market open (9:30 AM)")
                        if max_time.replace(tzinfo=None) < expected_end:
                            print(f"⚠️  Data ends earlier than expected market close (4:00 PM)")
                        
                        # Calculate coverage
                        session_duration = (expected_end - expected_start).total_seconds() / 60  # minutes
                        actual_duration = (max_time - min_time).total_seconds() / 60  # minutes
                        coverage = (actual_duration / session_duration) * 100 if session_duration > 0 else 0
                        print(f"Session coverage: {coverage:.1f}% ({len(symbol_df)} of ~{session_duration:.0f} possible minutes)")
                    
                    # Load alerts for this symbol if csv_date is available
                    alerts = []
                    if self.csv_date:
                        alerts = self._load_alerts_for_symbol(symbol, self.csv_date)
                        if alerts:
                            print(f"Loaded {len(alerts)} {alert_type_name} for {symbol}")
                    
                    # Generate chart with fresh data and alerts
                    if plot_candle_chart(symbol_df, symbol, plots_dir, alerts):
                        success_count += 1
                        print(f"✓ Chart generated for {symbol}")
                    else:
                        print(f"✗ Failed to generate chart for {symbol}")
                        
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue

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

        # Generate candlestick charts for all symbols
        # This now fetches fresh, complete market data directly
        chart_success = self._generate_candle_charts()

        if not chart_success:
            print("Failed to generate charts.")
            return False

        # Get market data for ORB analysis (used for PCA analysis)
        market_data_success = self._get_orb_market_data()

        if not market_data_success:
            print("Failed to get market data for ORB analysis.")
            return False

        # Process market data from saved JSON file (used for PCA analysis)
        process_success = self._load_market_dataframe()

        if not process_success:
            print("Failed to process market data.")
            return False

        # Perform PCA analysis on all symbols
        pca_success = self._perform_pca_analysis()

        if not pca_success:
            print("Failed to perform PCA analysis.")
            return False

        return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Analysis Tool")
    
    parser.add_argument(
        "--plot-alerts",
        action="store_true",
        help="Plot regular alerts instead of super alerts (default: plot super alerts)"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to setup and run the ORB class.
    Handles keyboard interrupts gracefully.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Determine whether to plot super alerts or regular alerts
        plot_super_alerts = not args.plot_alerts  # Default is True (super alerts)
        
        # Create and run ORB analysis
        orb = ORB(plot_super_alerts=plot_super_alerts)
        
        alert_type = "super alerts" if plot_super_alerts else "regular alerts"
        print(f"ORB analysis will plot {alert_type}")
        
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
