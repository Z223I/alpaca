"""
Symbol Polling System for Alpaca API

This module provides price polling functionality for stock symbols loaded
from CSV files. It monitors symbols from daily data files and gainers files,
polling their prices at regular intervals.

Usage Examples:
    # Test mode (no API calls, uses static data)
    python atoms/api/symbol_polling.py --test

    # Test mode with verbose output
    python atoms/api/symbol_polling.py --test --verbose

    # Live polling with verbose output
    python atoms/api/symbol_polling.py --verbose

    # Different account configuration
    python atoms/api/symbol_polling.py --account-name Janice --account live

    # Production usage with default settings (Bruce/paper account)
    python atoms/api/symbol_polling.py

Features:
    - Loads symbols from data/{YYYYMMDD}.csv files
    - Monitors gainers_nasdaq_amex.csv for real-time updates
    - Polls prices every 5 seconds using Alpaca API
    - Supports multiple account configurations
    - File system monitoring for automatic symbol updates
    - Thread-safe price polling with error handling
"""

import threading
import time
import argparse
import os
import sys
import csv
import glob
import datetime
from typing import Set, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'code'))

from atoms.api.init_alpaca_client import init_alpaca_client  # noqa: E402

# Configuration constants
PRICE_POLL_INTERVAL = 5  # seconds


class GainersFileHandler(FileSystemEventHandler):
    """Handles file system events for gainers CSV file updates."""

    def __init__(self, symbol_manager):
        self.symbol_manager = symbol_manager

    def on_modified(self, event):
        """Handle file modification events."""
        if (not event.is_directory and
                event.src_path.endswith('gainers_nasdaq_amex.csv')):
            print(f"Gainers file updated: {event.src_path}")
            self.symbol_manager.update_symbols()


class SymbolManager:
    """Manages symbol loading and updating from CSV sources."""

    def __init__(self, test_mode=False, verbose=False):
        self.test_mode = test_mode
        self.verbose = verbose
        self.symbols = set()
        self.data_dir = os.path.join(project_root, 'data')
        self.historical_dir = os.path.join(project_root, 'historical_data')

    def get_latest_data_file(self) -> Optional[str]:
        """Get data file based on mode: most recent in test mode, today only in normal mode."""
        if self.test_mode:
            # Test mode: Get the most recent data file available
            pattern = os.path.join(self.data_dir, '*.csv')
            files = glob.glob(pattern)
            if not files:
                return None
            # Sort by filename (which contains date) and get the latest
            return max(files)
        else:
            # Normal mode: Only use today's data file
            today = datetime.date.today().strftime('%Y%m%d')
            today_file = os.path.join(self.data_dir, f'{today}.csv')

            if os.path.exists(today_file):
                return today_file

            # If today's file doesn't exist, return None (wait for it)
            if self.verbose:
                print(f"Waiting for today's data file: {today_file}")
            return None

    def get_latest_gainers_file(self) -> Optional[str]:
        """Get gainers file based on mode: most recent in test mode, today only in normal mode."""
        if self.test_mode:
            # Test mode: Get the most recent gainers file available
            pattern = os.path.join(self.historical_dir,
                                   '*/market/gainers_nasdaq_amex.csv')
            files = glob.glob(pattern)
            if not files:
                return None
            # Sort by path (which contains date) and get the latest
            return max(files)
        else:
            # Normal mode: Only use today's gainers file
            today = datetime.date.today().strftime('%Y-%m-%d')
            today_pattern = os.path.join(self.historical_dir,
                                         f'{today}/market/gainers_nasdaq_amex.csv')

            if os.path.exists(today_pattern):
                return today_pattern

            # If today's file doesn't exist, return None (wait for it)
            if self.verbose:
                print(f"Waiting for today's gainers file: {today_pattern}")
            return None

    def load_symbols_from_data_file(self, file_path: str) -> Set[str]:
        """Load symbols from a data/{YYYYMMDD}.csv file."""
        symbols = set()
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('Symbol', '').strip()
                    if symbol:
                        symbols.add(symbol)
        except Exception as e:
            print(f"Error loading symbols from {file_path}: {e}")
        return symbols

    def load_symbols_from_gainers_file(self, file_path: str) -> Set[str]:
        """Load symbols from a gainers_nasdaq_amex.csv file."""
        symbols = set()
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('symbol', '').strip()
                    if symbol:
                        symbols.add(symbol)
        except Exception as e:
            print(f"Error loading symbols from {file_path}: {e}")
        return symbols

    def update_symbols(self):
        """Update the symbol list from both data sources."""
        new_symbols = set()

        # Load from data file
        data_file = self.get_latest_data_file()
        if data_file:
            data_symbols = self.load_symbols_from_data_file(data_file)
            new_symbols.update(data_symbols)
            if self.verbose:
                print(f"Loaded {len(data_symbols)} symbols from {data_file}")

        # Load from gainers file
        gainers_file = self.get_latest_gainers_file()
        if gainers_file:
            gainers_symbols = self.load_symbols_from_gainers_file(
                gainers_file)
            new_symbols.update(gainers_symbols)
            if self.verbose:
                print(f"Loaded {len(gainers_symbols)} symbols from "
                      f"{gainers_file}")

        if new_symbols != self.symbols:
            old_count = len(self.symbols)
            self.symbols = new_symbols
            if self.verbose:
                print(f"Symbol list updated: {old_count} -> "
                      f"{len(self.symbols)} symbols")

        return self.symbols


class PricePoller:
    """Handles price polling for symbols."""

    def __init__(self, account_name='Bruce', account='paper',
                 verbose=False, test_mode=False):
        self.account_name = account_name
        self.account = account
        self.verbose = verbose
        self.test_mode = test_mode
        self.symbol_manager = SymbolManager(test_mode=test_mode,
                                            verbose=verbose)
        self.rest = None
        self.running = False
        self.observer = None

        # Initialize Alpaca client
        if not test_mode:
            try:
                self.rest = init_alpaca_client("alpaca", account_name,
                                               account)
                if verbose:
                    print(f"Initialized Alpaca client for "
                          f"{account_name}/{account}")
            except Exception as e:
                print(f"Error initializing Alpaca client: {e}")
                sys.exit(1)

        # Setup file monitoring
        self.setup_file_monitoring()

    def setup_file_monitoring(self):
        """Setup file system monitoring for gainers file updates."""
        if self.test_mode:
            return

        try:
            self.observer = Observer()
            handler = GainersFileHandler(self.symbol_manager)

            # Monitor the historical_data directory
            historical_dir = os.path.join(project_root, 'historical_data')
            if os.path.exists(historical_dir):
                self.observer.schedule(handler, historical_dir,
                                       recursive=True)
                self.observer.start()
                if self.verbose:
                    print(f"Started monitoring {historical_dir} for "
                          f"gainers file updates")
        except Exception as e:
            print(f"Warning: Could not setup file monitoring: {e}")

    def send_alert(self, alert_type: str, message: str,
                   details: str = "", symbols: List[str] = None):
        """Local send_alert method that prints the arguments."""
        print(f"ALERT: {alert_type} | {message} | {details} | "
              f"{symbols or []}")

    def poll_prices(self):
        """Main price polling loop."""
        while self.running:
            try:
                # Update symbol list
                symbols = self.symbol_manager.update_symbols()

                if not symbols:
                    if self.verbose:
                        print("No symbols to poll")
                    time.sleep(PRICE_POLL_INTERVAL)
                    continue

                # Poll prices for each symbol
                for symbol in symbols:
                    try:
                        if self.test_mode:
                            # In test mode, just simulate price data
                            price_display = f"TEST_PRICE_{symbol}"
                        else:
                            trade = self.rest.get_latest_trade(symbol)
                            price = trade.price
                            # Format price to show minimum 2 decimal places
                            price_display = f"{price:.2f}"

                        self.send_alert("Price", f"{symbol}: {price_display}",
                                        "", [symbol])

                    except Exception as e:
                        if self.verbose:
                            print(f"Error polling {symbol}: {e}")

            except Exception as e:
                print(f"Price poll error: {e}")

            time.sleep(PRICE_POLL_INTERVAL)

    def run_price_poll_thread(self):
        """Start the price polling thread."""
        self.running = True
        thread = threading.Thread(target=self.poll_prices, daemon=True)
        thread.start()
        if self.verbose:
            print(f"Started price polling thread "
                  f"(interval: {PRICE_POLL_INTERVAL}s)")
        return thread

    def stop(self):
        """Stop the price polling and file monitoring."""
        self.running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Symbol Price Polling System')

    parser.add_argument('--account-name', default='Bruce',
                        help='Account name for API credentials '
                        '(default: Bruce)')
    parser.add_argument('--account', default='paper',
                        help='Account type: paper, live, cash '
                        '(default: paper)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--test', action='store_true',
                        help='Use test mode with static data (no API calls)')

    return parser.parse_args()


def test_functionality():
    """Test basic functionality without infinite loop."""
    print("Testing symbol polling functionality...")

    # Test symbol manager
    symbol_manager = SymbolManager(test_mode=True, verbose=True)
    symbols = symbol_manager.update_symbols()
    # Show first 5
    print(f"Found {len(symbols)} symbols: {list(symbols)[:5]}...")

    # Test price poller initialization
    poller = PricePoller(test_mode=True, verbose=True)
    print("Price poller initialized successfully")

    # Test single poll iteration (without threading)
    print("Testing single poll iteration...")
    symbols = poller.symbol_manager.update_symbols()
    if symbols:
        # Test with first few symbols
        test_symbols = list(symbols)[:3]
        for symbol in test_symbols:
            poller.send_alert("Price", f"{symbol}: TEST_PRICE_{symbol}", "", [symbol])

    print("Functionality test completed successfully!")


def main():
    """Main entry point."""
    args = parse_args()

    # Add a special test mode that just runs functionality test
    if args.test and len(sys.argv) == 2:  # Only --test flag
        test_functionality()
        return

    if args.verbose:
        print("Starting symbol polling system...")
        print(f"Account: {args.account_name}/{args.account}")
        print(f"Test mode: {args.test}")
        print(f"Poll interval: {PRICE_POLL_INTERVAL} seconds")

    # Create price poller
    poller = PricePoller(
        account_name=args.account_name,
        account=args.account,
        verbose=args.verbose,
        test_mode=args.test
    )

    try:
        # Start polling thread
        poller.run_price_poll_thread()

        if args.verbose:
            print("Price polling started. Press Ctrl+C to stop.")

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        if args.verbose:
            print("\nStopping price polling...")
        poller.stop()
        sys.exit(0)
    except Exception as e:
        print(f"Error in main: {e}")
        poller.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
