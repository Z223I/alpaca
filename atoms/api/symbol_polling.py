"""
Symbol Polling System for Alpaca API

This module provides real-time price monitoring using WebSocket subscriptions
for stock symbols loaded from CSV files. It monitors symbols from daily data
files and gainers files, streaming their prices via WebSocket.

**IMPORTANT**: This module was updated from REST API polling to WebSocket streaming.
See specs/symbol_polling.md for complete documentation, architecture details,
performance comparison, and migration notes.

Usage Examples:
    # Test mode (no API calls, uses static data)
    python atoms/api/symbol_polling.py --test

    # Monitor specific symbol (single symbol)
    python atoms/api/symbol_polling.py --symbol AAPL --test --verbose

    # Monitor multiple symbols (NO SPACES after commas - recommended)
    python atoms/api/symbol_polling.py --symbol aapl,googl,msft --test

    # Monitor multiple symbols (with spaces - must use quotes)
    python atoms/api/symbol_polling.py --symbol "aapl, googl, msft" --test

    # Live WebSocket for specific symbols (no spaces)
    python atoms/api/symbol_polling.py --symbol AAPL,TSLA --verbose

    # Use historical data from specific date (YYYYMMDD format)
    python atoms/api/symbol_polling.py --date 20251017 --test

    # Live WebSocket streaming with historical data
    python atoms/api/symbol_polling.py --date 20251017 --verbose

    # Live WebSocket streaming with verbose output (today's data from files)
    python atoms/api/symbol_polling.py --verbose

    # Different account configuration
    python atoms/api/symbol_polling.py --account-name Janice --account live

    # Production usage with default settings (Bruce/paper account)
    python atoms/api/symbol_polling.py

Features:
    - Loads symbols from data/{YYYYMMDD}.csv files
    - Monitors gainers_nasdaq_amex.csv for real-time updates
    - Real-time WebSocket streaming for trade data (time and sales)
    - Supports multiple account configurations
    - File system monitoring for automatic symbol updates
    - Automatic subscription updates when symbol list changes
    - Sub-second latency for price updates
    - Async/await architecture for efficient concurrent operations

Architecture:
    - Uses AlpacaStreamClient from atoms/websocket/alpaca_stream.py
    - Single WebSocket connection for all symbols (no rate limits)
    - Automatic reconnection on connection failures
    - Dynamic subscription management when symbol list changes

Documentation:
    See specs/symbol_polling.md for:
    - Complete architecture documentation
    - WebSocket vs REST performance comparison
    - Technical implementation details
    - Testing instructions
    - Migration notes from previous version
"""

import asyncio
import argparse
import os
import sys
import csv
import glob
import datetime
from typing import Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'code'))

from atoms.websocket.alpaca_stream import AlpacaStreamClient, MarketData  # noqa: E402

# Configuration constants
SYMBOL_UPDATE_INTERVAL = 10  # seconds between symbol list checks


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

    def __init__(self, test_mode=False, verbose=False, target_date=None, symbols_override=None):
        self.test_mode = test_mode
        self.verbose = verbose
        self.target_date = target_date  # YYYYMMDD format string, or None for today
        self.symbols_override = symbols_override  # Set of symbols to use instead of files
        self.symbols = set()
        self.data_dir = os.path.join(project_root, 'data')
        self.historical_dir = os.path.join(project_root, 'historical_data')

    def get_latest_data_file(self) -> Optional[str]:
        """Get data file based on mode and target_date."""
        if self.test_mode:
            # Test mode: Get the most recent data file available
            pattern = os.path.join(self.data_dir, '*.csv')
            files = glob.glob(pattern)
            if not files:
                return None
            # Sort by filename (which contains date) and get the latest
            return max(files)
        elif self.target_date:
            # Use specified historical date (YYYYMMDD format)
            target_file = os.path.join(self.data_dir, f'{self.target_date}.csv')

            if os.path.exists(target_file):
                return target_file
            else:
                if self.verbose:
                    print(f"Historical data file not found: {target_file}")
                return None
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
        """Get gainers file based on mode and target_date."""
        if self.test_mode:
            # Test mode: Get the most recent gainers file available
            pattern = os.path.join(self.historical_dir,
                                   '*/market/gainers_nasdaq_amex.csv')
            files = glob.glob(pattern)
            if not files:
                return None
            # Sort by path (which contains date) and get the latest
            return max(files)
        elif self.target_date:
            # Use specified historical date (convert YYYYMMDD to YYYY-MM-DD)
            try:
                date_obj = datetime.datetime.strptime(self.target_date, '%Y%m%d')
                date_str = date_obj.strftime('%Y-%m-%d')
                target_pattern = os.path.join(self.historical_dir,
                                              f'{date_str}/market/gainers_nasdaq_amex.csv')

                if os.path.exists(target_pattern):
                    return target_pattern
                else:
                    if self.verbose:
                        print(f"Historical gainers file not found: {target_pattern}")
                    return None
            except ValueError:
                print(f"Invalid date format: {self.target_date} (expected YYYYMMDD)")
                return None
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
        """Update the symbol list from both data sources or use override.

        If symbols_override is set, use those symbols instead of reading from files.
        """
        # If override is provided, use it and skip file reading
        if self.symbols_override is not None:
            if self.symbols != self.symbols_override:
                old_count = len(self.symbols)
                self.symbols = self.symbols_override
                if self.verbose:
                    print(f"Using symbol override: {old_count} -> "
                          f"{len(self.symbols)} symbols: {sorted(self.symbols)}")
            return self.symbols

        # Otherwise, load from files as usual
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
    """Handles real-time price monitoring using WebSocket streams."""

    def __init__(self, account_name='Bruce', account='paper',
                 verbose=False, test_mode=False, target_date=None, symbols_override=None):
        self.account_name = account_name
        self.account = account
        self.verbose = verbose
        self.test_mode = test_mode
        self.target_date = target_date
        self.symbol_manager = SymbolManager(test_mode=test_mode,
                                            verbose=verbose,
                                            target_date=target_date,
                                            symbols_override=symbols_override)
        self.stream_client = None
        self.running = False
        self.observer = None
        self.current_symbols = set()

        # Initialize WebSocket stream client
        if not test_mode:
            try:
                # Get API credentials directly from config
                import os
                sys.path.append(os.path.join(project_root, 'code'))

                try:
                    from alpaca_config import get_api_credentials
                    api_key, secret_key, base_url = get_api_credentials(
                        "alpaca", account_name, account
                    )
                except ImportError:
                    # Fallback to environment variables
                    api_key = os.getenv('ALPACA_API_KEY')
                    secret_key = os.getenv('ALPACA_SECRET_KEY')
                    base_url = os.getenv('ALPACA_BASE_URL')

                self.stream_client = AlpacaStreamClient(
                    api_key=api_key,
                    secret_key=secret_key,
                    base_url=base_url
                )

                # Register data handler
                self.stream_client.add_data_handler(self.handle_market_data)

                if verbose:
                    print(f"Initialized WebSocket stream client for "
                          f"{account_name}/{account}")
            except Exception as e:
                print(f"Error initializing WebSocket client: {e}")
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

    def handle_market_data(self, data: MarketData):
        """
        Handle incoming market data from WebSocket stream.

        All timestamps are in Eastern Time (ET).
        Volume is in individual shares (1 share = 1, NOT thousands).

        Args:
            data: MarketData object from WebSocket (timestamp already in ET)
        """
        # Format price to show minimum 2 decimal places
        price_display = f"{data.price:.2f}"

        # Format timestamp - data.timestamp is already timezone-naive Eastern Time
        # Show full date and time with ET label
        time_display = data.timestamp.strftime('%Y-%m-%d %H:%M:%S ET')

        # Print real-time trade data (time and sales)
        # Volume is displayed as individual shares (not thousands)
        print(f"TRADE: {data.symbol} | Price: ${price_display} | "
              f"Volume: {data.volume:,} | Time: {time_display}")

    async def update_subscriptions(self):
        """Monitor symbol list and update WebSocket subscriptions when changed."""
        while self.running:
            try:
                # Get current symbol list
                new_symbols = self.symbol_manager.update_symbols()

                if new_symbols != self.current_symbols:
                    # Symbol list has changed
                    if self.verbose:
                        print(f"Symbol list changed: {len(self.current_symbols)} -> "
                              f"{len(new_symbols)} symbols")

                    if new_symbols:
                        # Subscribe to new symbol list
                        if self.test_mode:
                            # In test mode, just simulate subscriptions
                            print(f"TEST MODE: Would subscribe to {len(new_symbols)} symbols")
                            for symbol in list(new_symbols)[:5]:
                                print(f"  - {symbol}")
                        else:
                            # Update WebSocket subscriptions
                            symbols_list = list(new_symbols)
                            success = await self.stream_client.subscribe_bars(symbols_list)
                            if success:
                                if self.verbose:
                                    print(f"Subscribed to {len(symbols_list)} symbols")
                            else:
                                print("Failed to update subscriptions")

                    self.current_symbols = new_symbols

                # Wait before checking again
                await asyncio.sleep(SYMBOL_UPDATE_INTERVAL)

            except Exception as e:
                print(f"Error updating subscriptions: {e}")
                await asyncio.sleep(SYMBOL_UPDATE_INTERVAL)

    async def run_stream(self):
        """Main WebSocket streaming loop."""
        try:
            # Initial symbol list
            symbols = self.symbol_manager.update_symbols()
            if not symbols:
                print("No symbols to monitor")
                return

            # Connect to WebSocket
            if not await self.stream_client.connect():
                print("Failed to connect to WebSocket")
                return

            # Subscribe to initial symbols
            symbols_list = list(symbols)
            if not await self.stream_client.subscribe_bars(symbols_list):
                print("Failed to subscribe to symbols")
                return

            self.current_symbols = symbols

            if self.verbose:
                print(f"WebSocket streaming started for {len(symbols)} symbols")

            # Run subscription updater and listener concurrently
            await asyncio.gather(
                self.update_subscriptions(),
                self.stream_client.listen()
            )

        except Exception as e:
            print(f"Error in WebSocket stream: {e}")
        finally:
            await self.stream_client.disconnect()

    async def run_test_mode(self):
        """Run in test mode with simulated data.

        All timestamps are in Eastern Time (ET).
        """
        import random
        import pytz

        # Get symbols
        symbols = self.symbol_manager.update_symbols()
        if not symbols:
            print("No symbols found in test mode")
            return

        print(f"TEST MODE: Simulating WebSocket stream for {len(symbols)} symbols")
        print(f"First 5 symbols: {list(symbols)[:5]}")

        # Simulate real-time updates
        while self.running:
            for symbol in list(symbols)[:5]:  # Just show first 5 for testing
                # Simulate market data with Eastern Time
                et_tz = pytz.timezone('US/Eastern')
                timestamp_et = datetime.datetime.now(et_tz).replace(tzinfo=None)
                time_display = timestamp_et.strftime('%Y-%m-%d %H:%M:%S ET')

                price = random.uniform(50, 500)
                volume = random.randint(100, 10000)

                print(f"TRADE: {symbol} | Price: ${price:.2f} | "
                      f"Volume: {volume} | Time: {time_display}")

            await asyncio.sleep(2)

            # Check for symbol updates
            new_symbols = self.symbol_manager.update_symbols()
            if new_symbols != symbols:
                print(f"\nSymbol list updated: {len(symbols)} -> {len(new_symbols)} symbols")
                symbols = new_symbols

    async def run(self):
        """Start the price monitoring system."""
        self.running = True

        if self.test_mode:
            await self.run_test_mode()
        else:
            await self.run_stream()

    async def stop(self):
        """Stop the WebSocket streaming and file monitoring."""
        self.running = False
        if self.stream_client:
            await self.stream_client.disconnect()
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
    parser.add_argument('--date', type=str, default=None,
                        help='Use historical data from specific date (YYYYMMDD format, '
                        'e.g., 20251017). If not specified, uses today\'s date.')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Monitor specific symbol(s). Comma-separated list for multiple '
                        '(e.g., AAPL or AAPL,GOOGL,MSFT - NO SPACES, or use quotes: "AAPL, GOOGL, MSFT"). '
                        'Symbols are automatically converted to uppercase. '
                        'When specified, symbols are NOT read from files.')

    return parser.parse_args()


async def test_functionality(target_date=None):
    """Test basic functionality without infinite loop."""
    print("Testing symbol polling functionality...")

    # Test symbol manager
    symbol_manager = SymbolManager(test_mode=True, verbose=True, target_date=target_date)
    symbols = symbol_manager.update_symbols()
    # Show first 5
    print(f"Found {len(symbols)} symbols: {list(symbols)[:5]}...")

    # Test price poller initialization
    poller = PricePoller(test_mode=True, verbose=True, target_date=target_date)
    print("Price poller initialized successfully")

    # Test simulated WebSocket data
    print("Testing simulated WebSocket stream (5 seconds)...")
    poller.running = True

    # Run for 5 seconds
    try:
        await asyncio.wait_for(poller.run_test_mode(), timeout=5.0)
    except asyncio.TimeoutError:
        poller.running = False
        print("\nFunctionality test completed successfully!")


async def main_async():
    """Async main entry point."""
    args = parse_args()

    # Parse and process --symbol argument
    symbols_override = None
    if args.symbol:
        # Convert to uppercase and split by comma
        symbols_list = [s.strip().upper() for s in args.symbol.split(',')]
        symbols_override = set(symbols_list)

    # Add a special test mode that just runs functionality test
    if args.test and len(sys.argv) == 2:  # Only --test flag
        await test_functionality(target_date=args.date)
        return

    if args.verbose:
        print("Starting WebSocket symbol streaming system...")
        print(f"Account: {args.account_name}/{args.account}")
        print(f"Test mode: {args.test}")
        if args.date:
            print(f"Using historical date: {args.date}")
        if symbols_override:
            print(f"Monitoring specific symbols: {sorted(symbols_override)}")
        else:
            print(f"Symbol update interval: {SYMBOL_UPDATE_INTERVAL} seconds")

    # Create price poller
    poller = PricePoller(
        account_name=args.account_name,
        account=args.account,
        verbose=args.verbose,
        test_mode=args.test,
        target_date=args.date,
        symbols_override=symbols_override
    )

    try:
        if args.verbose:
            print("WebSocket streaming started. Press Ctrl+C to stop.")

        # Run WebSocket stream
        await poller.run()

    except KeyboardInterrupt:
        if args.verbose:
            print("\nStopping WebSocket stream...")
        await poller.stop()
        sys.exit(0)
    except Exception as e:
        print(f"Error in main: {e}")
        await poller.stop()
        sys.exit(1)


def main():
    """Main entry point - wrapper for async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
