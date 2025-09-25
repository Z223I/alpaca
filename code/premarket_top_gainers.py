#!/usr/bin/env python3
"""
Premarket Top Gainers Scanner

A specialized scanner that identifies top gaining stocks by comparing current prices
to the previous market close. Uses 5-minute candles and can run at any time of day
to find stocks that have gained since the last market close.

Strategy:
- Collect 5-minute bars for the last 7 days
- Identify the last market close (4:00 PM ET on most recent trading day)
- Filter data to only include bars since that last market close
- Calculate gains from previous close to current price (premarket or regular hours)
- Rank stocks by highest percentage gains

Can be run during premarket hours (4:00-9:30 AM ET) for real-time premarket analysis,
or at any other time to analyze post-close and premarket activity since last close.

Mirrors the structure of alpaca_screener.py for consistency.
"""

"""
python code/premarket_top_gainers.py  --exchanges NASDAQ AMEX  --max-symbols 7000  --min-price 0.75  --max-price 40.00  --min-volume 50000 --top-gainers 20 --export-csv gainers_nasdaq_amex.csv --verbose
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional
import pandas as pd
import pytz
import alpaca_trade_api as tradeapi

# Add the atoms directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'atoms'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from atoms.api.init_alpaca_client import init_alpaca_client


@dataclass
class PremarketCriteria:
    """Configuration class for premarket top gainers screening."""
    # Minimum thresholds
    min_price: Optional[float] = 0.75
    max_price: Optional[float] = None
    min_volume: Optional[int] = 50_000  # Volume threshold (matches alpaca_screener.py)
    min_gain_percent: Optional[float] = 1.0  # Minimum gain to be considered

    # Data collection settings
    feed: str = "sip"  # iex, sip, or other feed names
    max_symbols: int = 1000  # Smaller set for premarket efficiency
    lookback_days: int = 7  # Days to look back for finding last market close

    # Exchange filtering (NYSE, NASDAQ, and AMEX only for safety)
    exchanges: Optional[List[str]] = None
    specific_symbols: Optional[List[str]] = None

    # Result limiting (match alpaca_screener.py interface)
    top_gainers: Optional[int] = 20  # Top N gainers to return
    top_losers: Optional[int] = None  # Top N losers (not used but matches interface)


@dataclass
class PremarketResult:
    """Data class for premarket top gainers results."""
    symbol: str
    current_price: float
    previous_close: float
    gain_percent: float
    premarket_volume: int
    premarket_high: float
    premarket_low: float
    premarket_range: float
    current_timestamp: datetime
    previous_close_timestamp: datetime

    # Additional metrics
    dollar_volume: float = 0.0
    total_premarket_bars: int = 0

    def __post_init__(self):
        self.dollar_volume = self.premarket_volume * self.current_price
        self.premarket_range = self.premarket_high - self.premarket_low


class PremarketTopGainersScanner:
    """Main premarket top gainers scanner class."""

    def __init__(self, provider: str = "alpaca", account: str = "Bruce", environment: str = "paper",
                 verbose: bool = False):
        """
        Initialize the premarket scanner.

        Args:
            provider: API provider (default: "alpaca")
            account: Account name (default: "Bruce")
            environment: Environment type (default: "paper")
            verbose: Enable verbose logging (default: False)
        """
        self.client = init_alpaca_client(provider, account, environment)
        self.provider = provider
        self.account = account
        self.environment = environment
        self.verbose = verbose

        # Eastern Time zone for all market operations
        self.et_tz = pytz.timezone('US/Eastern')

        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)  # 9:30 AM ET
        self.market_close = dt_time(16, 0)  # 4:00 PM ET
        self.premarket_start = dt_time(4, 0)  # 4:00 AM ET

        # Rate limiting configuration
        self.rate_limit_calls_per_minute = 200
        self.call_times = []

        if self.verbose:
            print(f"Initialized Premarket Scanner - Account: {account}, Environment: {environment}")
            current_et = datetime.now(self.et_tz)
            print(f"Current ET time: {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    def _rate_limit_check(self):
        """Implement rate limiting to stay within API limits."""
        current_time = time.time()

        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if current_time - t < 60]

        if len(self.call_times) >= self.rate_limit_calls_per_minute:
            sleep_time = 60 - (current_time - self.call_times[0])
            if sleep_time > 0:
                if self.verbose:
                    print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

        self.call_times.append(current_time)

    def is_premarket_hours(self) -> bool:
        """Check if current time is during premarket hours (4:00 AM - 9:30 AM ET)."""
        current_et = datetime.now(self.et_tz)
        current_time = current_et.time()
        current_weekday = current_et.weekday()  # Monday = 0, Sunday = 6

        # Only Monday-Friday
        if current_weekday >= 5:
            return False

        # Check if between 4:00 AM and 9:30 AM ET
        return self.premarket_start <= current_time < self.market_open

    def get_active_symbols(self, max_symbols: int = 1000, exchanges: Optional[List[str]] = None) -> List[str]:
        """
        Get list of actively traded symbols suitable for premarket scanning.

        Args:
            max_symbols: Maximum number of symbols to return
            exchanges: List of exchanges to filter by

        Returns:
            List of symbol strings
        """
        if self.verbose:
            print(f"Fetching active symbols for premarket scanning...")
            if exchanges:
                print(f"Filtering by exchanges: {', '.join(exchanges)}")

        self._rate_limit_check()

        # If exchange filtering is requested, use list_assets
        if exchanges:
            return self._get_symbols_by_exchange(exchanges, max_symbols)

        try:
            # Try to get most active stocks
            if hasattr(self.client, 'get_most_actives'):
                most_actives = self.client.get_most_actives(by='volume', top=min(max_symbols, 1000))
                symbols = [stock.symbol for stock in most_actives]

                if self.verbose:
                    print(f"Found {len(symbols)} active symbols for premarket scanning")

                return symbols
            else:
                if self.verbose:
                    print("get_most_actives method not available, using fallback symbols")
        except Exception as e:
            if self.verbose:
                print(f"Error fetching active symbols: {e}")

        # Fallback to popular premarket trading symbols
        premarket_favorites = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'INTC', 'SPY', 'QQQ', 'IWM', 'BABA', 'DIS', 'PYPL',
            'ADBE', 'CRM', 'ORCL', 'UBER', 'LYFT', 'SNAP', 'ZOOM', 'DOCU',
            'SHOP', 'SQ', 'ROKU', 'PINS', 'DKNG', 'PLTR', 'GME', 'AMC',
            'COIN', 'HOOD', 'F', 'GE', 'BAC', 'JPM', 'WMT', 'KO', 'PEP',
            'XOM', 'CVX', 'JNJ', 'PFE', 'MRNA', 'BNTX', 'V', 'MA'
        ]

        if self.verbose:
            print(f"Using premarket favorites: {len(premarket_favorites[:max_symbols])} symbols")
        return premarket_favorites[:max_symbols]

    def _get_symbols_by_exchange(self, exchanges: List[str], max_symbols: int) -> List[str]:
        """Get symbols filtered by specific exchanges (mirrors alpaca_screener.py)."""
        try:
            exchanges_upper = [ex.upper() for ex in exchanges]
            safe_exchanges = ['NYSE', 'NASDAQ', 'AMEX']

            # Safety check
            for exchange in exchanges_upper:
                if exchange not in safe_exchanges:
                    if self.verbose:
                        print(f"Warning: Exchange {exchange} not in safe list. Skipping.")
                    exchanges_upper.remove(exchange)

            if not exchanges_upper:
                if self.verbose:
                    print("No valid exchanges specified. Using default symbols.")
                return self.get_active_symbols(max_symbols, exchanges=None)

            if self.verbose:
                print(f"Fetching assets from exchanges: {', '.join(exchanges_upper)}")

            assets = self.client.list_assets(status='active', asset_class='us_equity')
            filtered_symbols = []

            for asset in assets:
                if (asset.exchange.upper() in exchanges_upper and
                    asset.tradable and
                    asset.status == 'active'):

                    filtered_symbols.append(asset.symbol)
                    if len(filtered_symbols) >= max_symbols:
                        break

            if self.verbose:
                print(f"Found {len(filtered_symbols)} tradable symbols from exchanges")

            return filtered_symbols

        except Exception as e:
            if self.verbose:
                print(f"Error filtering by exchange: {e}")
            return self.get_active_symbols(max_symbols, exchanges=None)

    def find_last_market_close(self, bars_data: Dict[str, pd.DataFrame]) -> Optional[datetime]:
        """
        Find the timestamp of the last market close from the collected data.

        Args:
            bars_data: Dictionary of symbol -> DataFrame of 5-minute bars

        Returns:
            Datetime of last market close in ET, or None if not found
        """
        if not bars_data:
            return None

        # Get a representative symbol's data to find market close times
        sample_symbol = next(iter(bars_data.keys()))
        sample_bars = bars_data[sample_symbol]

        if sample_bars.empty:
            return None

        # Look for the most recent COMPLETED trading day's close
        # Work backwards from most recent to find last market close
        current_et = datetime.now(self.et_tz)
        current_time = current_et.time()

        for timestamp in reversed(sample_bars.index):
            # Convert to ET for comparison
            et_timestamp = timestamp.tz_convert(self.et_tz) if timestamp.tz else timestamp.replace(tzinfo=self.et_tz)
            bar_time = et_timestamp.time()
            bar_weekday = et_timestamp.weekday()
            bar_date = et_timestamp.date()
            current_date = current_et.date()

            # Skip weekends
            if bar_weekday >= 5:
                continue

            # If it's the same trading day and market is still open, skip to previous day
            if (bar_date == current_date and
                current_time < self.market_close and
                current_et.weekday() < 5):
                continue

            # Check if this bar is at or near market close (4:00 PM ET)
            if bar_time <= self.market_close and bar_time >= dt_time(15, 30):  # Between 3:30-4:00 PM
                # This is our last completed market close reference point
                if self.verbose:
                    print(f"Found last market close: {et_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                return et_timestamp

        if self.verbose:
            print("Could not find last market close in data")
        return None

    def collect_premarket_data(self, symbols: List[str], criteria: PremarketCriteria) -> Dict[str, Dict]:
        """
        Collect 5-minute bars for the last 7 days, then filter for premarket data.

        Args:
            symbols: List of stock symbols
            criteria: Premarket screening criteria

        Returns:
            Dictionary mapping symbols to their premarket data
        """
        if self.verbose:
            print(f"Collecting premarket data for {len(symbols)} symbols...")
            print(f"Using {criteria.feed.upper()} feed with {criteria.lookback_days}-day lookback")

        # Calculate date range (7 days back to capture last market close)
        end_time = datetime.now(self.et_tz)
        start_time = end_time - timedelta(days=criteria.lookback_days)

        if self.verbose:
            print(f"Data range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} ET")

        all_bars_data = {}
        batch_size = 50  # Process symbols in batches to avoid API limits

        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]

            if self.verbose:
                print(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")

            self._rate_limit_check()

            try:
                # Get 5-minute bars for each symbol individually
                for symbol in batch_symbols:
                    try:
                        bars = self.client.get_bars(
                            symbol,  # Individual symbol request
                            tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),  # 5-minute bars
                            start=start_time.strftime('%Y-%m-%d'),
                            end=end_time.strftime('%Y-%m-%d'),
                            limit=5000,  # Large limit to capture all data
                            feed=criteria.feed
                        )

                        if bars and len(bars) > 0:
                            # Convert to DataFrame format
                            bar_data = []
                            for bar in bars:
                                bar_dict = {
                                    'open': float(bar.o),
                                    'high': float(bar.h),
                                    'low': float(bar.l),
                                    'close': float(bar.c),
                                    'volume': int(bar.v),
                                    'timestamp': bar.t
                                }
                                bar_data.append(bar_dict)

                            if bar_data:
                                df = pd.DataFrame(bar_data)
                                df.set_index('timestamp', inplace=True)
                                # Ensure timezone-aware index
                                if df.index.tz is None:
                                    df.index = df.index.tz_localize('UTC')
                                df.index = df.index.tz_convert(self.et_tz)

                                all_bars_data[symbol] = df

                    except Exception as symbol_error:
                        if self.verbose:
                            print(f"Error fetching data for {symbol}: {symbol_error}")
                        continue

            except Exception as e:
                if self.verbose:
                    print(f"Error collecting data for batch: {e}")
                continue

        if self.verbose:
            print(f"Successfully collected raw data for {len(all_bars_data)} symbols")

        # Find the last market close time
        last_market_close = self.find_last_market_close(all_bars_data)
        if not last_market_close:
            if self.verbose:
                print("Could not determine last market close. Using fallback method.")
            # Fallback: find the most recent completed trading day at 4 PM
            current_et = datetime.now(self.et_tz)
            fallback_date = current_et.replace(hour=16, minute=0, second=0, microsecond=0)

            # If market is still open today, use yesterday
            if current_et.time() < self.market_close and current_et.weekday() < 5:
                fallback_date = fallback_date - timedelta(days=1)

            # Skip backwards to most recent weekday
            while fallback_date.weekday() >= 5:
                fallback_date -= timedelta(days=1)

            last_market_close = fallback_date
            if self.verbose:
                print(f"Using fallback market close: {last_market_close.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Filter data to only include bars since last market close
        premarket_data = {}
        for symbol, bars_df in all_bars_data.items():
            # Filter for bars after last market close
            premarket_bars = bars_df[bars_df.index > last_market_close]

            if not premarket_bars.empty:
                # Get previous close price (last bar before market close)
                previous_close_bars = bars_df[bars_df.index <= last_market_close]
                if not previous_close_bars.empty:
                    previous_close = float(previous_close_bars.iloc[-1]['close'])
                    previous_close_time = previous_close_bars.index[-1]
                else:
                    continue  # Skip if no previous close found

                premarket_data[symbol] = {
                    'premarket_bars': premarket_bars,
                    'previous_close': previous_close,
                    'previous_close_time': previous_close_time,
                    'last_market_close_ref': last_market_close
                }

        if self.verbose:
            print(f"Filtered to {len(premarket_data)} symbols with premarket activity")
            if last_market_close:
                print(f"Reference market close: {last_market_close.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        return premarket_data

    def calculate_top_gainers(self, premarket_data: Dict[str, Dict], criteria: PremarketCriteria) -> List[PremarketResult]:
        """
        Calculate top gainers from premarket data.

        Args:
            premarket_data: Dictionary of symbol -> premarket data
            criteria: Screening criteria

        Returns:
            List of PremarketResult objects sorted by gain percentage
        """
        if self.verbose:
            print(f"Calculating gains for {len(premarket_data)} symbols...")

        results = []

        for symbol, data in premarket_data.items():
            try:
                premarket_bars = data['premarket_bars']
                previous_close = data['previous_close']
                previous_close_time = data['previous_close_time']

                if premarket_bars.empty:
                    continue

                # Get current premarket metrics
                current_bar = premarket_bars.iloc[-1]
                current_price = float(current_bar['close'])
                current_timestamp = premarket_bars.index[-1]

                # Calculate gain percentage
                gain_percent = ((current_price - previous_close) / previous_close) * 100

                # Apply filters
                if criteria.min_gain_percent and gain_percent < criteria.min_gain_percent:
                    continue

                if criteria.min_price and current_price < criteria.min_price:
                    continue

                if criteria.max_price and current_price > criteria.max_price:
                    continue

                # Calculate premarket aggregated metrics
                premarket_volume = int(premarket_bars['volume'].sum())
                premarket_high = float(premarket_bars['high'].max())
                premarket_low = float(premarket_bars['low'].min())

                if criteria.min_volume and premarket_volume < criteria.min_volume:
                    continue

                # Create result
                result = PremarketResult(
                    symbol=symbol,
                    current_price=current_price,
                    previous_close=previous_close,
                    gain_percent=gain_percent,
                    premarket_volume=premarket_volume,
                    premarket_high=premarket_high,
                    premarket_low=premarket_low,
                    premarket_range=premarket_high - premarket_low,
                    current_timestamp=current_timestamp,
                    previous_close_timestamp=previous_close_time,
                    total_premarket_bars=len(premarket_bars)
                )

                results.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"Error calculating gains for {symbol}: {e}")
                continue

        # Sort by gain percentage (highest first)
        results.sort(key=lambda x: x.gain_percent, reverse=True)

        # Limit to top N if requested
        if criteria.top_gainers:
            results = results[:criteria.top_gainers]

        if self.verbose:
            print(f"Found {len(results)} qualifying premarket gainers")
            if results:
                print(f"Top gainer: {results[0].symbol} (+{results[0].gain_percent:.2f}%)")

        return results

    def scan_premarket_gainers(self, criteria: PremarketCriteria) -> List[PremarketResult]:
        """
        Main scanning method for premarket top gainers.

        Args:
            criteria: Screening criteria configuration

        Returns:
            List of PremarketResult objects sorted by gain percentage
        """
        start_time = time.time()

        if self.verbose:
            print("Starting premarket top gainers scan...")
            current_et = datetime.now(self.et_tz)
            print(f"Current time: {current_et.strftime('%H:%M:%S %Z')}")
            if self.is_premarket_hours():
                print("✓ Running during premarket hours (4:00-9:30 AM ET)")
            else:
                print("ℹ Running outside premarket hours - will find gains since last market close")

        # Get symbols to analyze
        if criteria.specific_symbols:
            symbols = criteria.specific_symbols
            if self.verbose:
                print(f"Analyzing specific symbols: {', '.join(symbols)}")
        else:
            symbols = self.get_active_symbols(criteria.max_symbols, criteria.exchanges)

        if not symbols:
            print("No symbols found for premarket scanning")
            return []

        # Collect premarket data
        premarket_data = self.collect_premarket_data(symbols, criteria)

        if not premarket_data:
            print("No premarket data collected")
            return []

        # Calculate top gainers
        results = self.calculate_top_gainers(premarket_data, criteria)

        end_time = time.time()
        if self.verbose:
            print(f"Premarket scan completed in {end_time - start_time:.2f} seconds")

        return results

    def export_to_csv(self, results: List[PremarketResult], filename: str):
        """Export premarket results to CSV file."""
        if not results:
            print("No results to export")
            return

        # Auto-create directory structure
        if not filename.startswith('/'):
            today = datetime.now().strftime('%Y-%m-%d')
            directory = f"./historical_data/{today}/premarket"
            os.makedirs(directory, exist_ok=True)
            if not filename.startswith(directory):
                filename = os.path.join(directory, os.path.basename(filename))

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'symbol', 'current_price', 'previous_close', 'gain_percent',
                'premarket_volume', 'premarket_high', 'premarket_low', 'premarket_range',
                'dollar_volume', 'total_premarket_bars', 'current_timestamp'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'symbol': result.symbol,
                    'current_price': result.current_price,
                    'previous_close': result.previous_close,
                    'gain_percent': result.gain_percent,
                    'premarket_volume': result.premarket_volume,
                    'premarket_high': result.premarket_high,
                    'premarket_low': result.premarket_low,
                    'premarket_range': result.premarket_range,
                    'dollar_volume': result.dollar_volume,
                    'total_premarket_bars': result.total_premarket_bars,
                    'current_timestamp': result.current_timestamp.isoformat()
                }
                writer.writerow(row)

        print(f"Results exported to {filename}")

    def export_to_json(self, results: List[PremarketResult], filename: str, criteria: PremarketCriteria):
        """Export premarket results to JSON file with metadata."""
        if not results:
            print("No results to export")
            return

        # Auto-create directory structure
        if not filename.startswith('/'):
            today = datetime.now().strftime('%Y-%m-%d')
            directory = f"./historical_data/{today}/premarket"
            os.makedirs(directory, exist_ok=True)
            if not filename.startswith(directory):
                filename = os.path.join(directory, os.path.basename(filename))

        export_data = {
            "scan_metadata": {
                "timestamp": datetime.now(self.et_tz).isoformat(),
                "scan_type": "premarket_top_gainers",
                "total_symbols_scanned": criteria.max_symbols,
                "results_count": len(results),
                "account": self.account,
                "environment": self.environment,
                "is_premarket_hours": self.is_premarket_hours(),
                "criteria": {
                    "min_price": criteria.min_price,
                    "max_price": criteria.max_price,
                    "min_volume": criteria.min_volume,
                    "min_gain_percent": criteria.min_gain_percent,
                    "lookback_days": criteria.lookback_days,
                    "top_gainers": criteria.top_gainers,
                    "top_losers": criteria.top_losers
                }
            },
            "results": []
        }

        for result in results:
            result_dict = {
                "symbol": result.symbol,
                "current_price": result.current_price,
                "previous_close": result.previous_close,
                "gain_percent": result.gain_percent,
                "premarket_volume": result.premarket_volume,
                "premarket_high": result.premarket_high,
                "premarket_low": result.premarket_low,
                "premarket_range": result.premarket_range,
                "dollar_volume": result.dollar_volume,
                "total_premarket_bars": result.total_premarket_bars,
                "current_timestamp": result.current_timestamp.isoformat(),
                "previous_close_timestamp": result.previous_close_timestamp.isoformat()
            }
            export_data["results"].append(result_dict)

        with open(filename, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, default=str)

        print(f"Results exported to {filename}")


def print_premarket_results(results: List[PremarketResult], criteria: PremarketCriteria):
    """Print premarket results in a formatted table."""
    if not results:
        print("No premarket gainers found matching the criteria.")
        return

    current_et = datetime.now(pytz.timezone('US/Eastern'))

    print("\nPremarket Top Gainers Scanner Results")
    print("=" * 80)
    print(f"Scan completed at: {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Premarket gainers found: {len(results)} stocks")
    if criteria.top_gainers:
        print(f"Showing top {min(criteria.top_gainers, len(results))} gainers")
    print()

    # Print header
    header = f"{'Symbol':<8} {'Current':<8} {'PrevClose':<10} {'Gain%':<8} {'PM Vol':<10} {'PM Range':<10} {'$Volume':<10}"
    print(header)
    print("-" * len(header))

    # Print results (already sorted by gain percentage)
    for result in results:
        volume_str = f"{result.premarket_volume/1e6:.1f}M" if result.premarket_volume >= 1e6 else f"{result.premarket_volume:,}"
        dollar_volume_str = f"${result.dollar_volume/1e6:.1f}M" if result.dollar_volume >= 1e6 else f"${result.dollar_volume/1e3:.1f}K"
        range_str = f"${result.premarket_range:.2f}"

        print(f"{result.symbol:<8} ${result.current_price:<7.2f} ${result.previous_close:<9.2f} "
              f"{result.gain_percent:>+6.2f}% {volume_str:<10} {range_str:<10} {dollar_volume_str:<10}")

    # Summary statistics
    if results:
        avg_gain = sum(r.gain_percent for r in results) / len(results)
        total_volume = sum(r.premarket_volume for r in results)
        print()
        print(f"Average gain: {avg_gain:.2f}%")
        print(f"Total premarket volume: {total_volume/1e6:.1f}M shares")


def parse_premarket_args() -> argparse.Namespace:
    """Parse command line arguments for the premarket scanner."""
    parser = argparse.ArgumentParser(
        description='Premarket Top Gainers Scanner - Find stocks gaining since last market close',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python premarket_top_gainers.py --verbose
  python premarket_top_gainers.py --min-gain 2.0 --top-gainers 10
  python premarket_top_gainers.py --exchanges NYSE NASDAQ --min-price 5.0 --max-price 40.0
  python premarket_top_gainers.py --symbols AAPL TSLA NVDA --verbose
  python premarket_top_gainers.py --exchanges NASDAQ AMEX --max-symbols 7000 --min-price 0.75 --max-price 40.00 --min-volume 50000 --top-gainers 20 --export-csv gainers_nasdaq_amex.csv --verbose

Note: This scanner finds stocks gaining since the last market close and can run at any time of day
        """
    )

    # Account and environment configuration
    parser.add_argument('--provider', default='alpaca', help='API provider (default: alpaca)')
    parser.add_argument('--account-name', default='Bruce', help='Account name (Bruce, Dale Wilson, Janice)')
    parser.add_argument('--account', default='paper', help='Account type (paper, live, cash)')

    # Screening criteria (matching alpaca_screener.py)
    parser.add_argument('--min-price', type=float, help='Minimum stock price (USD)')
    parser.add_argument('--max-price', type=float, help='Maximum stock price (USD)')
    parser.add_argument('--min-volume', type=int, help='Minimum volume (shares)')
    parser.add_argument('--min-gain', type=float, help='Minimum gain percentage (e.g., 1.0 for 1%%)')

    # Data source and limits
    parser.add_argument('--max-symbols', type=int, default=1000, help='Maximum symbols to analyze (default: 1000)')
    parser.add_argument('--feed', choices=['iex', 'sip'], default='sip', help='Data feed to use (default: sip)')
    parser.add_argument('--lookback-days', type=int, default=7, help='Days to look back for last market close (default: 7)')

    # Exchange filtering (NYSE, NASDAQ, and AMEX for safety)
    parser.add_argument('--exchanges', type=str, nargs='+', choices=['NYSE', 'NASDAQ', 'AMEX'],
                       help='Filter by stock exchanges (NYSE, NASDAQ, AMEX only)')

    # Specific symbol analysis
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Analyze specific symbols (e.g., AAPL MSFT TSLA)')

    # Top performers (matching alpaca_screener.py)
    parser.add_argument('--top-gainers', type=int, help='Get top N gainers (e.g., --top-gainers 10)')
    parser.add_argument('--top-losers', type=int, help='Get top N losers (e.g., --top-losers 10)')

    # Output options
    parser.add_argument('--export-csv', help='Export results to CSV file')
    parser.add_argument('--export-json', help='Export results to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    return parser.parse_args()


def main():
    """Main entry point for the premarket scanner."""
    args = parse_premarket_args()

    # Create screening criteria
    criteria = PremarketCriteria(
        min_price=args.min_price,
        max_price=args.max_price,
        min_volume=args.min_volume,
        min_gain_percent=args.min_gain,
        feed=args.feed,
        max_symbols=args.max_symbols,
        lookback_days=args.lookback_days,
        exchanges=args.exchanges,
        specific_symbols=[symbol.upper() for symbol in args.symbols] if args.symbols else None,
        top_gainers=args.top_gainers,
        top_losers=args.top_losers
    )

    # Initialize scanner
    scanner = PremarketTopGainersScanner(
        provider=args.provider,
        account=args.account_name,
        environment=args.account,
        verbose=args.verbose
    )

    try:
        # Run premarket scan
        results = scanner.scan_premarket_gainers(criteria)

        # Display results
        print_premarket_results(results, criteria)

        # Export results if requested
        if args.export_csv:
            scanner.export_to_csv(results, args.export_csv)

        if args.export_json:
            scanner.export_to_json(results, args.export_json, criteria)

    except KeyboardInterrupt:
        print("\nPremarket scan interrupted by user")
    except Exception as e:
        print(f"Error during premarket scanning: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()