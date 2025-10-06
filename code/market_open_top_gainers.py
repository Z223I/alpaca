#!/usr/bin/env python3
"""
Market Open Top Gainers Scanner

A specialized scanner that identifies top gaining stocks by comparing current prices
to the market open. Uses 1-minute candles and can run during market hours or after
market close to find stocks that have gained since the market open.

Strategy:
- Collect 1-minute bars for the last 3 days
- Identify the market open (9:30 AM ET on current or most recent trading day)
- Filter data from market open to current time (during market hours) or market close (after hours)
- Calculate gains from market open to current/close price
- Rank stocks by highest percentage gains

Can be run during market hours (9:30 AM-4:00 PM ET) for real-time market analysis,
or after market close to analyze full-day performance since market open.

Mirrors the structure of premarket_top_gainers.py for consistency.
"""

"""
Usage:
python code/market_open_top_gainers.py  --exchanges NASDAQ AMEX  --max-symbols 7000  --min-price 0.75  --max-price 40.00  --min-volume 50000 --top-gainers 20 --export-csv gainers_nasdaq_amex.csv --verbose
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
class MarketOpenCriteria:
    """Configuration class for market open top gainers screening."""
    # Minimum thresholds
    min_price: Optional[float] = 0.75
    max_price: Optional[float] = None
    min_volume: Optional[int] = 50_000  # Volume threshold during market hours
    min_gain_percent: Optional[float] = 1.0  # Minimum gain to be considered

    # Data collection settings
    feed: str = "sip"  # iex, sip, or other feed names
    max_symbols: int = 1000  # Smaller set for market efficiency
    lookback_days: int = 3  # Days to look back for finding market open

    # Exchange filtering (NYSE, NASDAQ, and AMEX only for safety)
    exchanges: Optional[List[str]] = None
    specific_symbols: Optional[List[str]] = None

    # Result limiting (match alpaca_screener.py interface)
    top_gainers: Optional[int] = 20  # Top N gainers to return
    top_losers: Optional[int] = None  # Top N losers (not used but matches interface)


@dataclass
class MarketOpenResult:
    """Data class for market open top gainers results."""
    symbol: str
    current_price: float
    market_open_price: float
    gain_percent: float
    market_volume: int
    market_high: float
    market_low: float
    market_range: float
    current_timestamp: datetime
    market_open_timestamp: datetime

    # Additional metrics
    dollar_volume: float = 0.0
    total_market_bars: int = 0

    def __post_init__(self):
        self.dollar_volume = self.market_volume * self.current_price
        self.market_range = self.market_high - self.market_low


class MarketOpenTopGainersScanner:
    """Main market open top gainers scanner class."""

    def __init__(self, provider: str = "alpaca", account: str = "Bruce", environment: str = "paper",
                 verbose: bool = False):
        """
        Initialize the market open scanner.

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
            print(f"Initialized Market Open Scanner - Account: {account}, Environment: {environment}")
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

    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM ET)."""
        current_et = datetime.now(self.et_tz)
        current_time = current_et.time()
        current_weekday = current_et.weekday()  # Monday = 0, Sunday = 6

        # Only Monday-Friday
        if current_weekday >= 5:
            return False

        # Check if between 9:30 AM and 4:00 PM ET
        return self.market_open <= current_time <= self.market_close

    def get_active_symbols(self, max_symbols: int = 1000, exchanges: Optional[List[str]] = None) -> List[str]:
        """
        Get list of actively traded symbols suitable for market scanning.

        Args:
            max_symbols: Maximum number of symbols to return
            exchanges: List of exchanges to filter by

        Returns:
            List of symbol strings
        """
        if self.verbose:
            print(f"Fetching active symbols for market scanning...")
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
                    print(f"Found {len(symbols)} active symbols for market scanning")

                return symbols
            else:
                if self.verbose:
                    print("get_most_actives method not available, using fallback symbols")
        except Exception as e:
            if self.verbose:
                print(f"Error fetching active symbols: {e}")

        # Fallback to popular market trading symbols
        market_favorites = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'INTC', 'SPY', 'QQQ', 'IWM', 'BABA', 'DIS', 'PYPL',
            'ADBE', 'CRM', 'ORCL', 'UBER', 'LYFT', 'SNAP', 'ZOOM', 'DOCU',
            'SHOP', 'SQ', 'ROKU', 'PINS', 'DKNG', 'PLTR', 'GME', 'AMC',
            'COIN', 'HOOD', 'F', 'GE', 'BAC', 'JPM', 'WMT', 'KO', 'PEP',
            'XOM', 'CVX', 'JNJ', 'PFE', 'MRNA', 'BNTX', 'V', 'MA'
        ]

        if self.verbose:
            print(f"Using market favorites: {len(market_favorites[:max_symbols])} symbols")
        return market_favorites[:max_symbols]

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

    def find_market_open_data(self, bars_data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """
        Find the actual market open bar (9:30 AM ET) from the collected data.

        Args:
            bars_data: Dictionary of symbol -> DataFrame of 1-minute bars

        Returns:
            Dictionary with open_timestamp and open_bars_dict, or None if not found
        """
        if not bars_data:
            return None

        # Get a representative symbol's data to find market open times
        sample_symbol = next(iter(bars_data.keys()))
        sample_bars = bars_data[sample_symbol]

        if sample_bars.empty:
            return None

        current_et = datetime.now(self.et_tz)
        current_time = current_et.time()
        current_date = current_et.date()

        # Determine which trading day's market open to use
        target_date = current_date

        # If it's before market open today, use yesterday's open
        if (current_et.weekday() < 5 and current_time < self.market_open):
            target_date = current_date - timedelta(days=1)

        # Skip back to most recent weekday
        while target_date.weekday() >= 5:  # Skip weekends
            target_date -= timedelta(days=1)

        if self.verbose:
            print(f"Looking for market open on: {target_date}")

        # Look for the 9:30 AM ET bar on the target date across all symbols
        target_open_time = self.et_tz.localize(
            datetime.combine(target_date, self.market_open)
        )

        # Find bars closest to 9:30 AM ET (within 5 minutes)
        open_bars_dict = {}
        found_any_open = False

        for symbol, bars_df in bars_data.items():
            if bars_df.empty:
                continue

            # Filter to target date
            target_date_bars = bars_df[bars_df.index.date == target_date]

            if target_date_bars.empty:
                continue

            # Find the bar closest to 9:30 AM ET
            time_diffs = abs(target_date_bars.index - target_open_time)
            min_diff_idx = time_diffs.argmin()
            closest_idx = target_date_bars.index[min_diff_idx]
            closest_bar = target_date_bars.loc[closest_idx]
            closest_time = closest_idx

            # Verify it's reasonably close to market open (within 5 minutes)
            time_diff_minutes = abs((closest_time - target_open_time).total_seconds()) / 60

            if time_diff_minutes <= 5:  # Within 5 minutes of 9:30 AM ET
                open_bars_dict[symbol] = {
                    'open_price': float(closest_bar['open']),
                    'open_time': closest_time,
                    'bar_data': closest_bar
                }
                found_any_open = True

                if self.verbose and symbol == sample_symbol:
                    print(f"Found market open bar for {symbol}: {closest_time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                          f"(${closest_bar['open']:.2f})")

        if found_any_open:
            return {
                'open_timestamp': target_open_time,
                'open_bars_dict': open_bars_dict,
                'target_date': target_date
            }

        if self.verbose:
            print("Could not find market open bars in data")
        return None

    def collect_market_data(self, symbols: List[str], criteria: MarketOpenCriteria) -> Dict[str, Dict]:
        """
        Collect 1-minute bars for the last 3 days, then filter for market data.

        Args:
            symbols: List of stock symbols
            criteria: Market open screening criteria

        Returns:
            Dictionary mapping symbols to their market data
        """
        if self.verbose:
            print(f"Collecting market data for {len(symbols)} symbols...")
            print(f"Using {criteria.feed.upper()} feed with {criteria.lookback_days}-day lookback")

        # Calculate date range (3 days back to capture market open)
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
                # Get 1-minute bars for each symbol individually
                for symbol in batch_symbols:
                    try:
                        bars = self.client.get_bars(
                            symbol,  # Individual symbol request
                            tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Minute),  # 1-minute bars
                            start=start_time.strftime('%Y-%m-%d'),
                            end=end_time.strftime('%Y-%m-%d'),
                            limit=10000,  # Larger limit for 1-minute bars
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

        # Find the actual market open data
        market_open_data = self.find_market_open_data(all_bars_data)
        if not market_open_data:
            if self.verbose:
                print("Could not determine market open. Using fallback method.")
            # Fallback: find the current or most recent trading day at 9:30 AM
            current_et = datetime.now(self.et_tz)
            target_date = current_et.date()

            # If it's before market open today, use yesterday's open
            if (current_et.weekday() < 5 and current_et.time() < self.market_open):
                target_date = current_et.date() - timedelta(days=1)

            # Skip backwards to most recent weekday
            while target_date.weekday() >= 5:
                target_date -= timedelta(days=1)

            fallback_open_time = self.et_tz.localize(
                datetime.combine(target_date, self.market_open)
            )

            if self.verbose:
                print(f"Using fallback market open: {fallback_open_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            # Create fallback open data using first available bars after fallback time
            market_open_data = {
                'open_timestamp': fallback_open_time,
                'open_bars_dict': {},
                'target_date': target_date
            }

            for symbol, bars_df in all_bars_data.items():
                fallback_open_bars = bars_df[bars_df.index >= fallback_open_time]
                if not fallback_open_bars.empty:
                    first_bar = fallback_open_bars.iloc[0]
                    market_open_data['open_bars_dict'][symbol] = {
                        'open_price': float(first_bar['open']),
                        'open_time': fallback_open_bars.index[0],
                        'bar_data': first_bar
                    }

        open_timestamp = market_open_data['open_timestamp']
        open_bars_dict = market_open_data['open_bars_dict']

        # Determine end time for data filtering
        current_et = datetime.now(self.et_tz)
        if self.is_market_hours():
            # During market hours, use current time
            end_filter_time = current_et
            if self.verbose:
                print("Running during market hours - calculating gains to current time")
        else:
            # After market hours, use market close
            target_date = market_open_data['target_date']
            end_filter_time = self.et_tz.localize(
                datetime.combine(target_date, self.market_close)
            )
            if self.verbose:
                print("Running after market hours - calculating gains to market close")

        # Filter data from market open to end time
        market_data = {}
        for symbol, bars_df in all_bars_data.items():
            # Skip symbols without open data
            if symbol not in open_bars_dict:
                continue

            # Filter for bars from market open to end time
            market_bars = bars_df[
                (bars_df.index >= open_timestamp) &
                (bars_df.index <= end_filter_time)
            ]

            if not market_bars.empty:
                open_info = open_bars_dict[symbol]
                market_data[symbol] = {
                    'market_bars': market_bars,
                    'market_open_price': open_info['open_price'],
                    'market_open_time': open_info['open_time'],
                    'market_open_ref': open_timestamp,
                    'end_filter_time': end_filter_time
                }

        if self.verbose:
            print(f"Filtered to {len(market_data)} symbols with market activity")
            if open_timestamp:
                print(f"Reference market open: {open_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"End filter time: {end_filter_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        return market_data

    def calculate_top_gainers(self, market_data: Dict[str, Dict], criteria: MarketOpenCriteria) -> List[MarketOpenResult]:
        """
        Calculate top gainers from market data.

        Args:
            market_data: Dictionary of symbol -> market data
            criteria: Screening criteria

        Returns:
            List of MarketOpenResult objects sorted by gain percentage
        """
        if self.verbose:
            print(f"Calculating gains for {len(market_data)} symbols...")

        results = []

        for symbol, data in market_data.items():
            try:
                market_bars = data['market_bars']
                market_open_price = data['market_open_price']
                market_open_time = data['market_open_time']

                if market_bars.empty:
                    continue

                # Get current market metrics
                current_bar = market_bars.iloc[-1]
                current_price = float(current_bar['close'])
                current_timestamp = market_bars.index[-1]

                # Calculate gain percentage
                gain_percent = ((current_price - market_open_price) / market_open_price) * 100

                # Apply filters
                if criteria.min_gain_percent and gain_percent < criteria.min_gain_percent:
                    continue

                if criteria.min_price and current_price < criteria.min_price:
                    continue

                if criteria.max_price and current_price > criteria.max_price:
                    continue

                # Calculate market aggregated metrics
                market_volume = int(market_bars['volume'].sum())
                market_high = float(market_bars['high'].max())
                market_low = float(market_bars['low'].min())

                if criteria.min_volume and market_volume < criteria.min_volume:
                    continue

                # Create result
                result = MarketOpenResult(
                    symbol=symbol,
                    current_price=current_price,
                    market_open_price=market_open_price,
                    gain_percent=gain_percent,
                    market_volume=market_volume,
                    market_high=market_high,
                    market_low=market_low,
                    market_range=market_high - market_low,
                    current_timestamp=current_timestamp,
                    market_open_timestamp=market_open_time,
                    total_market_bars=len(market_bars)
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
            print(f"Found {len(results)} qualifying market gainers")
            if results:
                print(f"Top gainer: {results[0].symbol} (+{results[0].gain_percent:.2f}%)")

        return results

    def scan_market_gainers(self, criteria: MarketOpenCriteria) -> List[MarketOpenResult]:
        """
        Main scanning method for market open top gainers.

        Args:
            criteria: Screening criteria configuration

        Returns:
            List of MarketOpenResult objects sorted by gain percentage
        """
        start_time = time.time()

        if self.verbose:
            print("Starting market open top gainers scan...")
            current_et = datetime.now(self.et_tz)
            print(f"Current time: {current_et.strftime('%H:%M:%S %Z')}")
            if self.is_market_hours():
                print("✓ Running during market hours (9:30 AM-4:00 PM ET)")
            else:
                print("ℹ Running outside market hours - will find gains from market open to close")

        # Get symbols to analyze
        if criteria.specific_symbols:
            symbols = criteria.specific_symbols
            if self.verbose:
                print(f"Analyzing specific symbols: {', '.join(symbols)}")
        else:
            symbols = self.get_active_symbols(criteria.max_symbols, criteria.exchanges)

        if not symbols:
            print("No symbols found for market scanning")
            return []

        # Collect market data
        market_data = self.collect_market_data(symbols, criteria)

        if not market_data:
            print("No market data collected")
            return []

        # Calculate top gainers
        results = self.calculate_top_gainers(market_data, criteria)

        end_time = time.time()
        if self.verbose:
            print(f"Market scan completed in {end_time - start_time:.2f} seconds")

        return results

    def export_to_csv(self, results: List[MarketOpenResult], filename: str):
        """Export market results to CSV file."""
        if not results:
            print("No results to export")
            return

        # Auto-create directory structure
        if not filename.startswith('/'):
            today = datetime.now().strftime('%Y-%m-%d')
            directory = f"./historical_data/{today}/market"
            os.makedirs(directory, exist_ok=True)
            if not filename.startswith(directory):
                filename = os.path.join(directory, os.path.basename(filename))

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'symbol', 'current_price', 'market_open_price', 'gain_percent',
                'market_volume', 'market_high', 'market_low', 'market_range',
                'dollar_volume', 'total_market_bars', 'current_timestamp'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'symbol': result.symbol,
                    'current_price': round(result.current_price, 2),
                    'market_open_price': round(result.market_open_price, 2),
                    'gain_percent': round(result.gain_percent, 2),
                    'market_volume': result.market_volume,
                    'market_high': round(result.market_high, 2),
                    'market_low': round(result.market_low, 2),
                    'market_range': round(result.market_range, 2),
                    'dollar_volume': round(result.dollar_volume, 2),
                    'total_market_bars': result.total_market_bars,
                    'current_timestamp': result.current_timestamp.isoformat()
                }
                writer.writerow(row)

        print(f"Results exported to {filename}")

    def export_to_json(self, results: List[MarketOpenResult], filename: str, criteria: MarketOpenCriteria):
        """Export market results to JSON file with metadata."""
        if not results:
            print("No results to export")
            return

        # Auto-create directory structure
        if not filename.startswith('/'):
            today = datetime.now().strftime('%Y-%m-%d')
            directory = f"./historical_data/{today}/market"
            os.makedirs(directory, exist_ok=True)
            if not filename.startswith(directory):
                filename = os.path.join(directory, os.path.basename(filename))

        export_data = {
            "scan_metadata": {
                "timestamp": datetime.now(self.et_tz).isoformat(),
                "scan_type": "market_open_top_gainers",
                "total_symbols_scanned": criteria.max_symbols,
                "results_count": len(results),
                "account": self.account,
                "environment": self.environment,
                "is_market_hours": self.is_market_hours(),
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
                "current_price": round(result.current_price, 2),
                "market_open_price": round(result.market_open_price, 2),
                "gain_percent": round(result.gain_percent, 2),
                "market_volume": result.market_volume,
                "market_high": round(result.market_high, 2),
                "market_low": round(result.market_low, 2),
                "market_range": round(result.market_range, 2),
                "dollar_volume": round(result.dollar_volume, 2),
                "total_market_bars": result.total_market_bars,
                "current_timestamp": result.current_timestamp.isoformat(),
                "market_open_timestamp": result.market_open_timestamp.isoformat()
            }
            export_data["results"].append(result_dict)

        with open(filename, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, default=str)

        print(f"Results exported to {filename}")


def print_market_results(results: List[MarketOpenResult], criteria: MarketOpenCriteria):
    """Print market results in a formatted table."""
    if not results:
        print("No market gainers found matching the criteria.")
        return

    current_et = datetime.now(pytz.timezone('US/Eastern'))

    print("\nMarket Open Top Gainers Scanner Results")
    print("=" * 80)
    print(f"Scan completed at: {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Market gainers found: {len(results)} stocks")
    if criteria.top_gainers:
        print(f"Showing top {min(criteria.top_gainers, len(results))} gainers")
    print()

    # Print header
    header = f"{'Symbol':<8} {'Current':<8} {'MarketOpen':<11} {'Gain%':<8} {'Mkt Vol':<10} {'Mkt Range':<10} {'$Volume':<10}"
    print(header)
    print("-" * len(header))

    # Print results (already sorted by gain percentage)
    for result in results:
        volume_str = f"{result.market_volume/1e6:.1f}M" if result.market_volume >= 1e6 else f"{result.market_volume:,}"
        dollar_volume_str = f"${result.dollar_volume/1e6:.1f}M" if result.dollar_volume >= 1e6 else f"${result.dollar_volume/1e3:.1f}K"
        range_str = f"${result.market_range:.2f}"

        print(f"{result.symbol:<8} ${result.current_price:<7.2f} ${result.market_open_price:<10.2f} "
              f"{result.gain_percent:>+6.2f}% {volume_str:<10} {range_str:<10} {dollar_volume_str:<10}")

    # Summary statistics
    if results:
        avg_gain = sum(r.gain_percent for r in results) / len(results)
        total_volume = sum(r.market_volume for r in results)
        print()
        print(f"Average gain: {avg_gain:.2f}%")
        print(f"Total market volume: {total_volume/1e6:.1f}M shares")


def parse_market_args() -> argparse.Namespace:
    """Parse command line arguments for the market scanner."""
    parser = argparse.ArgumentParser(
        description='Market Open Top Gainers Scanner - Find stocks gaining since market open',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python market_open_top_gainers.py --verbose
  python market_open_top_gainers.py --min-gain 2.0 --top-gainers 10
  python market_open_top_gainers.py --exchanges NYSE NASDAQ --min-price 5.0 --max-price 40.0
  python market_open_top_gainers.py --symbols AAPL TSLA NVDA --verbose
  python market_open_top_gainers.py --exchanges NASDAQ AMEX --max-symbols 7000 --min-price 0.75 --max-price 40.00 --min-volume 50000 --top-gainers 20 --export-csv gainers_nasdaq_amex.csv --verbose

Note: This scanner finds stocks gaining since market open and can run during market hours or after close
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
    parser.add_argument('--lookback-days', type=int, default=3, help='Days to look back for market open (default: 3)')

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
    """Main entry point for the market scanner."""
    args = parse_market_args()

    # Create screening criteria
    criteria = MarketOpenCriteria(
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
    scanner = MarketOpenTopGainersScanner(
        provider=args.provider,
        account=args.account_name,
        environment=args.account,
        verbose=args.verbose
    )

    try:
        # Run market scan
        results = scanner.scan_market_gainers(criteria)

        # Display results
        print_market_results(results, criteria)

        # Export results if requested
        if args.export_csv:
            scanner.export_to_csv(results, args.export_csv)

        if args.export_json:
            scanner.export_to_json(results, args.export_json, criteria)

    except KeyboardInterrupt:
        print("\nMarket scan interrupted by user")
    except Exception as e:
        print(f"Error during market scanning: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()