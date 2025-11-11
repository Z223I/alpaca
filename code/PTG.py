#!/usr/bin/env python3
"""
PTG.py - Pre-Market Top Gainers Scanner (Standalone)

A completely standalone script that uses alpaca-py to scan for pre-market top gainers.
Loads credentials from .env file and requires no external project dependencies.

Features:
    - Fetches 5-minute bars from Alpaca's SIP feed
    - Calculates gains since previous market close (4:00 PM ET)
    - Supports filtering by price range and volume
    - Can scan specific symbols or fetch from exchanges (NASDAQ, AMEX, NYSE)
    - Uses alpaca-py (modern Python SDK)

Prerequisites:
    - .env file with ALPACA_API_KEY and ALPACA_SECRET_KEY
    - conda alpaca environment activated

Usage Examples:

    # Scan default symbols (57 popular stocks)
    conda run -n alpaca python code/PTG.py

    # Show top 20 gainers from default symbols
    conda run -n alpaca python code/PTG.py --top 20

    # Scan specific symbols
    conda run -n alpaca python code/PTG.py --symbols AAPL TSLA NVDA MSFT --top 5

    # Scan NASDAQ and AMEX exchanges (up to 100 symbols)
    conda run -n alpaca python code/PTG.py --exchanges NASDAQ AMEX --max-symbols 100 --top 20

    # Filter by price range ($1-$20) and minimum volume
    conda run -n alpaca python code/PTG.py --exchanges NASDAQ --max-symbols 200 \\
        --min-price 1.0 --max-price 20.0 --min-volume 10000 --top 15

    # Scan with all filters
    conda run -n alpaca python code/PTG.py --exchanges NASDAQ AMEX --max-symbols 500 --top 25 --min-price 0.75 --max-price 40.0 --min-volume 250000

    # Use live account instead of paper
    conda run -n alpaca python code/PTG.py --live --exchanges NASDAQ --top 10

    # Export results to CSV
    conda run -n alpaca python code/PTG.py --exchanges NASDAQ AMEX \\
        --max-symbols 100 --top 20 --export-csv top_gainers.csv

Output:
    Displays a formatted table with:
    - Symbol
    - Current price
    - Previous close price
    - Gain percentage
    - Volume
    - High/Low prices
    - Summary statistics (average gain, total volume)
"""

import os
import sys
import argparse
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz

# Remove the code directory from sys.path to prevent local alpaca.py from shadowing alpaca package
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

# Import alpaca-py BEFORE loading .env (which might add paths)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass

# Now import dotenv
from dotenv import load_dotenv


class PremarketScanner:
    """Standalone pre-market top gainers scanner."""

    def __init__(self, use_paper: bool = True, verbose: bool = False):
        """Initialize the scanner with credentials from .env file."""
        # Load environment variables
        load_dotenv()

        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.verbose = verbose

        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")

        # Initialize Alpaca clients
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=use_paper)

        # Eastern timezone for market hours
        self.et_tz = pytz.timezone('US/Eastern')

        if self.verbose:
            print(f"Initialized PTG Scanner (paper={use_paper})")

    def get_default_symbols(self) -> List[str]:
        """Get a default list of active symbols for scanning."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'INTC', 'SPY', 'QQQ', 'IWM', 'BABA', 'DIS', 'PYPL',
            'ADBE', 'CRM', 'ORCL', 'UBER', 'LYFT', 'SNAP', 'ZM', 'DOCU',
            'SHOP', 'SQ', 'ROKU', 'PINS', 'DKNG', 'PLTR', 'GME', 'AMC',
            'COIN', 'HOOD', 'F', 'GE', 'BAC', 'JPM', 'WMT', 'KO', 'PEP',
            'XOM', 'CVX', 'JNJ', 'PFE', 'MRNA', 'BNTX', 'V', 'MA',
            'NIO', 'RIVN', 'LCID', 'SOFI', 'AFRM', 'RBLX', 'U', 'ABNB'
        ]

    def get_symbols_from_exchanges(
        self,
        exchanges: List[str],
        max_symbols: int = 1000,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[str]:
        """
        Get symbols from specific exchanges (NASDAQ, AMEX).

        Args:
            exchanges: List of exchange names (e.g., ['NASDAQ', 'AMEX'])
            max_symbols: Maximum number of symbols to return
            min_price: Minimum price filter
            max_price: Maximum price filter

        Returns:
            List of symbols from the specified exchanges
        """
        if self.verbose:
            print(f"Fetching symbols from exchanges: {', '.join(exchanges)}")

        try:
            # Create request to get all active US equities
            request = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY
            )

            # Get all assets
            assets = self.trading_client.get_all_assets(filter=request)

            # Filter by exchange
            exchanges_upper = [ex.upper() for ex in exchanges]
            filtered_symbols = []

            for asset in assets:
                # Get exchange value
                asset_exchange = asset.exchange.value if hasattr(asset.exchange, 'value') else str(asset.exchange)

                # Check if asset matches criteria
                if (asset_exchange.upper() in exchanges_upper and
                    asset.tradable and
                    asset.status == AssetStatus.ACTIVE):

                    filtered_symbols.append(asset.symbol)

                    if len(filtered_symbols) >= max_symbols:
                        break

            if self.verbose:
                print(f"Found {len(filtered_symbols)} tradable symbols from {', '.join(exchanges)}")
            return filtered_symbols

        except Exception as e:
            if self.verbose:
                print(f"Error fetching exchange symbols: {e}")
                print("Falling back to default symbols...")
            return self.get_default_symbols()

    def get_bars(self, symbols: List[str], days_back: int = 2) -> Dict[str, List]:
        """
        Fetch 5-minute bars for the given symbols.

        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back

        Returns:
            Dictionary mapping symbols to their bar data
        """
        end_time = datetime.now(self.et_tz)
        start_time = end_time - timedelta(days=days_back)

        if self.verbose:
            print(f"Fetching bars from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")

        # Request 5-minute bars
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time,
            feed='sip'
        )

        # Get bars
        bars_response = self.data_client.get_stock_bars(request)

        return bars_response.data

    def find_previous_close(self, bars: List, current_time: datetime) -> Optional[float]:
        """
        Find the previous market close price (4:00 PM ET).

        Args:
            bars: List of bar data
            current_time: Current time

        Returns:
            Previous close price or None
        """
        if not bars:
            return None

        # Look for the most recent market close (4:00 PM ET)
        market_close_hour = 16  # 4:00 PM

        # Find bars from yesterday or before
        yesterday = current_time.date() - timedelta(days=1)

        # Skip back to most recent weekday
        while yesterday.weekday() >= 5:  # Saturday=5, Sunday=6
            yesterday -= timedelta(days=1)

        # Look for close price around 4:00 PM on the most recent trading day
        for bar in reversed(bars):
            bar_time = bar.timestamp.astimezone(self.et_tz)
            bar_date = bar_time.date()

            # Check if this is from a previous trading day
            if bar_date <= yesterday:
                # Look for bars around market close (3:55 PM to 4:05 PM)
                if 15 <= bar_time.hour <= 16:
                    return float(bar.close)

        # Fallback: return close of oldest bar
        if bars:
            return float(bars[0].close)

        return None

    def calculate_gainers(
        self,
        bars_data: Dict[str, List],
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_volume: Optional[int] = None
    ) -> List[Dict]:
        """
        Calculate top gainers from bar data.

        Args:
            bars_data: Dictionary of symbol -> bars
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_volume: Minimum volume filter

        Returns:
            List of gainer dictionaries
        """
        current_time = datetime.now(self.et_tz)
        gainers = []

        for symbol, bars in bars_data.items():
            if not bars or len(bars) < 2:
                continue

            # Get current price (latest bar)
            current_bar = bars[-1]
            current_price = float(current_bar.close)
            current_volume = int(current_bar.volume)

            # Find previous close
            prev_close = self.find_previous_close(bars, current_time)

            if prev_close is None or prev_close == 0:
                continue

            # Calculate gain percentage
            gain_pct = ((current_price - prev_close) / prev_close) * 100

            # Apply filters
            if min_price and current_price < min_price:
                continue
            if max_price and current_price > max_price:
                continue
            if min_volume and current_volume < min_volume:
                continue

            # Calculate additional metrics
            highs = [float(b.high) for b in bars[-20:]]  # Last 20 bars
            lows = [float(b.low) for b in bars[-20:]]
            volumes = [int(b.volume) for b in bars[-10:]]  # Last 10 bars

            gainer = {
                'symbol': symbol,
                'current_price': current_price,
                'prev_close': prev_close,
                'gain_pct': gain_pct,
                'volume': current_volume,
                'avg_volume': sum(volumes) // len(volumes) if volumes else 0,
                'high': max(highs) if highs else current_price,
                'low': min(lows) if lows else current_price,
                'timestamp': current_bar.timestamp
            }

            gainers.append(gainer)

        # Sort by gain percentage (descending)
        gainers.sort(key=lambda x: x['gain_pct'], reverse=True)

        return gainers

    def print_results(self, gainers: List[Dict], top_n: int):
        """Print formatted results."""
        if not gainers:
            print("\nNo gainers found matching criteria.")
            return

        print(f"\n{'='*80}")
        print(f"Pre-Market Top {top_n} Gainers")
        print(f"{'='*80}")
        print(f"Scan time: {datetime.now(self.et_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Total matches: {len(gainers)}")
        print()

        # Print header
        print(f"{'Symbol':<8} {'Price':<10} {'Prev':<10} {'Gain%':<10} {'Volume':<12} {'High':<10} {'Low':<10}")
        print("-" * 80)

        # Print top N gainers
        for gainer in gainers[:top_n]:
            volume_str = f"{gainer['volume']:,}"
            print(
                f"{gainer['symbol']:<8} "
                f"${gainer['current_price']:<9.2f} "
                f"${gainer['prev_close']:<9.2f} "
                f"{gainer['gain_pct']:>+8.2f}% "
                f"{volume_str:<12} "
                f"${gainer['high']:<9.2f} "
                f"${gainer['low']:<9.2f}"
            )

        # Summary stats
        if gainers:
            avg_gain = sum(g['gain_pct'] for g in gainers[:top_n]) / min(top_n, len(gainers))
            total_volume = sum(g['volume'] for g in gainers[:top_n])
            print()
            print(f"Average gain (top {min(top_n, len(gainers))}): {avg_gain:+.2f}%")
            print(f"Total volume: {total_volume:,}")

    def scan(
        self,
        symbols: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None,
        max_symbols: int = 1000,
        top_n: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_volume: Optional[int] = None
    ):
        """
        Run the pre-market scan.

        Args:
            symbols: List of symbols to scan (uses defaults if None)
            exchanges: List of exchanges to scan (e.g., ['NASDAQ', 'AMEX'])
            max_symbols: Maximum symbols to fetch from exchanges
            top_n: Number of top gainers to return
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_volume: Minimum volume filter
        """
        # Determine which symbols to scan
        if exchanges:
            symbols = self.get_symbols_from_exchanges(
                exchanges,
                max_symbols=max_symbols,
                min_price=min_price,
                max_price=max_price
            )
            if self.verbose:
                print(f"Scanning {len(symbols)} symbols from exchanges...")
        elif symbols is None:
            symbols = self.get_default_symbols()
            if self.verbose:
                print(f"Scanning {len(symbols)} default symbols...")
        else:
            if self.verbose:
                print(f"Scanning {len(symbols)} symbols...")

        if not symbols:
            if self.verbose:
                print("No symbols to scan")
            return

        # Get bar data
        bars_data = self.get_bars(symbols, days_back=2)
        if self.verbose:
            print(f"Retrieved data for {len(bars_data)} symbols")

        # Calculate gainers
        gainers = self.calculate_gainers(
            bars_data,
            min_price=min_price,
            max_price=max_price,
            min_volume=min_volume
        )

        # Print results
        self.print_results(gainers, top_n)

        return gainers

    def export_to_csv(self, gainers: List[Dict], filename: str):
        """
        Export gainers to CSV file (matches premarket_top_gainers.py format).

        Args:
            gainers: List of gainer dictionaries
            filename: CSV filename
        """
        if not gainers:
            print("No results to export")
            return

        # Auto-create directory structure (same as premarket_top_gainers.py)
        if not filename.startswith('/'):
            today = datetime.now().strftime('%Y-%m-%d')
            directory = f"./historical_data/{today}/premarket"
            os.makedirs(directory, exist_ok=True)
            if not filename.startswith(directory):
                filename = os.path.join(directory, os.path.basename(filename))

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'symbol', 'current_price', 'previous_close', 'gain_percent',
                'volume', 'avg_volume', 'high', 'low', 'price_range',
                'dollar_volume', 'timestamp'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for gainer in gainers:
                row = {
                    'symbol': gainer['symbol'],
                    'current_price': gainer['current_price'],
                    'previous_close': gainer['prev_close'],
                    'gain_percent': gainer['gain_pct'],
                    'volume': gainer['volume'],
                    'avg_volume': gainer['avg_volume'],
                    'high': gainer['high'],
                    'low': gainer['low'],
                    'price_range': gainer['high'] - gainer['low'],
                    'dollar_volume': gainer['volume'] * gainer['current_price'],
                    'timestamp': gainer['timestamp'].isoformat()
                }
                writer.writerow(row)

        print(f"Results exported to {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Pre-Market Top Gainers Scanner (Standalone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code/PTG.py
  python code/PTG.py --top 20
  python code/PTG.py --top 10 --min-price 5.0 --max-price 50.0
  python code/PTG.py --symbols AAPL TSLA NVDA --top 5
  python code/PTG.py --exchanges NASDAQ AMEX --max-symbols 100 --top 20
  python code/PTG.py --exchanges NASDAQ --min-price 1.0 --max-price 10.0 --top 15
  python code/PTG.py --exchanges NASDAQ AMEX --max-symbols 100 --export-csv gainers.csv
        """
    )

    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Symbols to scan (default: built-in list)')
    parser.add_argument('--exchanges', type=str, nargs='+',
                       choices=['NASDAQ', 'AMEX', 'NYSE'],
                       help='Exchanges to scan (e.g., NASDAQ AMEX)')
    parser.add_argument('--max-symbols', type=int, default=1000,
                       help='Maximum symbols to fetch from exchanges (default: 1000)')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top gainers to show (default: 10)')
    parser.add_argument('--min-price', type=float,
                       help='Minimum stock price filter')
    parser.add_argument('--max-price', type=float,
                       help='Maximum stock price filter')
    parser.add_argument('--min-volume', type=int,
                       help='Minimum volume filter')
    parser.add_argument('--export-csv', type=str,
                       help='Export results to CSV file')
    parser.add_argument('--paper', action='store_true', default=True,
                       help='Use paper trading account (default: True)')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading account')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Determine paper vs live
    use_paper = not args.live

    try:
        # Create scanner
        scanner = PremarketScanner(use_paper=use_paper, verbose=args.verbose)

        # Run scan
        gainers = scanner.scan(
            symbols=args.symbols,
            exchanges=args.exchanges,
            max_symbols=args.max_symbols,
            top_n=args.top,
            min_price=args.min_price,
            max_price=args.max_price,
            min_volume=args.min_volume
        )

        # Export to CSV if requested
        if args.export_csv and gainers:
            scanner.export_to_csv(gainers, args.export_csv)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
