#!/usr/bin/env python3
"""
Alpaca Stock Screener

A comprehensive stock screener using Alpaca Trading API v2 with volume surge detection
and traditional screening metrics. Built to integrate with existing infrastructure.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import alpaca_trade_api as tradeapi

# Add the atoms directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'atoms'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from atoms.api.init_alpaca_client import init_alpaca_client


@dataclass
class ScreeningCriteria:
    """Configuration class for stock screening criteria."""
    # Price filters
    min_price: Optional[float] = 0.75
    max_price: Optional[float] = None
    
    # Volume filters  
    min_volume: Optional[int] = 1_000_000
    min_avg_volume_5d: Optional[int] = None
    
    # Change filters
    min_percent_change: Optional[float] = None
    max_percent_change: Optional[float] = None
    
    # Market cap filters (external data required)
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    
    # Technical filters
    min_trades: Optional[int] = None
    sma_periods: List[int] = field(default_factory=list)
    
    # Volume surge detection
    volume_surge_multiplier: Optional[float] = None
    volume_surge_days: Optional[int] = None
    
    # Data source configuration
    feed: str = "iex"  # iex, sip, or other feed names
    max_symbols: int = 3000
    
    # Exchange filtering (NYSE and NASDAQ only for safety)
    exchanges: Optional[List[str]] = None  # NYSE, NASDAQ only


@dataclass
class VolumeSurge:
    """Data class for volume surge analysis results."""
    symbol: str
    current_volume: int
    avg_volume: float
    surge_ratio: float
    detected: bool
    days_analyzed: int


@dataclass
class StockResult:
    """Data class for stock screening results."""
    symbol: str
    price: float
    volume: int
    percent_change: float
    dollar_volume: float
    day_range: float
    timestamp: datetime
    
    # Optional fields requiring external data
    market_cap: Optional[float] = None
    trades: Optional[int] = None
    avg_volume_5d: Optional[float] = None
    avg_range_5d: Optional[float] = None
    
    # Volume surge analysis
    volume_surge_detected: bool = False
    volume_surge_ratio: Optional[float] = None
    
    # Technical indicators
    sma_values: Dict[int, float] = field(default_factory=dict)


class AlpacaScreener:
    """Main stock screener class with volume surge detection capabilities."""
    
    def __init__(self, provider: str = "alpaca", account: str = "Bruce", environment: str = "paper", 
                 verbose: bool = False):
        """
        Initialize the Alpaca screener.
        
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
        
        # Rate limiting configuration
        self.rate_limit_calls_per_minute = 200
        self.last_call_time = 0
        self.call_count = 0
        self.call_times = []
        
        if self.verbose:
            print(f"Initialized Alpaca Screener - Account: {account}, Environment: {environment}")

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

    def get_active_symbols(self, max_symbols: int = 3000, exchanges: Optional[List[str]] = None) -> List[str]:
        """
        Get list of actively traded symbols from Alpaca's most active endpoint or assets list.
        
        Args:
            max_symbols: Maximum number of symbols to return
            exchanges: List of exchanges to filter by (e.g., ['NYSE', 'NASDAQ'])
            
        Returns:
            List of symbol strings
        """
        if self.verbose:
            print("Fetching active symbols from Alpaca...")
            if exchanges:
                print(f"Filtering by exchanges: {', '.join(exchanges)}")
            
        self._rate_limit_check()
        
        # If exchange filtering is requested, use list_assets
        if exchanges:
            return self._get_symbols_by_exchange(exchanges, max_symbols)
        
        try:
            # Try to get most active stocks if the method exists
            if hasattr(self.client, 'get_most_actives'):
                most_actives = self.client.get_most_actives(by='volume', top=min(max_symbols, 1000))
                symbols = [stock.symbol for stock in most_actives]
                
                if self.verbose:
                    print(f"Found {len(symbols)} active symbols")
                    
                return symbols
            else:
                if self.verbose:
                    print("get_most_actives method not available, using fallback symbols")
        except Exception as e:
            if self.verbose:
                print(f"Error fetching active symbols: {e}")
                
        # Fallback to a predefined list of popular symbols
        fallback_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
            'AMD', 'INTC', 'BABA', 'DIS', 'PYPL', 'ADBE', 'CRM', 'ORCL',
            'UBER', 'LYFT', 'SNAP', 'ZOOM', 'DOCU', 'SHOP', 'SQ',
            'ROKU', 'PINS', 'DKNG', 'PLTR', 'GME', 'AMC', 'BB', 'NOK'
        ]
        if self.verbose:
            print(f"Using fallback symbols: {len(fallback_symbols[:max_symbols])} symbols")
        return fallback_symbols[:max_symbols]
    
    def _get_symbols_by_exchange(self, exchanges: List[str], max_symbols: int) -> List[str]:
        """
        Get symbols filtered by specific exchanges.
        
        Args:
            exchanges: List of exchange names to filter by
            max_symbols: Maximum number of symbols to return
            
        Returns:
            List of symbol strings from specified exchanges
        """
        try:
            # Convert exchanges to uppercase and validate safety
            exchanges_upper = [ex.upper() for ex in exchanges]
            safe_exchanges = ['NYSE', 'NASDAQ']
            
            # Safety check - only allow NYSE and NASDAQ
            for exchange in exchanges_upper:
                if exchange not in safe_exchanges:
                    if self.verbose:
                        print(f"Warning: Exchange {exchange} not in safe list (NYSE, NASDAQ). Skipping.")
                    exchanges_upper.remove(exchange)
            
            if not exchanges_upper:
                if self.verbose:
                    print("No valid exchanges specified. Defaulting to no exchange filtering.")
                return self.get_active_symbols(max_symbols, exchanges=None)
            
            if self.verbose:
                print(f"Fetching assets from safe exchanges: {', '.join(exchanges_upper)}")
            
            # Get all active assets
            assets = self.client.list_assets(status='active', asset_class='us_equity')
            
            # Filter by exchange and tradable status
            filtered_symbols = []
            exchange_counts = {}
            
            for asset in assets:
                if (asset.exchange.upper() in exchanges_upper and 
                    asset.tradable and 
                    asset.status == 'active'):
                    
                    filtered_symbols.append(asset.symbol)
                    exchange = asset.exchange.upper()
                    exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1
                    
                    if len(filtered_symbols) >= max_symbols:
                        break
            
            if self.verbose:
                print(f"Found {len(filtered_symbols)} tradable symbols from safe exchanges")
                for exchange, count in exchange_counts.items():
                    print(f"  {exchange}: {count} symbols")
            
            return filtered_symbols
            
        except Exception as e:
            if self.verbose:
                print(f"Error filtering by exchange: {e}")
            # Fallback to default symbols
            return self.get_active_symbols(max_symbols, exchanges=None)

    def collect_stock_data(self, symbols: List[str], lookback_days: int = 10) -> Dict[str, Dict]:
        """
        Collect current and historical data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days of historical data to collect
            
        Returns:
            Dictionary mapping symbols to their data
        """
        if self.verbose:
            print(f"Collecting data for {len(symbols)} symbols...")
            
        stock_data = {}
        batch_size = 200  # Process symbols in batches to avoid API limits
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            if self.verbose:
                print(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            self._rate_limit_check()
            
            try:
                # Get historical data for each symbol individually 
                for symbol in batch_symbols:
                    try:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=lookback_days)
                        
                        # Format dates properly for Alpaca API
                        start_str = start_date.strftime('%Y-%m-%d')
                        end_str = end_date.strftime('%Y-%m-%d')
                        
                        bars = self.client.get_bars(
                            symbol,
                            tradeapi.TimeFrame.Day,
                            start=start_str,
                            end=end_str,
                            limit=lookback_days + 5,
                            feed='iex'  # Use IEX for free tier
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
                                # Add trade count if available
                                if hasattr(bar, 'trade_count'):
                                    bar_dict['trade_count'] = bar.trade_count
                                bar_data.append(bar_dict)
                            
                            if bar_data:
                                df = pd.DataFrame(bar_data)
                                df.set_index('timestamp', inplace=True)
                                stock_data[symbol] = {
                                    'bars': df,
                                    'current_bar': df.iloc[-1],
                                    'historical_bars': df.iloc[:-1] if len(df) > 1 else df
                                }
                    except Exception as symbol_error:
                        if self.verbose:
                            print(f"Error fetching data for {symbol}: {symbol_error}")
                        continue
                            
            except Exception as e:
                if self.verbose:
                    print(f"Error collecting data for batch: {e}")
                continue
                
        if self.verbose:
            print(f"Successfully collected data for {len(stock_data)} symbols")
            
        return stock_data

    def detect_volume_surge(self, symbol: str, stock_data: Dict, n_multiplier: float, m_days: int) -> VolumeSurge:
        """
        Detect if current volume is N times higher than M-day average.
        
        Args:
            symbol: Stock ticker symbol
            stock_data: Historical stock data for the symbol
            n_multiplier: Volume multiplier threshold (e.g., 2.0 for 2x volume)
            m_days: Lookback period for average calculation
            
        Returns:
            VolumeSurge object with analysis results
        """
        try:
            bars = stock_data['bars']
            if len(bars) < 2:
                return VolumeSurge(symbol, 0, 0, 0, False, 0)
            
            current_volume = bars.iloc[-1]['volume']
            
            # Calculate average volume over m_days (excluding current day)
            historical_bars = bars.iloc[:-1]
            if len(historical_bars) == 0:
                avg_volume = current_volume
            else:
                avg_volume = historical_bars['volume'].mean()
            
            surge_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            detected = surge_ratio >= n_multiplier
            
            return VolumeSurge(
                symbol=symbol,
                current_volume=int(current_volume),
                avg_volume=float(avg_volume),
                surge_ratio=float(surge_ratio),
                detected=detected,
                days_analyzed=len(historical_bars)
            )
            
        except Exception as e:
            if self.verbose:
                print(f"Error detecting volume surge for {symbol}: {e}")
            return VolumeSurge(symbol, 0, 0, 0, False, 0)

    def calculate_sma(self, bars: pd.DataFrame, periods: List[int]) -> Dict[int, float]:
        """
        Calculate Simple Moving Averages for given periods.
        
        Args:
            bars: Historical price bars
            periods: List of periods for SMA calculation
            
        Returns:
            Dictionary mapping periods to SMA values
        """
        sma_values = {}
        close_prices = bars['close']
        
        for period in periods:
            if len(close_prices) >= period:
                sma_values[period] = float(close_prices.tail(period).mean())
            else:
                sma_values[period] = float(close_prices.mean())
                
        return sma_values

    def apply_filters(self, stock_data: Dict[str, Dict], criteria: ScreeningCriteria) -> List[StockResult]:
        """
        Apply screening criteria to stock data and return filtered results.
        
        Args:
            stock_data: Dictionary of stock data by symbol
            criteria: Screening criteria to apply
            
        Returns:
            List of StockResult objects that pass the filters
        """
        if self.verbose:
            print(f"Applying filters to {len(stock_data)} stocks...")
            
        results = []
        
        for symbol, data in stock_data.items():
            try:
                current_bar = data['current_bar']
                bars = data['bars']
                
                # Extract current metrics
                price = float(current_bar['close'])
                volume = int(current_bar['volume'])
                high = float(current_bar['high'])
                low = float(current_bar['low'])
                day_range = high - low
                
                # Calculate percent change (current vs previous close)
                if len(bars) >= 2:
                    prev_close = float(bars.iloc[-2]['close'])
                    percent_change = ((price - prev_close) / prev_close) * 100
                else:
                    percent_change = 0.0
                
                # Calculate dollar volume
                dollar_volume = volume * price
                
                # Apply price filters
                if criteria.min_price and price < criteria.min_price:
                    continue
                if criteria.max_price and price > criteria.max_price:
                    continue
                    
                # Apply volume filters
                if criteria.min_volume and volume < criteria.min_volume:
                    continue
                    
                # Apply change filters
                if criteria.min_percent_change and percent_change < criteria.min_percent_change:
                    continue
                if criteria.max_percent_change and percent_change > criteria.max_percent_change:
                    continue
                
                # Calculate average volume over 5 days
                avg_volume_5d = None
                avg_range_5d = None
                if len(bars) >= 5:
                    recent_bars = bars.tail(5)
                    avg_volume_5d = float(recent_bars['volume'].mean())
                    avg_range_5d = float((recent_bars['high'] - recent_bars['low']).mean())
                    
                    # Apply 5-day average volume filter
                    if criteria.min_avg_volume_5d and avg_volume_5d < criteria.min_avg_volume_5d:
                        continue
                
                # Volume surge detection
                volume_surge_detected = False
                volume_surge_ratio = None
                if criteria.volume_surge_multiplier and criteria.volume_surge_days:
                    surge = self.detect_volume_surge(
                        symbol, data, criteria.volume_surge_multiplier, criteria.volume_surge_days
                    )
                    volume_surge_detected = surge.detected
                    volume_surge_ratio = surge.surge_ratio if surge.surge_ratio > 0 else None
                
                # Calculate SMA values if requested
                sma_values = {}
                if criteria.sma_periods:
                    sma_values = self.calculate_sma(bars, criteria.sma_periods)
                
                # Get trade count if available
                trades = None
                if 'trade_count' in current_bar:
                    trades = int(current_bar['trade_count'])
                
                # Apply trade count filter
                if criteria.min_trades and trades and trades < criteria.min_trades:
                    continue
                
                # Create result object
                result = StockResult(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    percent_change=percent_change,
                    dollar_volume=dollar_volume,
                    day_range=day_range,
                    timestamp=pd.to_datetime(current_bar.name),
                    trades=trades,
                    avg_volume_5d=avg_volume_5d,
                    avg_range_5d=avg_range_5d,
                    volume_surge_detected=volume_surge_detected,
                    volume_surge_ratio=volume_surge_ratio,
                    sma_values=sma_values
                )
                
                results.append(result)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {symbol}: {e}")
                continue
        
        if self.verbose:
            print(f"Filtered results: {len(results)} stocks passed screening criteria")
            
        return results

    def screen_stocks(self, criteria: ScreeningCriteria) -> List[StockResult]:
        """
        Main screening method that orchestrates the entire screening process.
        
        Args:
            criteria: Screening criteria configuration
            
        Returns:
            List of StockResult objects that match the criteria
        """
        start_time = time.time()
        
        if self.verbose:
            print("Starting stock screening process...")
            
        # Get active symbols
        symbols = self.get_active_symbols(criteria.max_symbols, criteria.exchanges)
        
        if not symbols:
            print("No symbols found for screening")
            return []
        
        # Collect stock data
        lookback_days = max(10, criteria.volume_surge_days or 0, max(criteria.sma_periods) if criteria.sma_periods else 0)
        stock_data = self.collect_stock_data(symbols, lookback_days)
        
        if not stock_data:
            print("No stock data collected")
            return []
        
        # Apply filters and return results
        results = self.apply_filters(stock_data, criteria)
        
        end_time = time.time()
        if self.verbose:
            print(f"Screening completed in {end_time - start_time:.2f} seconds")
            
        return results

    def export_to_csv(self, results: List[StockResult], filename: str):
        """Export screening results to CSV file."""
        if not results:
            print("No results to export")
            return
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'symbol', 'price', 'volume', 'percent_change', 'dollar_volume',
                'day_range', 'timestamp', 'trades', 'avg_volume_5d', 'avg_range_5d',
                'volume_surge_detected', 'volume_surge_ratio'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'symbol': result.symbol,
                    'price': result.price,
                    'volume': result.volume,
                    'percent_change': result.percent_change,
                    'dollar_volume': result.dollar_volume,
                    'day_range': result.day_range,
                    'timestamp': result.timestamp.isoformat(),
                    'trades': result.trades,
                    'avg_volume_5d': result.avg_volume_5d,
                    'avg_range_5d': result.avg_range_5d,
                    'volume_surge_detected': result.volume_surge_detected,
                    'volume_surge_ratio': result.volume_surge_ratio
                }
                writer.writerow(row)
                
        print(f"Results exported to {filename}")

    def export_to_json(self, results: List[StockResult], filename: str, criteria: ScreeningCriteria):
        """Export screening results to JSON file with metadata."""
        if not results:
            print("No results to export")
            return
            
        export_data = {
            "scan_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_symbols_scanned": criteria.max_symbols,
                "results_count": len(results),
                "account": self.account,
                "environment": self.environment,
                "criteria": {
                    "min_price": criteria.min_price,
                    "max_price": criteria.max_price,
                    "min_volume": criteria.min_volume,
                    "min_avg_volume_5d": criteria.min_avg_volume_5d,
                    "min_percent_change": criteria.min_percent_change,
                    "max_percent_change": criteria.max_percent_change,
                    "volume_surge_multiplier": criteria.volume_surge_multiplier,
                    "volume_surge_days": criteria.volume_surge_days,
                    "min_trades": criteria.min_trades
                }
            },
            "results": []
        }
        
        for result in results:
            result_dict = {
                "symbol": result.symbol,
                "price": result.price,
                "volume": result.volume,
                "percent_change": result.percent_change,
                "dollar_volume": result.dollar_volume,
                "day_range": result.day_range,
                "timestamp": result.timestamp.isoformat(),
                "trades": result.trades,
                "avg_volume_5d": result.avg_volume_5d,
                "avg_range_5d": result.avg_range_5d,
                "volume_surge_detected": result.volume_surge_detected,
                "volume_surge_ratio": result.volume_surge_ratio,
                "market_cap": result.market_cap,
                "sma_values": result.sma_values
            }
            export_data["results"].append(result_dict)
        
        with open(filename, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, default=str)
            
        print(f"Results exported to {filename}")


def print_results(results: List[StockResult], criteria: ScreeningCriteria):
    """Print screening results in a formatted table."""
    if not results:
        print("No stocks found matching the screening criteria.")
        return
    
    print("\nAlpaca Stock Screener Results")
    print("=" * 80)
    print(f"Scan completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results found: {len(results)} stocks")
    print()
    
    # Sort results by volume surge ratio (if applicable), then by volume
    def sort_key(result):
        if result.volume_surge_ratio:
            return (result.volume_surge_detected, result.volume_surge_ratio, result.volume)
        return (False, 0, result.volume)
    
    results.sort(key=sort_key, reverse=True)
    
    # Print header
    header = f"{'Symbol':<8} {'Price':<8} {'Volume':<12} {'%Change':<8} {'$Volume':<12} {'Range':<8} {'Surge':<12}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for result in results:
        dollar_volume_str = f"${result.dollar_volume/1e9:.1f}B" if result.dollar_volume >= 1e9 else f"${result.dollar_volume/1e6:.1f}M"
        surge_str = "No"
        if result.volume_surge_detected and result.volume_surge_ratio:
            surge_str = f"Yes ({result.volume_surge_ratio:.1f}x)"
        
        volume_str = f"{result.volume/1e6:.1f}M" if result.volume >= 1e6 else f"{result.volume:,}"
        
        print(f"{result.symbol:<8} ${result.price:<7.2f} {volume_str:<12} {result.percent_change:>+6.2f}% "
              f"{dollar_volume_str:<12} ${result.day_range:<7.2f} {surge_str:<12}")


def parse_screener_args() -> argparse.Namespace:
    """Parse command line arguments for the stock screener."""
    parser = argparse.ArgumentParser(
        description='Alpaca Stock Screener with Volume Surge Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alpaca_screener.py --min-price 0.75 --min-volume 1000000
  python alpaca_screener.py --volume-surge 2.0 --surge-days 5
  python alpaca_screener.py --exchanges NYSE NASDAQ --min-price 1.0 --max-price 50.0
  python alpaca_screener.py --min-volume 500000 --export-csv results.csv --export-json results.json
        """
    )
    
    # Account and environment configuration
    parser.add_argument('--provider', default='alpaca', help='API provider (default: alpaca)')
    parser.add_argument('--account-name', default='Bruce', help='Account name (Bruce, Dale Wilson, Janice)')
    parser.add_argument('--account', default='paper', help='Account type (paper, live, cash)')
    
    # Screening criteria
    parser.add_argument('--min-price', type=float, help='Minimum stock price (USD)')
    parser.add_argument('--max-price', type=float, help='Maximum stock price (USD)')
    parser.add_argument('--min-volume', type=int, help='Minimum daily volume (shares)')
    parser.add_argument('--min-avg-volume-5d', type=int, help='Minimum 5-day average volume (shares)')
    parser.add_argument('--volume-surge', type=float, help='Volume surge multiplier (e.g., 2.0 for 2x)')
    parser.add_argument('--surge-days', type=int, default=5, help='Days for volume surge calculation (default: 5)')
    parser.add_argument('--min-percent-change', type=float, help='Minimum percent change (%%)')
    parser.add_argument('--max-percent-change', type=float, help='Maximum percent change (%%)')
    parser.add_argument('--min-trades', type=int, help='Minimum number of trades')
    parser.add_argument('--sma-periods', type=int, nargs='+', help='SMA periods to calculate (e.g., 20 50)')
    
    # Data source and limits
    parser.add_argument('--max-symbols', type=int, default=3000, help='Maximum symbols to analyze (default: 3000)')
    parser.add_argument('--feed', choices=['iex', 'sip', 'boats'], default='iex', help='Data feed to use (default: iex)')
    
    # Exchange filtering (NYSE and NASDAQ only for safety)
    parser.add_argument('--exchanges', type=str, nargs='+', choices=['NYSE', 'NASDAQ'], help='Filter by stock exchanges (NYSE, NASDAQ only)')
    
    # Output options  
    parser.add_argument('--export-csv', help='Export results to CSV file')
    parser.add_argument('--export-json', help='Export results to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main entry point for the stock screener."""
    args = parse_screener_args()
    
    # Available feed options
    valid_feeds = ['iex', 'sip', 'boats']
    if args.feed not in valid_feeds:
        args.feed = 'iex'  # Default to IEX
    
    # Create screening criteria
    criteria = ScreeningCriteria(
        min_price=args.min_price,
        max_price=args.max_price,
        min_volume=args.min_volume,
        min_avg_volume_5d=args.min_avg_volume_5d,
        min_percent_change=args.min_percent_change,
        max_percent_change=args.max_percent_change,
        volume_surge_multiplier=args.volume_surge,
        volume_surge_days=args.surge_days if args.volume_surge else None,
        min_trades=args.min_trades,
        sma_periods=args.sma_periods or [],
        feed=args.feed,
        max_symbols=args.max_symbols,
        exchanges=args.exchanges
    )
    
    # Initialize screener
    screener = AlpacaScreener(
        provider=args.provider,
        account=args.account_name,
        environment=args.account,
        verbose=args.verbose
    )
    
    try:
        # Run screening
        results = screener.screen_stocks(criteria)
        
        # Display results
        print_results(results, criteria)
        
        # Export results if requested
        if args.export_csv:
            screener.export_to_csv(results, args.export_csv)
            
        if args.export_json:
            screener.export_to_json(results, args.export_json, criteria)
            
    except KeyboardInterrupt:
        print("\nScreening interrupted by user")
    except Exception as e:
        print(f"Error during screening: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()