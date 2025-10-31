#!/usr/bin/env python3
"""
Alpaca Market Data API Wrapper for Market Sentinel Web Interface

This module provides data collection and display functionality for stock market data.
It is specifically designed for the Market Sentinel web interface and does NOT include
any trading functionality.

Features:
- Real-time quote retrieval
- Historical bar data (multiple timeframes)
- SIP (Securities Information Processor) data access
- Candlestick chart data preparation
- Time and sales data
"""

import sys
import os
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Add paths for importing from main codebase atoms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from atoms.api.init_alpaca_client import init_alpaca_client  # noqa: E402
from atoms.api.get_latest_quote import get_latest_quote  # noqa: E402
import alpaca_trade_api as tradeapi  # noqa: E402


class AlpacaMarketData:
    """
    Alpaca Market Data API wrapper for the Market Sentinel web interface.

    This class provides methods for retrieving real-time and historical market data,
    specifically designed for web-based chart displays and market monitoring.
    No trading functionality is included.
    """

    # Mapping of interval strings to Alpaca TimeFrame objects
    TIMEFRAME_MAP = {
        '10s': TimeFrame(10, 'Second'),
        '20s': TimeFrame(20, 'Second'),
        '30s': TimeFrame(30, 'Second'),
        '1m': TimeFrame.Minute,
        '5m': TimeFrame(5, 'Minute'),
        '30m': TimeFrame(30, 'Minute'),
        '1h': TimeFrame.Hour,
        '1d': TimeFrame.Day,
        '1w': TimeFrame.Week,
        '1mo': TimeFrame.Month,
    }

    # Mapping of display ranges to days
    DISPLAY_RANGE_MAP = {
        '1d': 1,
        '2d': 2,
        '5d': 5,
        '1mo': 30,
        '1y': 365,
    }

    def __init__(self, provider: str = "alpaca", account_name: str = "Bruce", account: str = "paper"):
        """
        Initialize the Alpaca market data client.

        Args:
            provider: Provider name (default: "alpaca")
            account_name: Account name (default: "Bruce")
            account: Account type - "paper", "live", or "cash" (default: "paper")
        """
        self.provider = provider
        self.account_name = account_name
        self.account = account

        # Initialize Alpaca API client
        self.api = init_alpaca_client(provider, account_name, account)

        # Initialize historical data client for SIP access
        from alpaca_config import get_api_credentials
        api_key, secret_key, base_url = get_api_credentials(provider, account_name, account)
        self.hist_client = StockHistoricalDataClient(api_key, secret_key)

        # Eastern Time timezone for market hours
        self.et_tz = pytz.timezone('America/New_York')

    def get_latest_quote_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary containing quote data with keys:
                - symbol: str
                - bid_price: float
                - ask_price: float
                - mid_price: float (average of bid/ask)
                - bid_size: int
                - ask_size: int
                - timestamp: str (ISO format)
        """
        try:
            quote = get_latest_quote(self.api, symbol, self.account_name, self.account)

            bid_price = float(quote.bid_price) if hasattr(quote, 'bid_price') else 0.0
            ask_price = float(quote.ask_price) if hasattr(quote, 'ask_price') else 0.0
            mid_price = (bid_price + ask_price) / 2.0

            return {
                'symbol': symbol,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'mid_price': mid_price,
                'bid_size': int(quote.bid_size) if hasattr(quote, 'bid_size') else 0,
                'ask_size': int(quote.ask_size) if hasattr(quote, 'ask_size') else 0,
                'timestamp': (quote.timestamp.isoformat() if hasattr(quote, 'timestamp')
                              else datetime.now(self.et_tz).isoformat())
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now(self.et_tz).isoformat()
            }

    def get_bar_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical bar data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe ('10s', '20s', '30s', '1m', '5m', '30m', '1h', '1d', '1w', '1mo')
            start_date: Start date/time for data
            end_date: End date/time for data

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, symbol
        """
        try:
            # Convert timeframe string to Alpaca TimeFrame object
            if timeframe not in self.TIMEFRAME_MAP:
                raise ValueError(f"Invalid timeframe: {timeframe}. Valid options: {list(self.TIMEFRAME_MAP.keys())}")

            tf = self.TIMEFRAME_MAP[timeframe]

            # Request bar data
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_date,
                end=end_date,
                limit=10000  # Max bars
            )

            bars = self.hist_client.get_stock_bars(request_params)

            # Convert to DataFrame
            if symbol in bars:
                symbol_bars = bars[symbol]
                data = []
                for bar in symbol_bars:
                    # CRITICAL: Use single-letter attributes (bar.o, bar.h, bar.l, bar.c, bar.v, bar.t)
                    data.append({
                        'timestamp': bar.t,
                        'open': float(bar.o),
                        'high': float(bar.h),
                        'low': float(bar.l),
                        'close': float(bar.c),
                        'volume': int(bar.v),
                        'symbol': symbol
                    })

                df = pd.DataFrame(data)

                # Ensure timestamp is datetime with timezone
                if not df.empty:
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    if df['timestamp'].dt.tz is None:
                        df['timestamp'] = df['timestamp'].dt.tz_localize(self.et_tz)
                    elif str(df['timestamp'].dt.tz) != 'America/New_York':
                        df['timestamp'] = df['timestamp'].dt.tz_convert(self.et_tz)

                return df.sort_values('timestamp').reset_index(drop=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Error getting bar data: {e}")
            return pd.DataFrame()

    def get_chart_data(self, symbol: str, interval: str = '1m', range_str: str = '1d') -> Dict[str, Any]:
        """
        Get chart data ready for display in the web interface.

        Args:
            symbol: Stock symbol
            interval: Candlestick interval ('10s', '20s', '30s', '1m', '5m', '30m', '1h', '1d', '1w', '1mo')
            range_str: Display range ('1d', '2d', '5d', '1mo', '1y')

        Returns:
            Dictionary containing:
                - symbol: str
                - interval: str
                - range: str
                - bars: list of dict (timestamp, open, high, low, close, volume)
                - start_date: str (ISO format)
                - end_date: str (ISO format)
        """
        try:
            # Calculate date range
            end_date = datetime.now(self.et_tz)

            if range_str not in self.DISPLAY_RANGE_MAP:
                raise ValueError(f"Invalid range: {range_str}. Valid options: {list(self.DISPLAY_RANGE_MAP.keys())}")

            days = self.DISPLAY_RANGE_MAP[range_str]
            start_date = end_date - timedelta(days=days)

            # Get bar data
            df = self.get_bar_data(symbol, interval, start_date, end_date)

            if df.empty:
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'range': range_str,
                    'bars': [],
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'error': 'No data available'
                }

            # Convert DataFrame to list of dictionaries
            bars = df.to_dict('records')

            # Convert timestamps to ISO format strings
            for bar in bars:
                if isinstance(bar['timestamp'], pd.Timestamp):
                    bar['timestamp'] = bar['timestamp'].isoformat()

            return {
                'symbol': symbol,
                'interval': interval,
                'range': range_str,
                'bars': bars,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'bar_count': len(bars)
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'interval': interval,
                'range': range_str,
                'error': str(e),
                'start_date': None,
                'end_date': None,
                'bars': []
            }

    def get_time_and_sales(self, symbol: str, start_date: Optional[datetime] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get time and sales (trade) data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date/time for trades (default: last 1 hour)
            limit: Maximum number of trades to return (default: 100)

        Returns:
            List of trade dictionaries with keys:
                - timestamp: str (ISO format)
                - price: float
                - size: int
                - exchange: str
        """
        try:
            if start_date is None:
                start_date = datetime.now(self.et_tz) - timedelta(hours=1)

            end_date = datetime.now(self.et_tz)

            # Request trade data
            request_params = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start_date,
                end=end_date,
                limit=limit
            )

            trades = self.hist_client.get_stock_trades(request_params)

            # Convert to list of dictionaries
            trade_list = []
            if symbol in trades:
                for trade in trades[symbol]:
                    trade_list.append({
                        'timestamp': (trade.t.isoformat() if hasattr(trade, 't')
                                      else datetime.now(self.et_tz).isoformat()),
                        'price': float(trade.p) if hasattr(trade, 'p') else 0.0,
                        'size': int(trade.s) if hasattr(trade, 's') else 0,
                        'exchange': str(trade.x) if hasattr(trade, 'x') else ''
                    })

            return trade_list

        except Exception as e:
            print(f"Error getting time and sales data: {e}")
            return []

    def to_json(self, data: Any) -> str:
        """
        Convert data to JSON string for web interface.

        Args:
            data: Data to convert (dict, list, DataFrame, etc.)

        Returns:
            JSON string
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dict records
            data = data.to_dict('records')
            # Convert timestamps
            for record in data:
                for key, value in record.items():
                    if isinstance(value, pd.Timestamp):
                        record[key] = value.isoformat()

        return json.dumps(data, indent=2, default=str)


# Example usage and testing
if __name__ == "__main__":
    def test_market_data():
        """Test the market data retrieval."""
        print("Testing Alpaca Market Data API...")

        # Initialize client
        client = AlpacaMarketData()

        # Test symbol
        symbol = "AAPL"

        # Test 1: Get latest quote
        print(f"\n{'='*60}")
        print(f"Test 1: Getting latest quote for {symbol}")
        print(f"{'='*60}")
        quote = client.get_latest_quote_data(symbol)
        print(json.dumps(quote, indent=2))

        # Test 2: Get 1-minute bar data for 1 day
        print(f"\n{'='*60}")
        print(f"Test 2: Getting 1-minute chart data for {symbol} (1 day)")
        print(f"{'='*60}")
        chart_data = client.get_chart_data(symbol, interval='1m', range_str='1d')
        print(f"Retrieved {chart_data.get('bar_count', 0)} bars")
        if chart_data.get('bars'):
            print(f"First bar: {chart_data['bars'][0]}")
            print(f"Last bar: {chart_data['bars'][-1]}")

        # Test 3: Get time and sales
        print(f"\n{'='*60}")
        print(f"Test 3: Getting time and sales for {symbol}")
        print(f"{'='*60}")
        trades = client.get_time_and_sales(symbol, limit=10)
        print(f"Retrieved {len(trades)} trades")
        if trades:
            print(f"Latest trade: {json.dumps(trades[-1], indent=2)}")

        print(f"\n{'='*60}")
        print("All tests completed!")
        print(f"{'='*60}")

    # Run tests
    test_market_data()
