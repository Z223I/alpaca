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
# Use realpath to resolve symlinks properly
script_real_path = os.path.realpath(__file__)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_real_path))))
sys.path.insert(0, repo_root)

from atoms.api.init_alpaca_client import init_alpaca_client  # noqa: E402
from atoms.api.get_latest_quote import get_latest_quote  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest, StockTradesRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: E402


class AlpacaMarketData:
    """
    Alpaca Market Data API wrapper for the Market Sentinel web interface.

    This class provides methods for retrieving real-time and historical market data,
    specifically designed for web-based chart displays and market monitoring.
    No trading functionality is included.
    """

    # Mapping of interval strings to Alpaca TimeFrame objects
    # Note: alpaca-py does not support second-level timeframes
    TIMEFRAME_MAP = {
        '1m': TimeFrame.Minute,
        '5m': TimeFrame(5, TimeFrameUnit.Minute),
        '15m': TimeFrame(15, TimeFrameUnit.Minute),
        '30m': TimeFrame(30, TimeFrameUnit.Minute),
        '1h': TimeFrame.Hour,
        '4h': TimeFrame(4, TimeFrameUnit.Hour),
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

    def __init__(self, provider: str = "alpaca", account_name: str = "Bruce", account: str = "live"):
        """
        Initialize the Alpaca market data client.

        IMPORTANT: Uses LIVE account for SIP (Securities Information Processor) data access.
        Live account provides real-time, high-quality market data from the consolidated tape.
        Paper account data has significant delays (15+ minutes to hours).

        Args:
            provider: Provider name (default: "alpaca")
            account_name: Account name (default: "Bruce")
            account: Account type - "paper", "live", or "cash" (default: "live")
                     NOTE: Default changed from "paper" to "live" to access SIP data.
                     This uses read-only market data API and does NOT place any trades.

        Credentials Configuration:
            Set your live Alpaca API credentials using environment variables:
            - ALPACA_LIVE_API_KEY: Your live account API key
            - ALPACA_LIVE_SECRET_KEY: Your live account secret key

            Or edit alpaca_config.py lines 136-137 directly:
            live=EnvironmentConfig(
                app_key=os.getenv("ALPACA_LIVE_API_KEY", "YOUR_KEY_HERE"),
                app_secret=os.getenv("ALPACA_LIVE_SECRET_KEY", "YOUR_SECRET_HERE"),
                url="https://api.alpaca.markets"
            )
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
            timeframe: Bar timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo')
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
            # alpaca-py returns a BarSet object with data in bars.data dictionary
            if hasattr(bars, 'data') and symbol in bars.data:
                symbol_bars = bars.data[symbol]
                data = []
                for bar in symbol_bars:
                    # alpaca-py uses full attribute names (not single letters like alpaca-trade-api)
                    data.append({
                        'timestamp': bar.timestamp,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume),
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
            interval: Candlestick interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo')
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
            # Note: Alpaca's historical bar data has a delay (typically 15-60 seconds)
            # Request data up to "now" to get the most recent available bars
            end_date = datetime.now(self.et_tz)

            if range_str not in self.DISPLAY_RANGE_MAP:
                raise ValueError(f"Invalid range: {range_str}. Valid options: {list(self.DISPLAY_RANGE_MAP.keys())}")

            days = self.DISPLAY_RANGE_MAP[range_str]
            start_date = end_date - timedelta(days=days)

            # Get bar data
            df = self.get_bar_data(symbol, interval, start_date, end_date)

            # If no data and requesting intraday data, extend lookback to capture recent market days
            # This handles cases when market is closed (weekends, holidays)
            if df.empty and interval in ['1m', '5m', '15m', '30m', '1h']:
                # Extend lookback by up to 7 days to find recent market data
                extended_start = end_date - timedelta(days=days + 7)
                df = self.get_bar_data(symbol, interval, extended_start, end_date)

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
            start_date: Start date/time for trades (default: last 10 minutes for recent trades)
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
                # Use last 2 hours to capture recent trades, especially for low-volume stocks
                # The trades will be sorted by timestamp (newest first) and limited anyway
                start_date = datetime.now(self.et_tz) - timedelta(hours=2)

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
            # alpaca-py returns a TradeSet object with data in trades.data dictionary
            trade_list = []
            if hasattr(trades, 'data') and symbol in trades.data:
                for trade in trades.data[symbol]:
                    # Ensure timestamp is in ET timezone before converting to ISO format
                    timestamp = trade.timestamp if hasattr(trade, 'timestamp') else datetime.now(self.et_tz)
                    # Convert UTC timestamp to ET
                    if timestamp.tzinfo is None:
                        # If naive, assume UTC
                        import pytz
                        timestamp = pytz.UTC.localize(timestamp)
                    timestamp_et = timestamp.astimezone(self.et_tz)

                    trade_list.append({
                        'timestamp': timestamp_et.isoformat(),
                        'timestamp_dt': timestamp_et,  # Keep datetime for sorting
                        'price': float(trade.price) if hasattr(trade, 'price') else 0.0,
                        'size': int(trade.size) if hasattr(trade, 'size') else 0,
                        'exchange': str(trade.exchange) if hasattr(trade, 'exchange') else ''
                    })

            # Sort trades by timestamp in descending order (newest first)
            # This ensures we always show the most recent trades at the top
            trade_list.sort(key=lambda x: x['timestamp_dt'], reverse=True)

            # Remove the datetime object used for sorting
            for trade in trade_list:
                del trade['timestamp_dt']

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
