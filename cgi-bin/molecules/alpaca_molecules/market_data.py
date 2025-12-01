#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
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

# Import VWAP calculator
from vwap_calculator import get_latest_vwap

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
from alpaca.trading.client import TradingClient  # noqa: E402
from pathlib import Path  # noqa: E402
import csv  # noqa: E402


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

        # Initialize historical data client
        # Note: We specify feed='iex' in requests for real-time free data
        from alpaca_config import get_api_credentials
        api_key, secret_key, base_url = get_api_credentials(provider, account_name, account)
        self.hist_client = StockHistoricalDataClient(api_key, secret_key)

        # Initialize trading client for asset information
        self.trading_client = TradingClient(api_key, secret_key, paper=(account == "paper"))

        # Eastern Time timezone for market hours
        self.et_tz = pytz.timezone('America/New_York')

        # Store repo root for CSV access
        self.repo_root = repo_root

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

    def _clean_company_name(self, full_name: str) -> str:
        """
        Clean company name by removing common suffixes like 'Common Stock', 'Inc.', etc.

        Args:
            full_name: Full asset name from Alpaca API

        Returns:
            Cleaned company name
        """
        if not full_name or full_name == "N/A":
            return full_name

        # List of suffixes to remove (order matters - more specific first)
        suffixes_to_remove = [
            ' Class A Ordinary Shares',
            ' Class B Ordinary Shares',
            ' Class C Ordinary Shares',
            ' Class A Common Stock',
            ' Class B Common Stock',
            ' Class C Common Stock',
            ' Common Stock',
            ' Ordinary Shares',
            ' American Depositary Shares',
            ' American Depositary Receipt',
            ' Depositary Shares',
            ' ETF Trust',
            ' Shares',
            ' Inc.',
            ' Inc',
            ' Corporation',
            ' Corp.',
            ' Corp',
            ' Limited',
            ' Ltd.',
            ' Ltd',
            ' L.P.',
            ' LP',
            ' LLC',
            ' PLC',
            ' plc',
            ' S.A.',
            ' SA',
            ' N.V.',
            ' NV',
            ' AG',
        ]

        cleaned = full_name.strip()

        # Remove suffixes
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
                # Check again for multiple suffixes (e.g., "Company Inc. Common Stock")
                for suffix2 in suffixes_to_remove:
                    if cleaned.endswith(suffix2):
                        cleaned = cleaned[:-len(suffix2)].strip()
                        break
                break

        # Remove trailing comma if present
        cleaned = cleaned.rstrip(',').strip()

        return cleaned

    def get_company_name(self, symbol: str) -> Optional[str]:
        """
        Get company name for a symbol from master.csv or Alpaca API.

        This method first tries to read from data_master/master.csv.
        If not found, it falls back to the Alpaca API using the TradingClient.

        Args:
            symbol: Stock symbol to look up

        Returns:
            Company name string if found, None otherwise
        """
        # First try: read from master.csv
        master_csv_path = Path(self.repo_root) / "data_master" / "master.csv"

        if master_csv_path.exists():
            try:
                with open(master_csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('symbol', '').upper() == symbol.upper():
                            company_name = row.get('company_name', '').strip()
                            if company_name and company_name != 'N/A':
                                return company_name
            except Exception as e:
                print(f"Error reading master.csv: {e}", file=sys.stderr)

        # Fallback: fetch from Alpaca API using TradingClient
        print(f"Company name not found in master.csv for {symbol}, falling back to Alpaca API", file=sys.stderr)
        try:
            asset = self.trading_client.get_asset(symbol)
            if asset and hasattr(asset, 'name') and asset.name:
                return self._clean_company_name(asset.name)
        except Exception as e:
            print(f"Error fetching asset from Alpaca for {symbol}: {e}", file=sys.stderr)
            return None

        return None

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
            # IMPORTANT: Use SIP feed - this account has paid SIP subscription for real-time data
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_date,
                end=end_date,
                limit=10000,  # Max bars
                feed='sip'  # Paid SIP feed for comprehensive real-time market data
            )

            bars = self.hist_client.get_stock_bars(request_params)

            # Convert to DataFrame
            # alpaca-py returns a BarSet object with data in bars.data dictionary
            if hasattr(bars, 'data') and symbol in bars.data:
                symbol_bars = bars.data[symbol]
                data = []
                for bar in symbol_bars:
                    # alpaca-py uses full attribute names (not single letters like alpaca-trade-api)
                    bar_dict = {
                        'timestamp': bar.timestamp,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume),
                        'symbol': symbol
                    }
                    # Add VWAP if available
                    if hasattr(bar, 'vwap') and bar.vwap is not None:
                        bar_dict['vwap'] = float(bar.vwap)
                    data.append(bar_dict)

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

                # Filter to keep only the most recent N trading days worth of data
                # Group by date and keep only the most recent 'days' worth of trading days
                if not df.empty and 'timestamp' in df.columns:
                    try:
                        # Make a copy to avoid SettingWithCopyWarning
                        df = df.copy()

                        # Ensure timestamp is datetime type (only convert if not already datetime)
                        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                        # Get unique trading dates in the data
                        df['date'] = df['timestamp'].dt.date
                        unique_dates = sorted(df['date'].unique(), reverse=True)

                        # Keep only the most recent N trading days
                        if len(unique_dates) > days:
                            keep_dates = unique_dates[:days]
                            df = df[df['date'].isin(keep_dates)]

                        # Remove temporary date column
                        df = df.drop('date', axis=1)
                    except Exception as e:
                        # Log the error but don't filter - return all data
                        import sys
                        print(f"ERROR in date filtering: {str(e)}", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)

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

    def get_latest_candlestick(self, symbol: str, timeframe: str = '1m') -> Dict[str, Any]:
        """
        Get the latest candlestick for a symbol with VWAP data calculated from 4:00 AM ET.

        IMPORTANT: VWAP is calculated over the entire time period of the request.
        To get accurate intraday VWAP, we fetch all bars since 4:00 AM ET and use
        the VWAP from the latest candlestick.

        Args:
            symbol: Stock symbol
            timeframe: Candlestick timeframe (default: '1m')

        Returns:
            Dictionary containing:
                - symbol: str
                - timestamp: str (ISO format)
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: int
                - vwap: float or None (VWAP calculated from 4:00 AM ET to latest bar)
                - error: str (if any error occurred)
        """
        try:
            # Get current time in ET
            now_et = datetime.now(self.et_tz)

            # Start from 4:00 AM ET today
            start_date = self.et_tz.localize(
                datetime.combine(now_et.date(), datetime.min.time())
            ).replace(hour=4, minute=0, second=0, microsecond=0)

            # End at current time
            end_date = now_et

            # Fetch all bars since 4:00 AM ET
            df = self.get_bar_data(symbol, timeframe, start_date, end_date)

            if df.empty:
                return {
                    'symbol': symbol,
                    'error': 'No candlestick data available'
                }

            # Get the latest bar (last row) - this has VWAP calculated from 4:00 AM to now
            latest_bar = df.iloc[-1]

            # Get VWAP from Alpaca bar data, or calculate it as fallback
            vwap = None
            if 'vwap' in latest_bar and pd.notna(latest_bar['vwap']):
                # Use VWAP from Alpaca API
                vwap = float(latest_bar['vwap'])
            else:
                # Fallback: Calculate VWAP using vwap_calculator module
                bars_list = df.to_dict('records')
                vwap = get_latest_vwap(bars_list)

            result = {
                'symbol': symbol,
                'timestamp': latest_bar['timestamp'].isoformat() if isinstance(latest_bar['timestamp'], pd.Timestamp) else str(latest_bar['timestamp']),
                'open': float(latest_bar['open']),
                'high': float(latest_bar['high']),
                'low': float(latest_bar['low']),
                'close': float(latest_bar['close']),
                'volume': int(latest_bar['volume']),
                'vwap': vwap,
                'bars_fetched': len(df)
            }

            return result

        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e)
            }

    def get_day_highs(self, symbol: str, trading_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate Premarket High and Regular Hours HOD for a symbol.

        Args:
            symbol: Stock symbol
            trading_date: Trading date to calculate for (default: today in ET timezone)

        Returns:
            Dictionary containing:
                - symbol: str
                - date: str (YYYY-MM-DD)
                - premarket_high: float or None (4:00 AM - 9:30 AM ET)
                - premarket_high_time: str or None (ISO format timestamp)
                - regular_hours_hod: float or None (9:30 AM - 4:00 PM ET)
                - regular_hours_hod_time: str or None (ISO format timestamp)
                - error: str (if any error occurred)
        """
        try:
            # Use today in ET timezone if no date provided
            if trading_date is None:
                trading_date = datetime.now(self.et_tz)

            # Ensure trading_date is timezone-aware in ET
            if trading_date.tzinfo is None:
                trading_date = self.et_tz.localize(trading_date)
            else:
                trading_date = trading_date.astimezone(self.et_tz)

            # Get the date string
            date_str = trading_date.strftime('%Y-%m-%d')

            # Define time ranges
            # Premarket: 4:00 AM - 9:30 AM ET
            premarket_start = self.et_tz.localize(
                datetime.combine(trading_date.date(), datetime.min.time())
            ).replace(hour=4, minute=0, second=0, microsecond=0)

            premarket_end = self.et_tz.localize(
                datetime.combine(trading_date.date(), datetime.min.time())
            ).replace(hour=9, minute=30, second=0, microsecond=0)

            # Regular hours: 9:30 AM - 4:00 PM ET
            regular_start = premarket_end
            regular_end = self.et_tz.localize(
                datetime.combine(trading_date.date(), datetime.min.time())
            ).replace(hour=16, minute=0, second=0, microsecond=0)

            # Get 1-minute bars for the entire day (4:00 AM - 4:00 PM)
            # This captures both premarket and regular hours
            df = self.get_bar_data(symbol, '1m', premarket_start, regular_end)

            if df.empty:
                return {
                    'symbol': symbol,
                    'date': date_str,
                    'premarket_high': None,
                    'premarket_high_time': None,
                    'regular_hours_hod': None,
                    'regular_hours_hod_time': None,
                    'error': 'No data available for this date'
                }

            # Ensure timestamp column is timezone-aware
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(self.et_tz)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.et_tz)

            # Filter for premarket hours (4:00 AM - 9:30 AM)
            premarket_df = df[(df['timestamp'] >= premarket_start) & (df['timestamp'] < premarket_end)]

            # Filter for regular hours (9:30 AM - 4:00 PM)
            regular_df = df[(df['timestamp'] >= regular_start) & (df['timestamp'] <= regular_end)]

            # Calculate premarket high
            premarket_high = None
            premarket_high_time = None
            if not premarket_df.empty:
                max_idx = premarket_df['high'].idxmax()
                premarket_high = float(premarket_df.loc[max_idx, 'high'])
                premarket_high_time = premarket_df.loc[max_idx, 'timestamp'].isoformat()

            # Calculate regular hours HOD
            regular_hours_hod = None
            regular_hours_hod_time = None
            if not regular_df.empty:
                max_idx = regular_df['high'].idxmax()
                regular_hours_hod = float(regular_df.loc[max_idx, 'high'])
                regular_hours_hod_time = regular_df.loc[max_idx, 'timestamp'].isoformat()

            return {
                'symbol': symbol,
                'date': date_str,
                'premarket_high': premarket_high,
                'premarket_high_time': premarket_high_time,
                'regular_hours_hod': regular_hours_hod,
                'regular_hours_hod_time': regular_hours_hod_time,
                'premarket_bars': len(premarket_df),
                'regular_hours_bars': len(regular_df)
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'date': date_str if 'date_str' in locals() else None,
                'premarket_high': None,
                'premarket_high_time': None,
                'regular_hours_hod': None,
                'regular_hours_hod_time': None,
                'error': str(e)
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
            # Use SIP feed for complete market coverage (all exchanges)
            request_params = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start_date,
                end=end_date,
                limit=limit,
                feed='sip'  # Paid SIP feed - 100% market coverage, all exchanges
            )

            trades = self.hist_client.get_stock_trades(request_params)

            # Convert to list of dictionaries
            # alpaca-py returns a TradeSet object with data in trades.data dictionary
            trade_list = []
            if hasattr(trades, 'data') and symbol in trades.data:
                for trade in trades.data[symbol]:
                    # Alpaca returns timestamps in UTC - convert to ET
                    # Note: IEX feed timestamps appear to be -1 hour off, so we add 1 hour
                    timestamp = trade.timestamp if hasattr(trade, 'timestamp') else datetime.now(self.et_tz)

                    # Convert to ET
                    if timestamp.tzinfo is not None:
                        timestamp_et = timestamp.astimezone(self.et_tz)
                    else:
                        # If naive, assume UTC
                        import pytz
                        timestamp_et = pytz.UTC.localize(timestamp).astimezone(self.et_tz)

                    # IEX feed appears to be 1 hour behind - add correction
                    timestamp_et = timestamp_et + timedelta(hours=1)

                    trade_list.append({
                        'timestamp': timestamp_et.isoformat(),
                        'price': float(trade.price) if hasattr(trade, 'price') else 0.0,
                        'size': int(trade.size) if hasattr(trade, 'size') else 0,
                        'exchange': str(trade.exchange) if hasattr(trade, 'exchange') else ''
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

        # Test 4: Get day highs (Premarket High and Regular Hours HOD)
        print(f"\n{'='*60}")
        print(f"Test 4: Getting Premarket High and Regular Hours HOD for {symbol}")
        print(f"{'='*60}")
        day_highs = client.get_day_highs(symbol)
        print(json.dumps(day_highs, indent=2))

        print(f"\n{'='*60}")
        print("All tests completed!")
        print(f"{'='*60}")

    # Run tests
    test_market_data()
