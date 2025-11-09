#!/home/wilsonb/miniconda3/envs/alpaca/bin/python
"""
Market Data CGI API Endpoint for Market Sentinel

This CGI script provides REST-like endpoints for the Market Sentinel web interface.
It bridges the frontend (HTML/JS) with the backend (market_data.py).

Endpoints:
- ?action=quote&symbol=AAPL
- ?action=chart&symbol=AAPL&interval=1m&range=1d
- ?action=trades&symbol=AAPL&limit=100
"""

import sys
import os
import json
import traceback
from urllib.parse import parse_qs
import cgitb

# Enable CGI error reporting for debugging
cgitb.enable()

# Add paths for importing market_data module
# cgi-bin/api/market_data_api.py -> go up to cgi-bin/molecules/alpaca_molecules/
script_dir = os.path.dirname(os.path.abspath(__file__))  # cgi-bin/api/
cgi_bin_dir = os.path.dirname(script_dir)  # cgi-bin/
molecules_dir = os.path.join(cgi_bin_dir, 'molecules', 'alpaca_molecules')  # cgi-bin/molecules/alpaca_molecules/
repo_root = os.path.dirname(cgi_bin_dir)  # Repository root

sys.path.insert(0, molecules_dir)
sys.path.insert(0, repo_root)


def send_response(data, status=200, error=None):
    """
    Send JSON response with proper HTTP headers.

    Args:
        data: Dictionary to send as JSON
        status: HTTP status code
        error: Optional error message
    """
    # HTTP headers
    print("Content-Type: application/json")
    print("Access-Control-Allow-Origin: *")  # Allow CORS
    print("Access-Control-Allow-Methods: GET, POST, OPTIONS")
    print("Access-Control-Allow-Headers: Content-Type")

    if status != 200:
        print(f"Status: {status}")

    print()  # End of headers

    # Body
    response = {
        'success': error is None,
        'data': data if error is None else None,
        'error': error
    }

    print(json.dumps(response, indent=2))


def get_query_params():
    """Parse CGI query string parameters."""
    query_string = os.environ.get('QUERY_STRING', '')
    params = parse_qs(query_string)

    # Convert lists to single values
    return {k: v[0] if len(v) == 1 else v for k, v in params.items()}


def handle_quote(params):
    """Handle quote request."""
    from market_data import AlpacaMarketData

    symbol = params.get('symbol', '').upper()
    if not symbol:
        send_response(None, 400, "Missing 'symbol' parameter")
        return

    try:
        client = AlpacaMarketData()
        quote_data = client.get_latest_quote_data(symbol)

        if 'error' in quote_data:
            send_response(None, 500, quote_data['error'])
        else:
            send_response(quote_data)

    except Exception as e:
        send_response(None, 500, f"Error fetching quote: {str(e)}")


def handle_chart(params):
    """Handle chart data request."""
    from market_data import AlpacaMarketData

    symbol = params.get('symbol', '').upper()
    interval = params.get('interval', '1m')
    range_str = params.get('range', '1d')

    if not symbol:
        send_response(None, 400, "Missing 'symbol' parameter")
        return

    try:
        client = AlpacaMarketData()
        chart_data = client.get_chart_data(symbol, interval, range_str)

        if 'error' in chart_data:
            send_response(None, 500, chart_data.get('error'))
        else:
            # Calculate indicators if requested
            indicators_param = params.get('indicators', '')
            if indicators_param:
                indicators = indicators_param.split(',')
                # Filter out VWAP if not a 1-day chart (VWAP only valid for intraday)
                if range_str != '1d' and 'vwap' in indicators:
                    indicators = [ind for ind in indicators if ind != 'vwap']
                chart_data['indicators'] = calculate_indicators(chart_data['bars'], indicators, range_str)

            send_response(chart_data)

    except Exception as e:
        send_response(None, 500, f"Error fetching chart data: {str(e)}")


def handle_trades(params):
    """Handle time and sales (trades) request."""
    from market_data import AlpacaMarketData
    from datetime import datetime, timedelta
    import pytz

    symbol = params.get('symbol', '').upper()
    limit = int(params.get('limit', 100))

    if not symbol:
        send_response(None, 400, "Missing 'symbol' parameter")
        return

    try:
        client = AlpacaMarketData()
        et_tz = pytz.timezone('America/New_York')
        start_date = datetime.now(et_tz) - timedelta(hours=1)

        trades = client.get_time_and_sales(symbol, start_date, limit)

        send_response({'trades': trades, 'count': len(trades)})

    except Exception as e:
        send_response(None, 500, f"Error fetching trades: {str(e)}")


def calculate_indicators(bars, indicators, range_str='1d'):
    """
    Calculate technical indicators for the chart.

    Args:
        bars: List of bar dictionaries
        indicators: List of indicator names to calculate
        range_str: Time range of the chart (e.g., '1d', '5d', '1mo', '1y')

    Returns:
        Dictionary of indicator data
    """
    if not bars:
        return {}

    import pandas as pd
    import numpy as np

    # Convert bars to DataFrame
    df = pd.DataFrame(bars)

    result = {}

    for indicator in indicators:
        indicator = indicator.strip().lower()

        if indicator == 'ema9':
            result['ema9'] = calculate_ema(df, 9)
        elif indicator == 'ema21':
            result['ema21'] = calculate_ema(df, 21)
        elif indicator == 'ema50':
            result['ema50'] = calculate_ema(df, 50)
        elif indicator == 'ema200':
            result['ema200'] = calculate_ema(df, 200)
        elif indicator == 'vwap':
            result['vwap'] = calculate_vwap(df)
        elif indicator == 'macd':
            result['macd'] = calculate_macd(df)
        elif indicator == 'volume':
            result['volume'] = calculate_volume_data(df)

    return result


def calculate_ema(df, period):
    """Calculate Exponential Moving Average."""
    import pandas as pd
    import numpy as np

    if len(df) < period:
        return []

    ema = df['close'].ewm(span=period, adjust=False).mean()

    # Return as list of {time, value} objects, filtering out invalid values
    result = []
    for i, (timestamp, value) in enumerate(zip(df['timestamp'], ema)):
        # Only include valid, finite, non-NaN values
        if not pd.isna(value) and not np.isinf(value) and value > 0:
            result.append({
                'time': timestamp,
                'value': float(value)
            })

    return result


def calculate_vwap(df):
    """Calculate Volume Weighted Average Price (resets daily)."""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import pytz

    if 'volume' not in df.columns or len(df) == 0:
        return []

    # Convert timestamps to datetime for grouping by day
    df = df.copy()

    # Ensure proper datetime conversion - timestamps may be strings or datetime objects
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        # If not datetime, convert (may be ISO strings)
        df['datetime'] = pd.to_datetime(df['timestamp'], utc=True)
    else:
        # Already datetime, just copy
        df['datetime'] = df['timestamp']

    df['date'] = df['datetime'].dt.date

    result = []

    # Calculate VWAP separately for each trading day
    for date, day_df in df.groupby('date'):
        typical_price = (day_df['high'] + day_df['low'] + day_df['close']) / 3

        # Calculate cumulative VWAP for this day
        cumulative_tp_volume = (typical_price * day_df['volume']).cumsum()
        cumulative_volume = day_df['volume'].cumsum()

        # Avoid division by zero
        vwap = cumulative_tp_volume / cumulative_volume.replace(0, np.nan)

        for timestamp, value, vol in zip(day_df['timestamp'], vwap, day_df['volume']):
            # Only include valid values with actual volume
            if not pd.isna(value) and vol > 0 and not np.isinf(value):
                result.append({
                    'time': timestamp,
                    'value': float(value)
                })

    return result


def calculate_macd(df):
    """Calculate MACD indicator."""
    import pandas as pd

    if len(df) < 26:
        return {'macd': [], 'signal': [], 'histogram': []}

    # Calculate MACD components
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    # Format for TradingView Lightweight Charts
    macd_data = []
    signal_data = []
    histogram_data = []

    for i, (timestamp, macd_val, signal_val, hist_val) in enumerate(
        zip(df['timestamp'], macd_line, signal_line, histogram)
    ):
        if not pd.isna(macd_val):
            macd_data.append({'time': timestamp, 'value': float(macd_val)})
            signal_data.append({'time': timestamp, 'value': float(signal_val)})
            histogram_data.append({'time': timestamp, 'value': float(hist_val)})

    return {
        'macd': macd_data,
        'signal': signal_data,
        'histogram': histogram_data
    }


def calculate_volume_data(df):
    """Format volume data for display."""
    if 'volume' not in df.columns:
        return []

    result = []
    for timestamp, volume in zip(df['timestamp'], df['volume']):
        result.append({
            'time': timestamp,
            'value': int(volume)
        })

    return result


def main():
    """Main CGI handler."""
    try:
        # Get query parameters
        params = get_query_params()
        action = params.get('action', '')

        # Route to appropriate handler
        if action == 'quote':
            handle_quote(params)
        elif action == 'chart':
            handle_chart(params)
        elif action == 'trades':
            handle_trades(params)
        else:
            send_response(None, 400, f"Invalid or missing 'action' parameter. Valid actions: quote, chart, trades")

    except Exception as e:
        # Catch-all error handler
        error_msg = f"Unexpected error: {str(e)}\n\n{traceback.format_exc()}"
        send_response(None, 500, error_msg)


if __name__ == "__main__":
    main()
