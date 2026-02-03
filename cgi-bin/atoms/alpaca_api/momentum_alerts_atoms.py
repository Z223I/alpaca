"""
Momentum alerts data collection and analysis utilities.

Functions for collecting momentum alert data from JSON files and
calculating max gains per symbol.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from glob import glob


def collect_alerts_by_symbol(date_str, base_path):
    """Collect all alerts with prices and timestamps for each symbol on a given date.

    Args:
        date_str: Date string in YYYY-MM-DD format
        base_path: Base path to the project root

    Returns:
        dict: Symbol -> list of {price, timestamp} dicts
    """
    symbol_alerts = defaultdict(list)
    alerts_path = os.path.join(
        base_path,
        'historical_data',
        date_str,
        'momentum_alerts_sent',
        'bullish',
        'alert_*.json'
    )

    for json_file in glob(alerts_path):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                symbol = data.get('symbol')
                price = data.get('current_price')
                timestamp_str = data.get('timestamp')
                if symbol and price is not None and timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    symbol_alerts[symbol].append({
                        'price': price,
                        'timestamp': timestamp
                    })
        except (json.JSONDecodeError, IOError, ValueError) as e:
            print(f"Warning: Could not read {json_file}: {e}")

    return symbol_alerts


def calculate_max_gains(symbol_alerts):
    """Calculate max gain for each symbol with timing info.

    Uses first alert (earliest timestamp) as starting price,
    and max price alert as ending price.

    Args:
        symbol_alerts: dict from collect_alerts_by_symbol()

    Returns:
        list: List of dicts with symbol, max_gain, max_price, first_price,
              first_time, max_time, num_alerts
    """
    max_gains = []
    for symbol, alerts in symbol_alerts.items():
        if len(alerts) >= 2:
            # Use first alert (earliest timestamp) as starting point
            first_alert = min(alerts, key=lambda x: x['timestamp'])
            max_alert = max(alerts, key=lambda x: x['price'])
            first_price = first_alert['price']
            max_price = max_alert['price']

            if first_price > 0:
                max_gain = max_price / first_price
                first_time = first_alert['timestamp']
                max_time = max_alert['timestamp']

                max_gains.append({
                    'symbol': symbol,
                    'max_gain': max_gain,
                    'max_price': max_price,
                    'first_price': first_price,
                    'first_time': first_time,
                    'max_time': max_time,
                    'num_alerts': len(alerts)
                })
    return max_gains
