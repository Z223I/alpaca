#!/home/wilsonb/miniconda3/envs/alpaca/bin/python
"""
Scanner API Endpoint

Provides REST API for retrieving recent Momentum Alerts for the Market Sentinel Scanner panel.
Reads JSON files from the momentum_alerts_sent directory.

Usage:
    GET /api/scanner - Returns recent momentum alerts

GoDaddy CGI compatible.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
# Resolve symlinks to get the actual repository path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytz


def get_squeeze_alerts() -> List[Dict]:
    """
    Load recent squeeze alerts from the sent directory.

    Reads from:
    historical_data/{YYYY-MM-DD}/squeeze_alerts_sent/alert_*.json

    Returns:
        List of dictionaries with fields:
        - symbol: Stock symbol
        - source: "Squeeze"
        - time: Time in ET (HH:MM:SS)
        - price: High price from squeeze
        - gain: Percent change during squeeze
        - volume: Trade size
        - text: Squeeze details
        - timestamp: ISO timestamp for sorting
    """
    et_tz = pytz.timezone('US/Eastern')
    today = datetime.now(et_tz).strftime('%Y-%m-%d')

    # Directory path for squeeze alerts sent
    alerts_dir = Path(project_root) / "historical_data" / today / "squeeze_alerts_sent"

    alerts_list = []

    # Check if directory exists
    if not alerts_dir.exists():
        print(f"Squeeze alerts directory not found: {alerts_dir}", file=sys.stderr)
        return alerts_list

    # Find all JSON alert files
    alert_files = sorted(alerts_dir.glob('alert_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)

    # Limit to last 50 alerts
    alert_files = alert_files[:50]

    print(f"Found {len(alert_files)} squeeze alert files", file=sys.stderr)

    for alert_file in alert_files:
        try:
            with open(alert_file, 'r') as f:
                alert_data = json.load(f)

            # Parse timestamp
            timestamp_str = alert_data.get('timestamp', '')
            if timestamp_str:
                # Parse ISO timestamp with timezone
                timestamp_dt = datetime.fromisoformat(timestamp_str)
                # Ensure it's in ET
                if timestamp_dt.tzinfo is None:
                    timestamp_dt = et_tz.localize(timestamp_dt)
                else:
                    timestamp_dt = timestamp_dt.astimezone(et_tz)
                time_str = timestamp_dt.strftime('%H:%M:%S')
            else:
                time_str = "N/A"

            # Extract data
            symbol = alert_data.get('symbol', 'N/A')
            low_price = alert_data.get('low_price', 0)
            high_price = alert_data.get('high_price', 0)
            percent_change = alert_data.get('percent_change', 0)
            size = alert_data.get('size', 0)
            window_trades = alert_data.get('window_trades', 0)

            # Build text field
            text = f"🚀 +{percent_change:.2f}% squeeze | ${low_price:.2f} → ${high_price:.2f} | {window_trades} trades"

            alert_obj = {
                'symbol': symbol,
                'source': 'Squeeze',
                'time': time_str,
                'price': high_price,  # Show high price
                'gain': percent_change,  # Show percent change as "gain"
                'volume': size,  # Show trade size as "volume"
                'text': text,
                'timestamp': timestamp_str  # For sorting
            }

            alerts_list.append(alert_obj)

        except Exception as e:
            print(f"Error reading squeeze alert file {alert_file}: {e}", file=sys.stderr)
            continue

    # Sort by timestamp descending (newest first)
    alerts_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    return alerts_list


def get_momentum_alerts() -> List[Dict]:
    """
    Load recent momentum alerts from the sent directory.

    Reads from:
    historical_data/{YYYY-MM-DD}/momentum_alerts_sent/bullish/alert_*.json

    Returns:
        List of dictionaries with fields:
        - symbol: Stock symbol
        - source: "Momentum"
        - time: Time in ET (HH:MM:SS)
        - price: Current price
        - gain: Percent gain since market open
        - volume: Current volume
        - text: Combined text with momentum indicators
        - timestamp: ISO timestamp for sorting
    """
    et_tz = pytz.timezone('US/Eastern')
    today = datetime.now(et_tz).strftime('%Y-%m-%d')

    # Directory path for momentum alerts sent
    alerts_dir = Path(project_root) / "historical_data" / today / "momentum_alerts_sent" / "bullish"

    alerts_list = []

    # Check if directory exists
    if not alerts_dir.exists():
        print(f"Alerts directory not found: {alerts_dir}", file=sys.stderr)
        return alerts_list

    # Find all JSON alert files
    alert_files = sorted(alerts_dir.glob('alert_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)

    # Limit to last 50 alerts
    alert_files = alert_files[:50]

    print(f"Found {len(alert_files)} alert files", file=sys.stderr)

    for alert_file in alert_files:
        try:
            with open(alert_file, 'r') as f:
                alert_data = json.load(f)

            # Parse timestamp
            timestamp_str = alert_data.get('timestamp', '')
            if timestamp_str:
                # Parse ISO timestamp with timezone
                timestamp_dt = datetime.fromisoformat(timestamp_str)
                # Ensure it's in ET
                if timestamp_dt.tzinfo is None:
                    timestamp_dt = et_tz.localize(timestamp_dt)
                else:
                    timestamp_dt = timestamp_dt.astimezone(et_tz)
                time_str = timestamp_dt.strftime('%H:%M:%S')
            else:
                time_str = "N/A"

            # Extract data
            symbol = alert_data.get('symbol', 'N/A')
            price = alert_data.get('current_price', 0)
            gain = alert_data.get('percent_gain_since_market_open', 0)
            volume = alert_data.get('current_volume', 0)

            # Build text field from momentum indicators
            momentum_emoji = alert_data.get('momentum_emoji', '')
            momentum_short_emoji = alert_data.get('momentum_short_emoji', '')
            squeeze_emoji = alert_data.get('squeeze_emoji', '')
            halt_emoji = alert_data.get('halt_emoji', '')
            float_shares = alert_data.get('float_shares')

            text_parts = []
            text_parts.append(f"Momentum {momentum_emoji}")
            text_parts.append(f"Short {momentum_short_emoji}")
            text_parts.append(f"Squeeze {squeeze_emoji}")
            text_parts.append(f"Halt {halt_emoji}")

            # Add float shares (formatted) - only if valid
            if float_shares is not None and float_shares > 0:
                # Convert to float if needed and validate
                try:
                    float_shares = float(float_shares)
                    if float_shares > 0:
                        if float_shares >= 1_000_000_000:
                            float_str = f"{float_shares / 1_000_000_000:.2f}B"
                        elif float_shares >= 1_000_000:
                            float_str = f"{float_shares / 1_000_000:.2f}M"
                        else:
                            float_str = f"{float_shares:,.0f}"
                        text_parts.append(f"Float {float_str}")
                except (ValueError, TypeError):
                    # Invalid float_shares value, skip it
                    pass

            text = " | ".join(text_parts)

            alert_obj = {
                'symbol': symbol,
                'source': 'Momentum',
                'time': time_str,
                'price': price,
                'gain': gain,
                'volume': volume,
                'text': text,
                'timestamp': timestamp_str  # For sorting
            }

            alerts_list.append(alert_obj)

        except Exception as e:
            print(f"Error reading alert file {alert_file}: {e}", file=sys.stderr)
            continue

    # Sort by timestamp descending (newest first)
    alerts_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    return alerts_list


def main():
    """Main CGI handler."""
    # Print CGI headers
    print("Content-Type: application/json")
    print("Access-Control-Allow-Origin: *")
    print("")  # End of headers

    try:
        # Get both momentum and squeeze alerts
        momentum_alerts = get_momentum_alerts()
        squeeze_alerts = get_squeeze_alerts()

        # Combine and sort by timestamp
        all_alerts = momentum_alerts + squeeze_alerts
        all_alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Limit to 50 most recent alerts total
        all_alerts = all_alerts[:50]

        response = {
            'success': True,
            'count': len(all_alerts),
            'momentum_count': len(momentum_alerts),
            'squeeze_count': len(squeeze_alerts),
            'data': all_alerts
        }

        print(json.dumps(response, indent=2))

    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_response, indent=2))
        print(f"Error in scanner_api: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
