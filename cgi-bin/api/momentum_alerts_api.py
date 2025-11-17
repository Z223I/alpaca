#!/usr/bin/env python3
"""
Momentum Alerts API

Provides access to momentum alerts for the web interface.
Supports polling for new alerts and retrieving alert history.
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pytz

# Add project root to path
# When running from /var/www/html/market_sentinel, use the actual project path
# When running from project directory, calculate relative path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check if we're running from the web directory or user directory
if '/var/www/html/market_sentinel' in script_dir or '/home/wilsonb/public_html' in script_dir:
    # Use absolute path to actual project directory
    project_root = '/home/wilsonb/dl/github.com/Z223I/alpaca'
else:
    # Calculate relative path from project directory
    api_dir = script_dir
    cgi_bin_dir = os.path.dirname(api_dir)
    project_root = os.path.dirname(cgi_bin_dir)

sys.path.insert(0, project_root)

# CGI headers
print("Content-Type: application/json")
print("Access-Control-Allow-Origin: *")
print()


def get_query_params():
    """Parse query string parameters."""
    import urllib.parse
    query_string = os.environ.get('QUERY_STRING', '')
    params = urllib.parse.parse_qs(query_string)
    return {k: v[0] if v else None for k, v in params.items()}


def get_recent_alerts(minutes=60):
    """
    Get momentum alerts from the last N minutes (both sent and unsent alerts).

    Args:
        minutes: Number of minutes to look back (default: 60)

    Returns:
        List of alert dictionaries sorted by timestamp (newest first)
    """
    et_tz = pytz.timezone('America/New_York')
    now = datetime.now(et_tz)
    cutoff_time = now - timedelta(minutes=minutes)

    alerts = []
    seen_alerts = set()  # Track unique alerts by (symbol, timestamp)

    # Check today's date
    today = now.strftime('%Y-%m-%d')

    # Check both sent and regular alerts directories (only bullish)
    alerts_dirs = [
        Path(project_root) / "historical_data" / today / "momentum_alerts_sent" / "bullish",
        Path(project_root) / "historical_data" / today / "momentum_alerts" / "bullish"
    ]

    for alerts_dir in alerts_dirs:
        if alerts_dir.exists():
            for alert_file in alerts_dir.glob("alert_*.json"):
                try:
                    with open(alert_file, 'r') as f:
                        alert_data = json.load(f)

                    # Parse timestamp
                    alert_timestamp = datetime.fromisoformat(alert_data['timestamp'])
                    if alert_timestamp.tzinfo is None:
                        alert_timestamp = et_tz.localize(alert_timestamp)

                    # Create unique key for this alert
                    alert_key = (alert_data['symbol'], alert_data['timestamp'])

                    # Only include alerts within the time window and not already seen
                    if alert_timestamp >= cutoff_time and alert_key not in seen_alerts:
                        alerts.append(alert_data)
                        seen_alerts.add(alert_key)

                except Exception as e:
                    print(f"Error reading alert file {alert_file}: {e}", file=sys.stderr)
                    continue

    # Sort by timestamp (newest first)
    alerts.sort(key=lambda x: x['timestamp'], reverse=True)

    return alerts


def get_new_alerts(since_timestamp):
    """
    Get momentum alerts since a given timestamp (both sent and unsent alerts).

    Args:
        since_timestamp: ISO format timestamp string

    Returns:
        List of alert dictionaries sorted by timestamp (newest first)
    """
    et_tz = pytz.timezone('America/New_York')

    # Parse the since_timestamp
    try:
        cutoff_time = datetime.fromisoformat(since_timestamp)
        if cutoff_time.tzinfo is None:
            cutoff_time = et_tz.localize(cutoff_time)
    except Exception as e:
        return {'error': f'Invalid timestamp format: {e}'}

    alerts = []
    seen_alerts = set()  # Track unique alerts by (symbol, timestamp)

    # Check today's date
    now = datetime.now(et_tz)
    today = now.strftime('%Y-%m-%d')

    # Check both sent and regular alerts directories (only bullish)
    alerts_dirs = [
        Path(project_root) / "historical_data" / today / "momentum_alerts_sent" / "bullish",
        Path(project_root) / "historical_data" / today / "momentum_alerts" / "bullish"
    ]

    for alerts_dir in alerts_dirs:
        if alerts_dir.exists():
            for alert_file in alerts_dir.glob("alert_*.json"):
                try:
                    with open(alert_file, 'r') as f:
                        alert_data = json.load(f)

                    # Parse timestamp
                    alert_timestamp = datetime.fromisoformat(alert_data['timestamp'])
                    if alert_timestamp.tzinfo is None:
                        alert_timestamp = et_tz.localize(alert_timestamp)

                    # Create unique key for this alert
                    alert_key = (alert_data['symbol'], alert_data['timestamp'])

                    # Only include alerts after the cutoff and not already seen
                    if alert_timestamp > cutoff_time and alert_key not in seen_alerts:
                        alerts.append(alert_data)
                        seen_alerts.add(alert_key)

                except Exception as e:
                    print(f"Error reading alert file {alert_file}: {e}", file=sys.stderr)
                    continue

    # Sort by timestamp (newest first)
    alerts.sort(key=lambda x: x['timestamp'], reverse=True)

    return alerts


def clean_company_name(full_name: str) -> str:
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


def get_company_name_from_alpaca(symbol: str) -> str:
    """
    Fetch company name from Alpaca API as fallback.

    Args:
        symbol: Stock symbol to look up

    Returns:
        Company name string if found, None otherwise
    """
    try:
        # Add atoms directory to path for imports
        atoms_api_path = os.path.join(project_root, 'atoms', 'api')
        if atoms_api_path not in sys.path:
            sys.path.insert(0, atoms_api_path)

        from init_alpaca_client import init_alpaca_client

        # Initialize Alpaca client (using Bruce/paper as default)
        client = init_alpaca_client("alpaca", "Bruce", "paper")

        # Fetch asset information for the symbol
        try:
            asset = client.get_asset(symbol)
            if asset and asset.name:
                return clean_company_name(asset.name)
        except Exception as e:
            print(f"Error fetching asset from Alpaca for {symbol}: {e}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Error initializing Alpaca client: {e}", file=sys.stderr)
        return None


def get_company_name(symbol):
    """
    Get company name for a symbol from data_master/master.csv.
    If not found, falls back to Alpaca API (same mechanism as add_company_names.py).

    Args:
        symbol: Stock symbol to look up

    Returns:
        Company name string if found, None otherwise
    """
    import csv

    master_csv_path = Path(project_root) / "data_master" / "master.csv"

    # First try: read from master.csv
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

    # Fallback: fetch from Alpaca API (same mechanism as add_company_names.py)
    print(f"Company name not found in master.csv for {symbol}, falling back to Alpaca API", file=sys.stderr)
    return get_company_name_from_alpaca(symbol)


def main():
    """Main API handler."""
    try:
        params = get_query_params()
        action = params.get('action', 'recent')

        if action == 'recent':
            # Get recent alerts (last 60 minutes by default)
            minutes = int(params.get('minutes', 60))
            alerts = get_recent_alerts(minutes)

            response = {
                'success': True,
                'action': 'recent',
                'minutes': minutes,
                'count': len(alerts),
                'alerts': alerts
            }

        elif action == 'poll':
            # Get new alerts since timestamp
            since = params.get('since')

            if not since:
                response = {
                    'success': False,
                    'error': 'Missing "since" parameter for poll action'
                }
            else:
                alerts = get_new_alerts(since)

                if isinstance(alerts, dict) and 'error' in alerts:
                    response = {
                        'success': False,
                        'error': alerts['error']
                    }
                else:
                    response = {
                        'success': True,
                        'action': 'poll',
                        'count': len(alerts),
                        'alerts': alerts
                    }

        elif action == 'company_name':
            # Get company name for a symbol
            symbol = params.get('symbol')

            if not symbol:
                response = {
                    'success': False,
                    'error': 'Missing "symbol" parameter for company_name action'
                }
            else:
                company_name = get_company_name(symbol)

                if company_name:
                    response = {
                        'success': True,
                        'action': 'company_name',
                        'symbol': symbol,
                        'company_name': company_name
                    }
                else:
                    response = {
                        'success': False,
                        'error': f'Company name not found for symbol: {symbol}'
                    }

        else:
            response = {
                'success': False,
                'error': f'Unknown action: {action}'
            }

        print(json.dumps(response, indent=2))

    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_response, indent=2))
        import traceback
        traceback.print_exc(file=sys.stderr)


if __name__ == '__main__':
    main()
