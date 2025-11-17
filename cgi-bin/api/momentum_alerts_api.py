#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
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

# Add cgi-bin/molecules to path for AlpacaMarketData
molecules_path = os.path.join(project_root, 'cgi-bin', 'molecules')
if molecules_path not in sys.path:
    sys.path.insert(0, molecules_path)

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


# Global market data client instance (initialized lazily)
_market_data_client = None


def get_company_name(symbol):
    """
    Get company name for a symbol using AlpacaMarketData class.

    This method uses the AlpacaMarketData class which first tries to read from
    data_master/master.csv, and if not found, falls back to the Alpaca API using
    the newer alpaca-py package.

    Args:
        symbol: Stock symbol to look up

    Returns:
        Company name string if found, None otherwise
    """
    global _market_data_client

    try:
        # Initialize market data client if not already done
        if _market_data_client is None:
            from alpaca_molecules.market_data import AlpacaMarketData
            _market_data_client = AlpacaMarketData(provider="alpaca", account_name="Bruce", account="paper")

        # Use the get_company_name method from AlpacaMarketData
        return _market_data_client.get_company_name(symbol)

    except Exception as e:
        print(f"Error getting company name for {symbol}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


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
