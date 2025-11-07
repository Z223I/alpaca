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
script_dir = os.path.dirname(os.path.abspath(__file__))
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
    Get momentum alerts from the last N minutes (only sent alerts).

    Args:
        minutes: Number of minutes to look back (default: 60)

    Returns:
        List of alert dictionaries sorted by timestamp (newest first)
    """
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    cutoff_time = now - timedelta(minutes=minutes)

    alerts = []

    # Check today's sent alerts (only bullish)
    today = now.strftime('%Y-%m-%d')
    alerts_dir = Path(project_root) / "historical_data" / today / "momentum_alerts_sent" / "bullish"

    if alerts_dir.exists():
        for alert_file in alerts_dir.glob("alert_*.json"):
            try:
                with open(alert_file, 'r') as f:
                    alert_data = json.load(f)

                # Parse timestamp
                alert_timestamp = datetime.fromisoformat(alert_data['timestamp'])
                if alert_timestamp.tzinfo is None:
                    alert_timestamp = et_tz.localize(alert_timestamp)

                # Only include alerts within the time window
                if alert_timestamp >= cutoff_time:
                    alerts.append(alert_data)

            except Exception as e:
                print(f"Error reading alert file {alert_file}: {e}", file=sys.stderr)
                continue

    # Sort by timestamp (newest first)
    alerts.sort(key=lambda x: x['timestamp'], reverse=True)

    return alerts


def get_new_alerts(since_timestamp):
    """
    Get momentum alerts since a given timestamp (only sent alerts).

    Args:
        since_timestamp: ISO format timestamp string

    Returns:
        List of alert dictionaries sorted by timestamp (newest first)
    """
    et_tz = pytz.timezone('US/Eastern')

    # Parse the since_timestamp
    try:
        cutoff_time = datetime.fromisoformat(since_timestamp)
        if cutoff_time.tzinfo is None:
            cutoff_time = et_tz.localize(cutoff_time)
    except Exception as e:
        return {'error': f'Invalid timestamp format: {e}'}

    alerts = []

    # Check today's sent alerts (only bullish)
    now = datetime.now(et_tz)
    today = now.strftime('%Y-%m-%d')
    alerts_dir = Path(project_root) / "historical_data" / today / "momentum_alerts_sent" / "bullish"

    if alerts_dir.exists():
        for alert_file in alerts_dir.glob("alert_*.json"):
            try:
                with open(alert_file, 'r') as f:
                    alert_data = json.load(f)

                # Parse timestamp
                alert_timestamp = datetime.fromisoformat(alert_data['timestamp'])
                if alert_timestamp.tzinfo is None:
                    alert_timestamp = et_tz.localize(alert_timestamp)

                # Only include alerts after the cutoff
                if alert_timestamp > cutoff_time:
                    alerts.append(alert_data)

            except Exception as e:
                print(f"Error reading alert file {alert_file}: {e}", file=sys.stderr)
                continue

    # Sort by timestamp (newest first)
    alerts.sort(key=lambda x: x['timestamp'], reverse=True)

    return alerts


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
