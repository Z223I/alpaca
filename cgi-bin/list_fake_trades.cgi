#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
"""
CGI script to list fake trades for a given date.

Returns JSON with all fake trade data for the specified date.
"""

import cgi
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def get_fake_trades(date_str):
    """
    Load all fake trades for a given date.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        List of fake trade dictionaries
    """
    try:
        # Convert YYYY-MM-DD to YYYY-MM-DD format for directory
        # The directory uses the same format
        fake_trades_dir = Path(project_root) / "historical_data" / date_str / "fake_trades"

        if not fake_trades_dir.exists():
            return []

        trades = []

        # Read all JSON files in the fake_trades directory
        for json_file in fake_trades_dir.glob("fake_trade_*.json"):
            try:
                with open(json_file, 'r') as f:
                    trade_data = json.load(f)
                    trades.append(trade_data)
            except Exception as e:
                # Log error but continue processing other files
                print(f"Error reading {json_file}: {e}", file=sys.stderr)
                continue

        # Sort by entry timestamp (newest first)
        trades.sort(key=lambda t: t.get('entry_timestamp', ''), reverse=True)

        return trades

    except Exception as e:
        print(f"Error loading fake trades: {e}", file=sys.stderr)
        return []

def main():
    """Main CGI handler"""
    # Parse query parameters
    form = cgi.FieldStorage()
    date_str = form.getvalue('date', datetime.now().strftime('%Y-%m-%d'))

    # Get fake trades
    trades = get_fake_trades(date_str)

    # Build response
    response = {
        'date': date_str,
        'count': len(trades),
        'trades': trades
    }

    # Send JSON response
    print("Content-Type: application/json")
    print("Cache-Control: no-cache")
    print()
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Return error as JSON
        print("Content-Type: application/json")
        print()
        print(json.dumps({
            'error': str(e),
            'trades': []
        }))
