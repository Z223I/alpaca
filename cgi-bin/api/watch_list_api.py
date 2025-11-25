#!/home/wilsonb/miniconda3/envs/alpaca/bin/python
"""
Watch List API Endpoint

Provides REST API for retrieving the current stock watch list with source indicators.
This endpoint reads the same CSV files as momentum_alerts.py but provides a lightweight
interface suitable for web consumption.

Usage:
    GET /api/watchlist - Returns current watch list with source indicators

GoDaddy CGI compatible.
"""

import csv
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


def find_latest_oracle_csv() -> Path:
    """
    Find the latest CSV file in the ./data directory.

    Returns:
        Path to the latest CSV file, or None if no CSV files found
    """
    data_dir = Path(project_root) / "data"

    if not data_dir.exists():
        return None

    # Find all CSV files matching the date pattern YYYYMMDD.csv
    csv_files = list(data_dir.glob('????????.csv'))

    if not csv_files:
        return None

    # Sort by filename (which is the date) and return the latest
    csv_files.sort(reverse=True)
    return csv_files[0]


def get_watch_list_symbols() -> List[Dict]:
    """
    Load watch list symbols from CSV files and return with source indicators.

    Reads from:
    1. historical_data/{YYYY-MM-DD}/top_gainers_nasdaq_amex.csv (premarket top gainers)
    2. historical_data/{YYYY-MM-DD}/volume_surge/relative_volume_nasdaq_amex.csv (surge)
    3. data/{YYYYMMDD}.csv (oracle) - falls back to latest CSV if today's not found

    Returns:
        List of dictionaries with fields:
        - symbol: Stock symbol
        - oracle: Boolean - from Oracle data source
        - manual: Boolean - manually added (always False for now)
        - top_gainers: Boolean - from premarket top gainers list
        - surge: Boolean - from volume surge list
        - gain_percent: Float or None - percentage gain from top_gainers or volume_surge
        - surge_amount: Float or None - volume surge ratio from volume_surge
    """
    et_tz = pytz.timezone('US/Eastern')
    today = datetime.now(et_tz).strftime('%Y-%m-%d')
    compact_date = datetime.now(et_tz).strftime('%Y%m%d')

    # File paths
    gainers_csv = Path(project_root) / "historical_data" / today / "premarket" / "top_gainers_nasdaq_amex.csv"
    volume_surge_csv = Path(project_root) / "historical_data" / today / "volume_surge" / "relative_volume_nasdaq_amex.csv"
    oracle_csv = Path(project_root) / "data" / f"{compact_date}.csv"

    # Fallback to latest CSV if today's doesn't exist
    if not oracle_csv.exists():
        latest_csv = find_latest_oracle_csv()
        if latest_csv:
            oracle_csv = latest_csv
            print(f"Using fallback Oracle CSV: {oracle_csv.name}", file=sys.stderr)

    symbols_dict = {}

    # Load from premarket top gainers CSV file - keep first 40 symbols that don't end in 'W'
    if gainers_csv.exists():
        try:
            gainers_count = 0
            with open(gainers_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('symbol', '').strip().upper()
                    # Filter: must have symbol and not end in 'W'
                    if symbol and not symbol.endswith('W'):
                        if gainers_count < 40:  # Only keep first 40
                            # Get gain_percent, default to None if not available
                            gain_percent = None
                            if 'gain_percent' in row:
                                try:
                                    gain_percent = float(row['gain_percent'])
                                except (ValueError, TypeError):
                                    pass

                            symbols_dict[symbol] = {
                                'oracle': False,
                                'manual': False,
                                'top_gainers': True,
                                'surge': False,
                                'gain_percent': gain_percent,
                                'surge_amount': None
                            }
                            gainers_count += 1
                        else:
                            break  # Stop after first 40
        except Exception as e:
            print(f"Error loading premarket top gainers CSV: {e}", file=sys.stderr)

    # Load from volume surge CSV file - keep first 40 symbols that don't end in 'W'
    if volume_surge_csv.exists():
        try:
            volume_surge_count = 0
            with open(volume_surge_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('symbol', '').strip().upper()
                    # Filter: must have symbol and not end in 'W'
                    if symbol and not symbol.endswith('W'):
                        # Get surge_amount (volume_surge_ratio), default to None if not available
                        surge_amount = None
                        if 'volume_surge_ratio' in row:
                            try:
                                surge_amount = float(row['volume_surge_ratio'])
                            except (ValueError, TypeError):
                                pass

                        # Also get percent_change for gain_percent if available
                        gain_percent = None
                        if 'percent_change' in row:
                            try:
                                gain_percent = float(row['percent_change'])
                            except (ValueError, TypeError):
                                pass

                        if symbol in symbols_dict:
                            # Symbol already exists - update surge flag and surge_amount
                            symbols_dict[symbol]['surge'] = True
                            symbols_dict[symbol]['surge_amount'] = surge_amount
                            # Update gain_percent only if not already set (prefer top_gainers value)
                            if symbols_dict[symbol].get('gain_percent') is None:
                                symbols_dict[symbol]['gain_percent'] = gain_percent
                        elif volume_surge_count < 40:  # Only add first 40 new symbols
                            symbols_dict[symbol] = {
                                'oracle': False,
                                'manual': False,
                                'top_gainers': False,
                                'surge': True,
                                'gain_percent': gain_percent,
                                'surge_amount': surge_amount
                            }
                            volume_surge_count += 1
                        else:
                            continue  # Skip additional symbols beyond first 40 new ones
        except Exception as e:
            print(f"Error loading volume surge CSV: {e}", file=sys.stderr)

    # Load from Oracle CSV file - all symbols (no limit)
    if oracle_csv.exists():
        try:
            with open(oracle_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try both 'Symbol' (capital) and 'symbol' (lowercase) for compatibility
                    symbol = row.get('Symbol', row.get('symbol', '')).strip().upper()
                    # Filter: must have symbol and not end in 'W'
                    if symbol and not symbol.endswith('W'):
                        if symbol in symbols_dict:
                            # Symbol already exists - update oracle flag
                            symbols_dict[symbol]['oracle'] = True
                        else:
                            # Add new symbol
                            symbols_dict[symbol] = {
                                'oracle': True,
                                'manual': False,
                                'top_gainers': False,
                                'surge': False,
                                'gain_percent': None,
                                'surge_amount': None
                            }
        except Exception as e:
            print(f"Error loading oracle CSV {oracle_csv}: {e}", file=sys.stderr)
    else:
        print(f"Oracle CSV not found: {oracle_csv}", file=sys.stderr)

    # Convert dict to list with symbol field
    symbol_list = []
    for symbol, sources in symbols_dict.items():
        symbol_info = {
            'symbol': symbol,
            **sources
        }
        symbol_list.append(symbol_info)

    # Sort by symbol name
    symbol_list.sort(key=lambda x: x['symbol'])

    return symbol_list


def get_watch_list_from_momentum_alerts() -> dict:
    """
    Check for watch list JSON exported by momentum_alerts.py.

    Returns:
        Dictionary with watch list data if available and recent, None otherwise
    """
    et_tz = pytz.timezone('US/Eastern')
    today = datetime.now(et_tz).strftime('%Y-%m-%d')

    # Path to the JSON file exported by momentum_alerts.py
    watch_list_json = Path(project_root) / "historical_data" / today / "scanner" / "watch_list.json"

    if not watch_list_json.exists():
        print(f"Watch list JSON not found: {watch_list_json}", file=sys.stderr)
        return None

    try:
        # Check file age (should be less than 2 minutes old to be considered fresh)
        file_mtime = datetime.fromtimestamp(watch_list_json.stat().st_mtime, et_tz)
        age_seconds = (datetime.now(et_tz) - file_mtime).total_seconds()

        if age_seconds > 120:  # 2 minutes
            print(f"Watch list JSON is stale ({age_seconds:.0f}s old), falling back to CSV", file=sys.stderr)
            return None

        # Read and parse the JSON file
        with open(watch_list_json, 'r') as f:
            data = json.load(f)

        if not data.get('success'):
            print("Watch list JSON indicates failure, falling back to CSV", file=sys.stderr)
            return None

        print(f"Using watch list from momentum_alerts.py (age: {age_seconds:.0f}s)", file=sys.stderr)
        return data

    except Exception as e:
        print(f"Error reading watch list JSON: {e}, falling back to CSV", file=sys.stderr)
        return None


def main():
    """Main CGI entry point."""
    # Try to get watch list from momentum_alerts.py JSON export first
    response = get_watch_list_from_momentum_alerts()

    # Fall back to reading CSV files directly if JSON not available
    if response is None:
        watch_list = get_watch_list_symbols()
        response = {
            'success': True,
            'data': watch_list,
            'count': len(watch_list),
            'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
            'source': 'csv_fallback'
        }
    else:
        # Add source indicator to show data came from momentum_alerts.py
        response['source'] = 'momentum_alerts'
        # Rename 'symbols' key to 'data' for API consistency
        if 'symbols' in response:
            response['data'] = response.pop('symbols')

    # Output CGI headers
    print("Content-Type: application/json")
    print("Access-Control-Allow-Origin: *")
    print()

    # Output JSON response
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Error response
        error_response = {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat()
        }

        print("Content-Type: application/json")
        print("Access-Control-Allow-Origin: *")
        print()
        print(json.dumps(error_response, indent=2))

        # Log error to stderr
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
