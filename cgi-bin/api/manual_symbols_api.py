#!/home/wilsonb/miniconda3/envs/alpaca/bin/python
"""
Manual Symbols API Endpoint

Provides REST API for managing manually added symbols that should be monitored
by momentum_alerts.py. Persists manual symbols to a JSON file.

Usage:
    GET  /api/manual_symbols - Returns list of manual symbols
    POST /api/manual_symbols - Add a manual symbol
        Body: {"symbol": "AAPL"}
    DELETE /api/manual_symbols - Remove a manual symbol
        Body: {"symbol": "AAPL"}

GoDaddy CGI compatible.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytz


# Path to manual symbols JSON file
MANUAL_SYMBOLS_FILE = project_root / "data" / "manual_symbols.json"


def load_manual_symbols() -> Set[str]:
    """
    Load manual symbols from JSON file.

    Returns:
        Set of symbol strings
    """
    if not MANUAL_SYMBOLS_FILE.exists():
        return set()

    try:
        with open(MANUAL_SYMBOLS_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('symbols', []))
    except Exception as e:
        print(f"Error loading manual symbols: {e}", file=sys.stderr)
        return set()


def save_manual_symbols(symbols: Set[str]) -> bool:
    """
    Save manual symbols to JSON file.

    Args:
        symbols: Set of symbol strings

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure data directory exists
        MANUAL_SYMBOLS_FILE.parent.mkdir(parents=True, exist_ok=True)

        et_tz = pytz.timezone('US/Eastern')
        data = {
            'symbols': sorted(list(symbols)),
            'count': len(symbols),
            'last_updated': datetime.now(et_tz).isoformat(),
            'last_updated_timestamp': datetime.now(et_tz).timestamp()
        }

        with open(MANUAL_SYMBOLS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving manual symbols: {e}", file=sys.stderr)
        return False


def send_json_response(data: Dict, status: int = 200):
    """
    Send JSON response with proper CGI headers.

    Args:
        data: Dictionary to send as JSON
        status: HTTP status code
    """
    print("Content-Type: application/json")
    print("Access-Control-Allow-Origin: *")
    print("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS")
    print("Access-Control-Allow-Headers: Content-Type")

    if status != 200:
        print(f"Status: {status}")

    print()  # End of headers
    print(json.dumps(data, indent=2))


def handle_get():
    """Handle GET request - return list of manual symbols."""
    symbols = load_manual_symbols()

    et_tz = pytz.timezone('US/Eastern')
    response = {
        'success': True,
        'symbols': sorted(list(symbols)),
        'count': len(symbols),
        'timestamp': datetime.now(et_tz).isoformat()
    }

    send_json_response(response)


def handle_post():
    """Handle POST request - add a manual symbol."""
    try:
        # Read POST data from stdin
        content_length = int(os.environ.get('CONTENT_LENGTH', 0))

        if content_length == 0:
            send_json_response({
                'success': False,
                'error': 'No data provided'
            }, 400)
            return

        post_data = sys.stdin.read(content_length)
        data = json.loads(post_data)

        symbol = data.get('symbol', '').strip().upper()

        if not symbol:
            send_json_response({
                'success': False,
                'error': 'Missing symbol parameter'
            }, 400)
            return

        # Basic symbol validation
        if not symbol.replace('.', '').replace('-', '').isalnum():
            send_json_response({
                'success': False,
                'error': f'Invalid symbol format: {symbol}'
            }, 400)
            return

        # Load current symbols
        symbols = load_manual_symbols()

        # Check if symbol already exists
        if symbol in symbols:
            send_json_response({
                'success': True,
                'message': f'Symbol {symbol} already exists',
                'symbol': symbol,
                'symbols': sorted(list(symbols)),
                'count': len(symbols)
            })
            return

        # Add symbol
        symbols.add(symbol)

        # Save to file
        if not save_manual_symbols(symbols):
            send_json_response({
                'success': False,
                'error': 'Failed to save manual symbols'
            }, 500)
            return

        et_tz = pytz.timezone('US/Eastern')
        send_json_response({
            'success': True,
            'message': f'Symbol {symbol} added successfully',
            'symbol': symbol,
            'symbols': sorted(list(symbols)),
            'count': len(symbols),
            'timestamp': datetime.now(et_tz).isoformat()
        })

    except json.JSONDecodeError:
        send_json_response({
            'success': False,
            'error': 'Invalid JSON data'
        }, 400)

    except Exception as e:
        send_json_response({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, 500)


def handle_delete():
    """Handle DELETE request - remove a manual symbol."""
    try:
        # Read DELETE data from stdin
        content_length = int(os.environ.get('CONTENT_LENGTH', 0))

        if content_length == 0:
            send_json_response({
                'success': False,
                'error': 'No data provided'
            }, 400)
            return

        delete_data = sys.stdin.read(content_length)
        data = json.loads(delete_data)

        symbol = data.get('symbol', '').strip().upper()

        if not symbol:
            send_json_response({
                'success': False,
                'error': 'Missing symbol parameter'
            }, 400)
            return

        # Load current symbols
        symbols = load_manual_symbols()

        # Check if symbol exists
        if symbol not in symbols:
            send_json_response({
                'success': True,
                'message': f'Symbol {symbol} does not exist',
                'symbol': symbol,
                'symbols': sorted(list(symbols)),
                'count': len(symbols)
            })
            return

        # Remove symbol
        symbols.discard(symbol)

        # Save to file
        if not save_manual_symbols(symbols):
            send_json_response({
                'success': False,
                'error': 'Failed to save manual symbols'
            }, 500)
            return

        et_tz = pytz.timezone('US/Eastern')
        send_json_response({
            'success': True,
            'message': f'Symbol {symbol} removed successfully',
            'symbol': symbol,
            'symbols': sorted(list(symbols)),
            'count': len(symbols),
            'timestamp': datetime.now(et_tz).isoformat()
        })

    except json.JSONDecodeError:
        send_json_response({
            'success': False,
            'error': 'Invalid JSON data'
        }, 400)

    except Exception as e:
        send_json_response({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, 500)


def handle_options():
    """Handle OPTIONS request for CORS preflight."""
    send_json_response({'success': True})


def main():
    """Main CGI entry point."""
    request_method = os.environ.get('REQUEST_METHOD', 'GET')

    if request_method == 'GET':
        handle_get()
    elif request_method == 'POST':
        handle_post()
    elif request_method == 'DELETE':
        handle_delete()
    elif request_method == 'OPTIONS':
        handle_options()
    else:
        send_json_response({
            'success': False,
            'error': f'Method {request_method} not allowed'
        }, 405)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_json_response({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }, 500)

        # Log error to stderr
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
