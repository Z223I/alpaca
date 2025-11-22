#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
"""
Stock Fundamentals API

Provides stock fundamental data using yfinance.
Returns: Shares Outstanding, Float Shares, and Market Cap.
"""

import json
import sys
import os
import yfinance as yf
from datetime import datetime

# Add project root to path
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


def format_large_number(num):
    """Format large numbers with K, M, B, T suffixes."""
    if num is None or num == 0:
        return "N/A"

    try:
        num = float(num)
        if abs(num) >= 1e12:
            return f"{num/1e12:.2f}T"
        elif abs(num) >= 1e9:
            return f"{num/1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return f"{num:.2f}"
    except (ValueError, TypeError):
        return "N/A"


def get_stock_fundamentals(symbol):
    """
    Fetch stock fundamentals using yfinance.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with fundamental data
    """
    try:
        # Fetch stock info
        stock = yf.Ticker(symbol)
        info = stock.info

        # Extract key fundamentals
        shares_outstanding = info.get('sharesOutstanding', None)
        float_shares = info.get('floatShares', None)
        market_cap = info.get('marketCap', None)

        # Additional useful data
        company_name = info.get('longName', info.get('shortName', symbol))
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')

        return {
            'success': True,
            'symbol': symbol,
            'company_name': company_name,
            'shares_outstanding': shares_outstanding,
            'shares_outstanding_formatted': format_large_number(shares_outstanding),
            'float_shares': float_shares,
            'float_shares_formatted': format_large_number(float_shares),
            'market_cap': market_cap,
            'market_cap_formatted': format_large_number(market_cap),
            'sector': sector,
            'industry': industry,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'symbol': symbol
        }


def main():
    """Main entry point for CGI script."""
    try:
        params = get_query_params()
        symbol = params.get('symbol', '').upper().strip()

        if not symbol:
            print(json.dumps({
                'success': False,
                'error': 'Missing required parameter: symbol'
            }))
            return

        # Get fundamentals
        result = get_stock_fundamentals(symbol)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Server error: {str(e)}'
        }))


if __name__ == '__main__':
    main()
