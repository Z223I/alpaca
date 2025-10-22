"""
Display top market gainers to console.

This atom provides formatted output for top gainers
from the Alpaca screener API.
"""

from typing import Any, Optional, Dict
from atoms.api.get_top_movers import get_top_movers


def print_top_movers(api_client: Any, market_type: str = 'stocks', top: int = 10,
                     account_name: str = None, environment: str = None) -> bool:
    """
    Display top market gainers to console.

    Args:
        api_client: Alpaca API client instance
        market_type: Market type - 'stocks' or 'crypto' (default: 'stocks')
        top: Number of top gainers to display (default: 10)
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)

    Returns:
        True if successful, False otherwise
    """
    # Fetch the data using the API atom
    data = get_top_movers(api_client, market_type, top, account_name, environment)

    if data is None:
        print(f"✗ Failed to fetch {market_type} movers")
        return False

    # Extract gainers only
    gainers = data.get('gainers', [])

    # Display results
    print("=" * 80)
    print(f"🟢 TOP GAINERS ({market_type.upper()})")
    print("=" * 80)

    if gainers:
        print(f"{'Rank':<6} {'Symbol':<10} {'Price':<12} {'Change':<12} {'% Change':<12}")
        print("-" * 80)
        for idx, mover in enumerate(gainers, 1):
            symbol = mover.get('symbol', 'N/A')
            price = mover.get('price', 0)
            change = mover.get('change', 0)
            percent_change = mover.get('percent_change', 0)

            print(f"{idx:<6} {symbol:<10} ${price:<11.2f} ${change:<11.2f} "
                  f"{percent_change:>10.2f}%")
    else:
        print("No gainers data available")

    print("=" * 80)

    # Display metadata if available
    if 'last_updated' in data:
        print(f"\nLast Updated: {data['last_updated']}")

    return True
