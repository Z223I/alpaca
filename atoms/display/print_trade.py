from typing import Any
from atoms.api.get_latest_trade import get_latest_trade


def print_trade(api_client: Any, symbol: str, account_name: str = None, environment: str = None) -> None:
    """
    Print the latest trade information for a symbol.

    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get trade for
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)
    """
    latest_trade = get_latest_trade(api_client, symbol, account_name, environment)
    if latest_trade is None:
        print(f"Unable to get trade for {symbol} due to authentication error")
        return
    print("Symbol:", symbol)
    print("Last trade price:", latest_trade.price)
    print("Last trade size:", latest_trade.size)
    print("Last trade time:", latest_trade.timestamp)
