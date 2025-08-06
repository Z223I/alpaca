from typing import Any
from atoms.api.get_latest_quote import get_latest_quote


def print_quote(api_client: Any, symbol: str, account_name: str = None, environment: str = None) -> None:
    """
    Print the latest quote information for a symbol.

    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get quote for
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)
    """
    latest_quote = get_latest_quote(api_client, symbol, account_name, environment)
    if latest_quote is None:
        print(f"Unable to get quote for {symbol} due to authentication error")
        return
    print("Symbol:", symbol)
    print("Bid price:", latest_quote.bid_price)
    print("Ask price:", latest_quote.ask_price)
