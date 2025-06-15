from typing import Any
from atoms.api.get_latest_quote import get_latest_quote


def print_quote(api_client: Any, symbol: str) -> None:
    """
    Print the latest quote information for a symbol.
    
    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get quote for
    """
    latest_quote = get_latest_quote(api_client, symbol)
    print("Symbol:", symbol)
    print("Bid price:", latest_quote.bid_price)
    print("Ask price:", latest_quote.ask_price)
