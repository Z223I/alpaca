from typing import Any
from atoms.api.get_latest_quote import get_latest_quote


def get_latest_quote_avg(api_client: Any, symbol: str) -> float:
    """
    Get the average of bid and ask prices for a given symbol.

    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get quote for

    Returns:
        Average price of bid and ask ((bid_price + ask_price) / 2)
    """
    quote = get_latest_quote(api_client, symbol)
    return (quote.bid_price + quote.ask_price) / 2