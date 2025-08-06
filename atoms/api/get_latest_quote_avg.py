from typing import Any
from atoms.api.get_latest_quote import get_latest_quote


def get_latest_quote_avg(api_client: Any, symbol: str, account_name: str = None, environment: str = None) -> float:
    """
    Get the average of bid and ask prices for a given symbol.

    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get quote for
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)

    Returns:
        Average price of bid and ask ((bid_price + ask_price) / 2)
    """
    quote = get_latest_quote(api_client, symbol, account_name, environment)
    # Return 0.0 if authentication failed
    if quote is None:
        return 0.0
    return (quote.bid_price + quote.ask_price) / 2