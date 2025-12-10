from typing import Any
from .api_error_handler import handle_alpaca_api_error


def get_latest_trade(api_client: Any, symbol: str, account_name: str = None, environment: str = None) -> Any:
    """
    Get the latest trade for a given symbol.

    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get trade for
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)

    Returns:
        Latest trade object from Alpaca API with price and other trade details
    """
    result = handle_alpaca_api_error(
        api_client.get_latest_trade,
        symbol,
        account_name=account_name,
        environment=environment
    )
    # Return None if authentication failed
    return result
