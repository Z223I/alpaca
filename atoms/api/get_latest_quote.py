from typing import Any
from .api_error_handler import handle_alpaca_api_error


def get_latest_quote(api_client: Any, symbol: str, account_name: str = None, environment: str = None) -> Any:
    """
    Get the latest quote for a given symbol.

    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get quote for
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)

    Returns:
        Latest quote object from Alpaca API
    """
    result = handle_alpaca_api_error(
        api_client.get_latest_quote,
        symbol,
        account_name=account_name,
        environment=environment
    )
    # Return None if authentication failed
    return result
