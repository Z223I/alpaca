from typing import Any, List
from .api_error_handler import handle_alpaca_api_error


def get_positions(api_client: Any, account_name: str = None, environment: str = None) -> List[Any]:
    """
    Get all current positions in the trading account.

    Args:
        api_client: Alpaca API client instance
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)

    Returns:
        List of position objects from Alpaca API
    """
    result = handle_alpaca_api_error(
        api_client.list_positions,
        account_name=account_name,
        environment=environment
    )
    # Return empty list if authentication failed
    return result if result is not None else []
