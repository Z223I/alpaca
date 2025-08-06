from typing import Any
from .api_error_handler import handle_alpaca_api_error


def get_cash(api_client: Any, account_name: str = None, environment: str = None) -> float:
    """
    Get the current cash balance in the trading account.

    Args:
        api_client: Alpaca API client instance
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)

    Returns:
        Current cash balance as a float
    """
    account_info = handle_alpaca_api_error(
        api_client.get_account, 
        account_name=account_name, 
        environment=environment
    )
    # Return 0.0 if authentication failed
    if account_info is None:
        return 0.0
    return float(account_info.cash)
