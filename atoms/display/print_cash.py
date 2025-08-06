from typing import Any
from atoms.api.get_cash import get_cash


def print_cash(api_client: Any, account_name: str = None, environment: str = None) -> None:
    """
    Print the current cash balance to console.

    Args:
        api_client: Alpaca API client instance
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)
    """
    cash = get_cash(api_client, account_name, environment)
    print(f"cash: {cash}")
