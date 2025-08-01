from typing import Any
from atoms.api.get_cash import get_cash


def print_cash(api_client: Any) -> None:
    """
    Print the current cash balance to console.

    Args:
        api_client: Alpaca API client instance
    """
    cash = get_cash(api_client)
    print(f"cash: {cash}")
