from typing import Any
from atoms.api.get_positions import get_positions


def print_positions(api_client: Any) -> None:
    """
    Print all current positions to console.
    
    Args:
        api_client: Alpaca API client instance
    """
    positions = get_positions(api_client)
    print(f"positions: {positions}")
