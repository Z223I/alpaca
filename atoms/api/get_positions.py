from typing import Any, List


def get_positions(api_client: Any) -> List[Any]:
    """
    Get all current positions in the trading account.
    
    Args:
        api_client: Alpaca API client instance
        
    Returns:
        List of position objects from Alpaca API
    """
    return api_client.list_positions()
