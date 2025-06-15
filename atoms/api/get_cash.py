from typing import Any


def get_cash(api_client: Any) -> float:
    """
    Get the current cash balance in the trading account.
    
    Args:
        api_client: Alpaca API client instance
        
    Returns:
        Current cash balance as a float
    """
    return api_client.get_account().cash
