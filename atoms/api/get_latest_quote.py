from typing import Any


def get_latest_quote(api_client: Any, symbol: str) -> Any:
    """
    Get the latest quote for a given symbol.
    
    Args:
        api_client: Alpaca API client instance
        symbol: Stock symbol to get quote for
        
    Returns:
        Latest quote object from Alpaca API
    """
    return api_client.get_latest_quote(symbol)
