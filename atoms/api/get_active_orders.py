from typing import Any, List


def get_active_orders(api_client: Any) -> List[Any]:
    """
    Retrieve all active (open) orders from Alpaca.

    Args:
        api_client: Alpaca API client instance

    Returns:
        List of active order objects, or empty dict if an error occurs
    """
    try:
        return api_client.list_orders(
            status='open',
            limit=100)
    except:
        return []
