import time
from typing import Any
from atoms.api.get_active_orders import get_active_orders


def delay(api_client: Any) -> None:
    """
    Wait until all active orders are completed.
    
    Continuously polls for active orders and sleeps until none remain.
    
    Args:
        api_client: Alpaca API client instance
    """
    while len(get_active_orders(api_client)) > 0:
        time.sleep(1)
