from typing import Any
from atoms.api.get_active_orders import get_active_orders


def print_active_orders(api_client: Any) -> None:
    """
    Print details of all active orders to console.
    
    Displays order information including symbol, quantity, side, status,
    filled quantity, remaining quantity, and timestamps.
    
    Args:
        api_client: Alpaca API client instance
    """
    orders = get_active_orders(api_client)

    if orders:
        print(f"orders: {orders}")

        for order in orders:
            print(f"order: {order}")
            print(f"symbol: {order.symbol}")
            print(f"qty: {order.qty}")
            print(f"side: {order.side}")
            print(f"status: {order.status}")
            print(f"filled_qty: {order.filled_qty}")
            print(f"remaining_qty: {order.remaining_qty}")
            print(f"created_at: {order.created_at}")
            print(f"updated_at: {order.updated_at}")
    else:
        print("No current orders")
