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
            print(f"created_at: {order.created_at}")
            print(f"updated_at: {order.updated_at}")

# 'asset_class': 'us_equity',
#     'asset_id': 'e4870680-5a31-43e0-a477-c47d158ac43d',
#     'canceled_at': None,
#     'client_order_id': '461425ae-7113-42d2-b2b7-2ba248d7c9ed',
#     'created_at': '2025-06-14T21:03:47.083853Z',
#     'expired_at': None,
#     'expires_at': '2025-06-16T20:00:00Z',
#     'extended_hours': False,
#     'failed_at': None,
#     'filled_at': None,
#     'filled_avg_price': None,
#     'filled_qty': '0',
#     'hwm': None,
#     'id': 'c3245876-8e83-4b58-9df0-b547402fc3d1',
#     'legs': None,
#     'limit_price': None,
#     'notional': None,
#     'order_class': '',
#     'order_type': 'market',
#     'position_intent': 'sell_to_close',
#     'qty': '227.23187088',
#     'replaced_at': None,
#     'replaced_by': None,
#     'replaces': None,
#     'side': 'sell',
#     'source': None,
#     'status': 'new',
#     'stop_price': None,
#     'submitted_at': '2025-06-16T08:02:00.573198Z',
#     'subtag': None,
#     'symbol': 'SMCI',
#     'time_in_force': 'day',
#     'trail_percent': None,
#     'trail_price': None,
#     'type': 'market',
#     'updated_at': '2025-06-16T08:02:00.574333Z'})
    else:
        print("No current orders")
