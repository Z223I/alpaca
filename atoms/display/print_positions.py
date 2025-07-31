from typing import Any
from atoms.api.get_positions import get_positions


def print_positions(api_client: Any) -> None:
    """
    Print all current positions to console with vital info only.
    
    Shows: Symbol, Quantity, Average Fill Price
    
    Args:
        api_client: Alpaca API client instance
    """
    positions = get_positions(api_client)
    
    if not positions:
        print("positions: No open positions")
        return
    
    print("positions:")
    print("  Symbol    Qty      Avg Fill")
    print("  ────────  ──────── ────────")
    for position in positions:
        symbol = position.symbol
        qty = position.qty
        avg_price = float(position.avg_entry_price)
        
        # Format quantity (show as integer if whole number, otherwise show 2 decimals)
        if float(qty) == int(float(qty)):
            qty_str = f"{int(float(qty)):>8}"
        else:
            qty_str = f"{float(qty):>8.2f}"
        
        print(f"  {symbol:<8} {qty_str} ${avg_price:>7.2f}")
