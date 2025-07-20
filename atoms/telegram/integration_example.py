"""
Example integration of Telegram notifications with existing Alpaca trading code.

This demonstrates how to add Telegram notifications to existing trading operations
without breaking the existing functionality.
"""

from .telegram_post import send_message, send_alert

def notify_trade_execution(symbol: str, side: str, quantity: int, price: float, total_value: float):
    """Send notification when a trade is executed."""
    try:
        message = f"🎯 Trade Executed: {side.upper()} {quantity} shares of {symbol} @ ${price:.2f} (Total: ${total_value:.2f})"
        result = send_message(message)
        
        if result['success']:
            print(f"Telegram notification sent to {result['sent_count']} users")
        else:
            print(f"Telegram notification failed: {result['errors']}")
            
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")

def notify_bracket_order(symbol: str, quantity: int, entry_price: float, stop_price: float, target_price: float):
    """Send notification when a bracket order is submitted."""
    try:
        stop_loss_pct = ((entry_price - stop_price) / entry_price) * 100
        take_profit_pct = ((target_price - entry_price) / entry_price) * 100
        
        message = (
            f"🔄 Bracket Order Submitted\n"
            f"📊 {symbol}: {quantity} shares\n"
            f"💲 Entry: ${entry_price:.2f}\n"
            f"🛑 Stop: ${stop_price:.2f} (-{stop_loss_pct:.1f}%)\n"
            f"🎯 Target: ${target_price:.2f} (+{take_profit_pct:.1f}%)"
        )
        
        result = send_message(message)
        
        if result['success']:
            print(f"Bracket order notification sent to {result['sent_count']} users")
        else:
            print(f"Bracket order notification failed: {result['errors']}")
            
    except Exception as e:
        print(f"Error sending bracket order notification: {e}")

def notify_error(error_type: str, error_message: str, symbol: str = None):
    """Send error notification."""
    try:
        symbol_text = f" for {symbol}" if symbol else ""
        message = f"❌ Trading Error{symbol_text}: {error_type} - {error_message}"
        
        send_alert(message, level='error')
        print(f"Error notification sent via Telegram")
        
    except Exception as e:
        print(f"Error sending error notification: {e}")

# Example of how to integrate with existing alpaca.py methods:
"""
In alpaca.py _buy method, add after successful order execution:

    if submit and order_filled:
        from atoms.telegram.integration_example import notify_trade_execution
        notify_trade_execution(symbol, 'buy', quantity, filled_price, total_value)

In alpaca.py _bracketOrder method, add after order submission:

    if submit and order_submitted:
        from atoms.telegram.integration_example import notify_bracket_order
        notify_bracket_order(symbol, quantity, market_price, stop_price, take_profit_price)
"""