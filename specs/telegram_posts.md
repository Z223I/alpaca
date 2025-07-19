# Telegram Posts Atom Specification

## Overview

The Telegram Posts atom provides functionality to send trading notifications and alerts to Telegram channels/chats via the Telegram Bot API. This atom integrates with the existing alpaca trading system to provide real-time notifications for trades, portfolio updates, and market alerts.

## Architecture

### Location
- **Module**: `atoms/telegram/`
- **Main File**: `atoms/telegram/telegram_posts.py`
- **Config File**: `atoms/telegram/config.py`
- **Test File**: `tests/test_telegram_posts.py`

### Dependencies
```python
# Required packages to add to requirements
python-telegram-bot>=20.0
requests>=2.28.0
asyncio>=3.4.3
```

## Core Functions

### `send_trade_notification(order_details, chat_id=None)`
Sends formatted trade execution notifications.

**Parameters:**
- `order_details` (dict): Order information from Alpaca API
- `chat_id` (str, optional): Target chat ID (defaults to env TELEGRAM_CHAT_ID)

**Returns:** `bool` - Success status

**Example Usage:**
```python
from atoms.telegram.telegram_posts import send_trade_notification

order_details = {
    'symbol': 'AAPL',
    'side': 'buy',
    'qty': 10,
    'filled_avg_price': 150.25,
    'status': 'filled'
}
send_trade_notification(order_details)
```

### `send_portfolio_update(portfolio_data, chat_id=None)`
Sends daily/periodic portfolio summary.

**Parameters:**
- `portfolio_data` (dict): Portfolio metrics and positions
- `chat_id` (str, optional): Target chat ID

**Message Format:**
```
📊 Portfolio Update
💰 Total Value: $12,450.30 (+2.3%)
💵 Cash: $2,100.00
📈 Day P&L: +$287.50 (+2.37%)

Top Positions:
• AAPL: 10 shares @ $150.25 (+$45.00)
• TSLA: 5 shares @ $220.00 (-$12.50)
```

### `send_alert(message, level='info', chat_id=None)`
Sends custom alerts and notifications.

**Parameters:**
- `message` (str): Alert message content
- `level` (str): Alert level ('info', 'warning', 'error', 'success')
- `chat_id` (str, optional): Target chat ID

**Level Formatting:**
- `info`: ℹ️ blue text
- `warning`: ⚠️ yellow text  
- `error`: ❌ red text
- `success`: ✅ green text

### `send_orb_signal(signal_data, chat_id=None)`
Sends Opening Range Breakout signals.

**Parameters:**
- `signal_data` (dict): ORB signal information
- `chat_id` (str, optional): Target chat ID

**Message Format:**
```
🚀 ORB Breakout Signal
📊 AAPL
📈 Direction: LONG
💲 Entry: $152.50
🛑 Stop: $145.00 (-4.9%)
🎯 Target: $165.00 (+8.2%)
⏰ Time: 09:45 ET
```

## Configuration

### Environment Variables
```env
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_default_chat_id

# Optional
TELEGRAM_ENABLED=true
TELEGRAM_PARSE_MODE=HTML  # or Markdown
TELEGRAM_DISABLE_NOTIFICATION=false
TELEGRAM_TIMEOUT=30
```

### Configuration Class
```python
# atoms/telegram/config.py
class TelegramConfig:
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    DEFAULT_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID') 
    ENABLED = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'
    PARSE_MODE = os.getenv('TELEGRAM_PARSE_MODE', 'HTML')
    TIMEOUT = int(os.getenv('TELEGRAM_TIMEOUT', '30'))
```

## Integration Points

### Main Alpaca Class Integration
```python
# In code/alpaca.py alpaca_private class

def _buy(self, symbol, submit=False):
    # Existing buy logic...
    
    if submit and order_filled:
        # Send Telegram notification
        from atoms.telegram.telegram_posts import send_trade_notification
        send_trade_notification({
            'symbol': symbol,
            'side': 'buy',
            'qty': quantity,
            'filled_avg_price': filled_price,
            'status': 'filled'
        })

def _bracketOrder(self, symbol, quantity, market_price, submit=False):
    # Existing bracket order logic...
    
    if submit and order_submitted:
        from atoms.telegram.telegram_posts import send_alert
        send_alert(f"🔄 Bracket order submitted for {symbol}", level='info')
```

### Command Line Integration
```python
# Add new CLI arguments to parse_args in atoms/utils/
'--telegram-notify': Enable Telegram notifications for this operation
'--telegram-chat': Override default chat ID for this message
```

## Error Handling

### Retry Logic
- Implement exponential backoff for failed API calls
- Maximum 3 retry attempts
- Log failures to application logs

### Fallback Behavior
- If Telegram is unavailable, continue normal operation
- Log notification failures without breaking trades
- Optional email fallback for critical alerts

### Rate Limiting
- Respect Telegram's 30 messages/second limit
- Queue messages if burst sending is needed
- Implement message batching for portfolio updates

## Security Considerations

### Token Protection
- Store bot token in environment variables only
- Never log or print bot token
- Validate token format before API calls

### Chat ID Validation
- Verify chat ID permissions before sending
- Implement whitelist of allowed chat IDs
- Handle unauthorized chat scenarios gracefully

## Testing Strategy

### Unit Tests
```python
# tests/test_telegram_posts.py
class TestTelegramPosts(unittest.TestCase):
    def test_send_trade_notification_success(self):
        # Mock successful API response
        
    def test_send_trade_notification_failure(self):
        # Mock API failure and retry logic
        
    def test_message_formatting(self):
        # Verify message format for different order types
        
    def test_rate_limiting(self):
        # Test message queuing under high volume
```

### Integration Tests
- Test with real Telegram bot (test chat)
- Verify message delivery timing
- Test error scenarios (invalid tokens, network issues)

## Message Templates

### Trade Execution
```html
<b>🎯 Trade Executed</b>
📊 Symbol: {symbol}
📈 Side: {side.upper()}
💯 Quantity: {qty}
💲 Price: ${filled_avg_price:.2f}
⏰ Time: {timestamp}
💰 Total: ${total_value:.2f}
```

### Portfolio Summary
```html
<b>📊 Daily Portfolio Summary</b>
💰 Total Value: ${total_value:.2f} ({pnl_percent:+.2f}%)
💵 Cash Available: ${cash:.2f}
📈 Day P&L: {day_pnl:+.2f} ({day_pnl_percent:+.2f}%)

<b>📋 Positions ({position_count}):</b>
{position_list}
```

### Error Alerts
```html
<b>❌ Trading Error</b>
🚨 {error_type}
📝 Details: {error_message}
⏰ Time: {timestamp}
🔧 Action Required: {suggested_action}
```

## Performance Considerations

### Async Implementation
- Use async/await for non-blocking message sending
- Don't delay trade execution for notification delivery
- Implement background message queue

### Memory Management
- Limit message history retention
- Clean up completed message queues
- Monitor memory usage for high-frequency trading

## Future Enhancements

### Advanced Features
1. **Interactive Commands**: Allow querying portfolio via Telegram
2. **Chart Integration**: Send trading charts and technical analysis
3. **Alert Subscriptions**: User-configurable notification preferences
4. **Multi-Chat Support**: Send different message types to different chats
5. **Message Threading**: Group related messages for better organization

### Analytics Integration
- Track notification delivery rates
- Monitor user engagement with messages
- A/B test message formats for effectiveness