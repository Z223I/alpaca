# Time and Sales WebSocket System

## Overview

The Time and Sales system provides real-time stock trade data streaming from Alpaca to the browser interface. It uses a multi-tier WebSocket architecture for reliable, scalable trade data delivery.

## Architecture

```
Browser (public_html/index.html)
    |
    | WebSocket ws://localhost:8766
    v
Proxy Server (services/browser_proxy.py)
    |
    | WebSocket ws://localhost:8765
    v
Backend Server (services/trade_stream_server.py)
    |
    | Alpaca WebSocket API (SIP Feed)
    v
Alpaca Markets Real-time Trade Stream
```

### Components

#### 1. Frontend Client (`public_html/index.html`)

**Connection URL**: `ws://localhost:8766` (or `ws://<hostname>:8766`)

**Key Functions**:
- `initializeTradeWebSocket()` (line 2033) - Establishes WebSocket connection to the proxy server
- `subscribeToSymbol(symbol)` (line 2092) - Subscribes to real-time trades for a specific stock symbol
- `unsubscribeFromSymbol(symbol)` (line 2111) - Unsubscribes from a symbol's trade stream

**WebSocket State**: Stored in `state.tradeWebSocket` (line 1923)

**Connection Features**:
- Automatic reconnection with exponential backoff (line 2061-2071)
- Heartbeat ping every 30 seconds to maintain connection (line 2080-2081)
- Auto-subscribe to high-performing stocks (gain > 30%, surge > 5x) (line 4539)

**Message Protocol**:

Outbound (Browser → Server):
```javascript
// Subscribe to a symbol
{ action: 'subscribe', symbol: 'AAPL' }

// Unsubscribe from a symbol
{ action: 'unsubscribe', symbol: 'AAPL' }

// Health check
{ action: 'ping' }

// Get server status
{ action: 'health' }
```

Inbound (Server → Browser):
```javascript
// Trade data
{
  type: 'trade',
  symbol: 'AAPL',
  data: {
    timestamp: '2025-12-14T10:30:45.123Z',
    price: 150.25,
    size: 100,
    exchange: 'NASDAQ',
    conditions: []
  }
}

// Subscription confirmation
{ type: 'subscribed', symbol: 'AAPL', message: 'Subscribed to AAPL trades' }

// Unsubscription confirmation
{ type: 'unsubscribed', symbol: 'AAPL', message: 'Unsubscribed from AAPL trades' }

// Connection status
{ type: 'connecting', symbol: 'AAPL', message: 'Connecting to Alpaca stream...' }

// Ping response
{ type: 'pong', alpaca_connected: true }

// Health status
{
  type: 'health',
  alpaca_connected: true,
  active_symbols: ['AAPL', 'TSLA'],
  total_clients: 3
}
```

#### 2. Proxy Server (`services/browser_proxy.py`)

**Purpose**: Provides a browser-compatible WebSocket interface and forwards messages between browser and backend

**Listen Port**: 8766
**Backend URL**: `ws://localhost:8765`

**Features**:
- Bidirectional message forwarding
- Connection pooling for multiple browser clients
- Automatic cleanup of disconnected clients
- Reliable Python websockets library for browser compatibility

**Class**: `ProxyServer`
- `handle_browser_client()` - Handles new browser connections and creates backend connection
- `_forward_messages()` - Forwards messages bidirectionally between browser and backend

#### 3. Backend Trade Stream Server (`services/trade_stream_server.py`)

**Purpose**: Maintains persistent connection to Alpaca's real-time trade stream and broadcasts to connected clients

**Listen Port**: 8765
**Data Source**: Alpaca StockDataStream API (SIP Feed)

**Environment Variables**:
- `ALPACA_API_KEY` - Required Alpaca API key
- `ALPACA_SECRET_KEY` - Required Alpaca secret key
- `TRADE_STREAM_HOST` - Server host (default: 0.0.0.0)
- `TRADE_STREAM_PORT` - Server port (default: 8765)

**Data Feed**: DataFeed.SIP (Securities Information Processor)
- Includes all exchanges: NYSE, NASDAQ, FINRA, Cboe, etc.
- Requires paid Alpaca subscription for real-time data
- Alternative: DataFeed.IEX for free IEX-only data (line 78-82)

**Class**: `TradeStreamServer`

Key Methods:
- `start()` - Initialize and start the WebSocket server
- `handle_client()` - Handle new WebSocket client connections
- `subscribe_symbol()` - Subscribe a client to a symbol's trades
- `unsubscribe_symbol()` - Unsubscribe a client from a symbol
- `_subscribe_alpaca_symbol()` - Subscribe to Alpaca's stream for a symbol
- `_unsubscribe_alpaca_symbol()` - Unsubscribe from Alpaca's stream
- `broadcast_trade()` - Broadcast received trade to all subscribed clients
- `_run_alpaca_stream()` - Maintain Alpaca connection with auto-reconnect

**Reconnection Strategy**:
- Automatic reconnection on connection loss (line 94-148)
- Exponential backoff (1s → 2s → 4s → ... → 60s max)
- Automatic re-subscription to all active symbols after reconnect
- 5-second delay after connection to allow WebSocket authentication (line 108)

**Subscription Management**:
- Lazy start: Alpaca stream only starts on first subscription (line 208)
- Symbol-level subscriptions: Only subscribes to Alpaca for symbols with active clients
- Automatic cleanup: Unsubscribes from Alpaca when last client unsubscribes
- Thread executor for non-blocking subscription calls (line 318-324)

**Logging**: Comprehensive logging with color-coded prefixes
- `🔵 [SUBSCRIBE]` - Client subscription events
- `🔴 [UNSUBSCRIBE]` - Client unsubscription events
- `🟢 [ALPACA_SUB]` - Alpaca stream subscription events
- `🔔` - Incoming trade from Alpaca
- `📊` - Broadcasting trade to clients

## Starting the Services

### Method 1: Using the start script (recommended)

```bash
./services/start_websocket_services.sh
```

This script:
1. Stops any existing instances
2. Activates the conda `alpaca` environment
3. Starts the backend server on port 8765
4. Starts the proxy server on port 8766
5. Verifies both servers are running
6. Displays status and log locations

**Logs**:
- Backend: `/tmp/trade_backend.log`
- Proxy: `/tmp/trade_proxy.log`

### Method 2: Manual start

```bash
# Terminal 1 - Backend server
conda activate alpaca
python3 services/trade_stream_server.py

# Terminal 2 - Proxy server
conda activate alpaca
python3 services/browser_proxy.py
```

## Stopping the Services

```bash
pkill -f trade_stream_server.py
pkill -f browser_proxy.py
```

Or use the restart script:
```bash
./services/restart_websocket_services.sh
```

## Testing the Connection

### Using the test page

```bash
# Open in browser
open public_html/test_ws.html
# or
open public_html/test_live.html
```

### Using Python client

```bash
python3 test_websocket_reconnect.py
```

## Trade Data Format

Trades received from Alpaca are broadcast to clients with the following structure:

```python
{
    'type': 'trade',
    'symbol': 'AAPL',           # Stock symbol
    'data': {
        'timestamp': '2025-12-14T10:30:45.123456',  # ISO format
        'price': 150.25,         # Trade price (float)
        'size': 100,             # Trade volume (int)
        'exchange': 'NASDAQ',    # Exchange code (if available)
        'conditions': []         # Trade conditions (if available)
    }
}
```

## Subscription Workflow

1. Browser calls `subscribeToSymbol('AAPL')`
2. Browser sends `{ action: 'subscribe', symbol: 'AAPL' }` to proxy (port 8766)
3. Proxy forwards to backend (port 8765)
4. Backend:
   - Adds client to symbol's subscription list
   - If first subscriber for symbol, subscribes to Alpaca stream
   - Sends confirmation back to client
5. When trades arrive from Alpaca:
   - Backend receives trade via Alpaca's WebSocket
   - Backend broadcasts trade to all subscribed clients
   - Proxy forwards to browser
   - Browser displays in UI

## Performance Considerations

- **Lazy Loading**: Alpaca stream only starts when first client subscribes
- **Efficient Broadcasting**: Trades only sent to clients subscribed to that symbol
- **Connection Pooling**: Single Alpaca connection serves all browser clients
- **Automatic Cleanup**: Inactive subscriptions removed automatically
- **Reconnection Logic**: Both servers handle disconnections gracefully

## Dependencies

Python packages (installed via conda/pip):
- `websockets` - WebSocket server/client library
- `alpaca-py` - Alpaca trading API (`alpaca.data.live.StockDataStream`)
- `python-dotenv` - Environment variable management

Browser:
- Native WebSocket API (no external libraries required)

## Troubleshooting

### Connection Issues

1. Check if servers are running:
   ```bash
   lsof -i :8765  # Backend
   lsof -i :8766  # Proxy
   ```

2. Check logs:
   ```bash
   tail -f /tmp/trade_backend.log
   tail -f /tmp/trade_proxy.log
   ```

3. Verify Alpaca credentials in `.env`:
   ```bash
   grep ALPACA_ .env
   ```

### No Trades Received

1. Check Alpaca connection status:
   - Send `{ action: 'health' }` message
   - Look for `alpaca_connected: true` in response

2. Verify market is open (trades only flow during market hours)

3. Check subscription status in backend logs (look for `🟢 [ALPACA_SUB]`)

4. Ensure you have a paid Alpaca account for SIP feed (or switch to IEX feed in line 82)

### High Latency

1. Check if multiple reconnections are occurring (indicates network issues)
2. Verify server load with `htop` or `top`
3. Monitor WebSocket message queue sizes in logs
4. Consider upgrading Alpaca data feed subscription

## Security Notes

- Backend binds to `0.0.0.0` (all interfaces) - restrict in production
- No authentication on WebSocket connections - add auth layer for production
- API keys stored in `.env` file - ensure proper file permissions (600)
- Proxy prevents direct browser access to Alpaca credentials

## Future Enhancements

Potential improvements:
- Authentication/authorization for WebSocket connections
- Rate limiting per client
- Historical trade data caching
- Multiple symbol subscription in single message
- Trade aggregation/batching for high-volume symbols
- Metrics/monitoring dashboard
- Redis pub/sub for horizontal scaling
