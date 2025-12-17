# Restarting WebSocket Services

## Overview

The Market Sentinel system uses two WebSocket services that work together:

1. **Trade Stream Server** (port 8765) - Connects to Alpaca's data stream and manages subscriptions
2. **Browser Proxy** (port 8766) - Forwards connections from browsers to the backend server

These services may occasionally need to be restarted if they become unresponsive or hung.

## When to Restart

Restart the services if you encounter any of these symptoms:

- Health check times out: `python test_health.py` fails with timeout or connection refused
- WebSocket connections from browser clients fail or hang
- Backend not responding to subscription requests
- Alpaca stream connection stuck in disconnected state
- High CPU usage or memory leaks in the service processes

## How to Restart

### Quick Restart

Simply run the restart script from the repository root:

```bash
./services/restart_websocket_services.sh
```

This will:
1. Stop any existing service processes
2. Wait for clean shutdown (2 seconds)
3. Start the trade stream server
4. Wait for backend initialization (3 seconds)
5. Start the browser proxy
6. Display process IDs and log locations

### Manual Restart

If the script doesn't work, you can manually restart:

```bash
# Stop services
pkill -f "trade_stream_server.py"
pkill -f "browser_proxy.py"
sleep 2

# Start backend
python3 services/trade_stream_server.py > /tmp/trade_stream.log 2>&1 &

# Wait for backend to initialize
sleep 3

# Start proxy
python3 services/browser_proxy.py > /tmp/browser_proxy.log 2>&1 &
```

## Verifying the Restart

After restarting, verify the services are healthy:

### 1. Check Process Status

```bash
ps aux | grep -E "(browser_proxy|trade_stream_server)" | grep -v grep
```

You should see two Python processes running.

### 2. Check Port Listeners

```bash
netstat -tuln | grep -E "(8765|8766)"
```

Both ports 8765 and 8766 should be in LISTEN state.

### 3. Run Health Check

```bash
python test_health.py
```

Expected output:
```
Checking WebSocket backend health...
------------------------------------------------------------
✅ Connected to proxy
📤 Sent health check request

📊 Backend Status:
  Type: health
  Alpaca Connected: True
  Active Symbols: [list of symbols]
  Total Clients: [number]

✅ Alpaca stream is connected and healthy
```

## Monitoring Logs

The services write logs to /tmp for easy monitoring:

### Backend Logs
```bash
tail -f /tmp/trade_stream.log
```

Look for:
- `✅ Trade Stream Server running` - Server started successfully
- `✅ Alpaca stream connected successfully` - Connected to Alpaca
- `🔵 [SUBSCRIBE]` messages - Subscription activity
- `🔔 Received trade from Alpaca` - Trade data flowing

### Proxy Logs
```bash
tail -f /tmp/browser_proxy.log
```

Look for:
- `✅ Proxy server running` - Proxy started successfully
- `Browser client connected` - Clients connecting
- `Connected to backend for client` - Forwarding established

## Troubleshooting

### Services Won't Start

**Problem:** Restart script completes but health check still fails

**Solutions:**
1. Check if ports are already in use:
   ```bash
   lsof -i :8765
   lsof -i :8766
   ```

2. Kill any zombie processes:
   ```bash
   pkill -9 -f "trade_stream_server.py"
   pkill -9 -f "browser_proxy.py"
   ```

3. Check for Python environment issues:
   ```bash
   which python3
   python3 --version
   ```

### Alpaca Stream Not Connecting

**Problem:** Health check shows `Alpaca Connected: False`

**Solutions:**
1. Verify API credentials in `.env`:
   ```bash
   grep ALPACA .env
   ```

2. Check if Alpaca API is accessible:
   ```bash
   curl https://data.alpaca.markets/v2/stocks/AAPL/quotes/latest?feed=sip \
     -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY"
   ```

3. Review backend logs for authentication errors:
   ```bash
   grep -i "error\|failed" /tmp/trade_stream.log
   ```

### Connection Refused on Port 8766

**Problem:** `[Errno 111] Connect call failed ('127.0.0.1', 8766)`

**Solutions:**
1. Wait a few seconds after restart - services need initialization time
2. Check if proxy process is actually running
3. Verify firewall isn't blocking local connections

### High Memory or CPU Usage

**Problem:** Services consuming excessive resources

**Solutions:**
1. Check number of active connections:
   ```bash
   python test_health.py  # Look at "Total Clients"
   ```

2. Review logs for connection leaks or error loops
3. Restart services to clear accumulated state

## Automatic Service Management

For production deployments, consider using systemd services for automatic restart on failure.

Example systemd service files are available in the `services/` directory:
- `market_sentinel_trade_stream.service`
- See `services/README.md` for setup instructions

## Related Commands

- **Start services:** `./services/start_websocket_services.sh`
- **View active connections:** `netstat -an | grep -E "(8765|8766)"`
- **Monitor both logs:** `tail -f /tmp/trade_stream.log /tmp/browser_proxy.log`
- **Check Alpaca connection:** `python test_health.py`

## Architecture Notes

The two-tier architecture (backend + proxy) exists because:

1. **Browser Compatibility:** Modern browsers have strict WebSocket requirements that the backend's Python websockets library handles differently
2. **Connection Stability:** The proxy can maintain browser connections even if backend subscriptions change
3. **Isolation:** The proxy shields the backend from browser-specific connection handling

The proxy (port 8766) is the public-facing endpoint. It forwards all messages bidirectionally to the backend (port 8765), which manages the Alpaca stream connection and subscription logic.
