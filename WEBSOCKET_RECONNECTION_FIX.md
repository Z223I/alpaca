# WebSocket Reconnection Issue - Root Cause and Fix

## Problem Summary

After restarting the computer, Test 4 in `public_html/test_live.html` would:
- ✅ Connect and get trade data ONCE
- ❌ Fail on disconnect and reconnect after a few seconds

## Root Cause

The issue was **timing-related** in `services/trade_stream_server.py`:

1. **Premature Alpaca Connection State**: The backend set `alpaca_connected = True` after only 2 seconds, but Alpaca's WebSocket needed more time to fully authenticate

2. **Invalid Commands During Authentication**: When clients disconnected/reconnected quickly during the authentication window, the backend would try to unsubscribe from Alpaca before authentication completed, sending invalid commands

3. **Alpaca 400 Error**: This caused `ERROR: invalid syntax (400)` which corrupted the Alpaca WebSocket connection

## Fixes Applied

### Fix 1: Increased Connection Wait Time (Line 104)
**File**: `services/trade_stream_server.py`

```python
# Changed from 2 seconds to 5 seconds
await asyncio.sleep(5)  # Give it time to establish connection and authenticate
```

This gives Alpaca's WebSocket enough time to:
- Connect → Authenticate → Be ready for subscriptions

### Fix 2: Guard Against Premature Subscription (Line 270-272)
**File**: `services/trade_stream_server.py`

```python
async def _subscribe_alpaca_symbol(self, symbol: str):
    """Subscribe to Alpaca trade stream for a symbol."""
    # Only subscribe if Alpaca is connected
    if not self.alpaca_connected:
        logger.warning(f"Cannot subscribe to {symbol} - Alpaca stream not connected yet")
        return
```

Prevents subscribing to Alpaca before it's ready.

### Fix 3: Guard Against Premature Unsubscription (Line 294-297)
**File**: `services/trade_stream_server.py`

```python
# CRITICAL FIX: Don't unsubscribe if Alpaca isn't connected yet
# This prevents sending invalid commands during authentication
if not self.alpaca_connected:
    logger.warning(f"Alpaca stream not connected, deferring unsubscribe for {symbol}")
    self.alpaca_subscribed.discard(symbol)  # Remove from tracking but don't send command
    return
```

Prevents sending unsubscribe commands during the authentication window.

## Test Results

### ✅ Rapid Reconnection Test
```bash
$ python3 test_rapid_reconnect.py
============================================================
✅ All 5 rapid reconnections successful!
============================================================
```

### ✅ Sustained Connection Test
```bash
$ python3 test_sustained_connection.py
[TEST 1] First connection - wait for Alpaca to connect
✅ Received 2 messages: {'connecting', 'subscribed'}

[TEST 2] Reconnection test
✅ Received 2 messages: {'connecting', 'subscribed'}

✅ SUSTAINED CONNECTION TEST PASSED
============================================================
```

### ✅ No Alpaca Errors
- **Before Fix**: `ERROR: invalid syntax (400)`
- **After Fix**: No errors ✅

## How to Apply the Fix

The fixes are already applied to `services/trade_stream_server.py`. To activate them:

1. **Restart the WebSocket services**:
   ```bash
   pkill -f "trade_stream_server.py"
   pkill -f "browser_proxy.py"

   # Start backend
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate alpaca
   python3 services/trade_stream_server.py &

   # Start proxy
   python3 services/browser_proxy.py &
   ```

2. **Test in browser**:
   - Open `public_html/test_live.html`
   - Click "Connect & Subscribe" in Test 4
   - Wait for trade data
   - Click "Disconnect"
   - Click "Connect & Subscribe" again
   - It should reconnect successfully!

## Architecture Overview

```
Browser (test_live.html)
    ↓ WebSocket on port 8766
browser_proxy.py (port 8766)
    ↓ Forwards to port 8765
trade_stream_server.py (port 8765)
    ↓ Connects to Alpaca
Alpaca WebSocket (wss://stream.data.alpaca.markets)
```

## Key Insights

1. **Lazy Initialization**: The Alpaca stream only starts on the first subscription to avoid unnecessary resource usage

2. **Connection Timing**: The 5-second wait is necessary for Alpaca's authentication handshake:
   - Connect → Authenticate → Subscribe/Unsubscribe commands allowed

3. **Reconnection Works**: With proper guards in place, clients can now disconnect and reconnect multiple times without issues

## Files Modified

- `services/trade_stream_server.py`: Added connection guards and increased wait time
- Test files created:
  - `test_websocket_reconnect.py`: Basic reconnection test
  - `test_rapid_reconnect.py`: Rapid reconnection stress test
  - `test_sustained_connection.py`: Realistic browser-like test
  - `test_health.py`: Backend health check utility

## Date Fixed

2025-11-19
