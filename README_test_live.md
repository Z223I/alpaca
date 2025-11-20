# Test Live WebSocket - Setup and Troubleshooting Guide

## Overview

This document describes the WebSocket real-time trade streaming system for Market Sentinel and the fixes applied to resolve reconnection issues in Test 4 (`public_html/test_live.html`).

**Last Updated**: 2025-11-20
**Status**: ✅ **PRODUCTION READY** - SIP feed configured for all exchanges, full exchange names displayed

---

## Quick Start After Computer Restart

### 1. Start WebSocket Services

```bash
cd /home/wilsonb/dl/github.com/Z223I/alpaca
./services/restart_websocket_services.sh
```

This will start:
- **Trade Stream Server** (port 8765) - Connects to Alpaca WebSocket
- **Browser Proxy** (port 8766) - Proxies browser connections to backend

### 2. Verify Services Are Running

```bash
# Check if services are running
ps aux | grep -E "(trade_stream|browser_proxy)" | grep -v grep

# Run automated test
python3 test_websocket_reconnect.py

# Expected: All tests pass ✅
```

### 3. Test in Browser

Open in your web browser:
```
file:///home/wilsonb/dl/github.com/Z223I/alpaca/public_html/test_live.html
```

Go to **Test 4: WebSocket Real-Time Streaming**:
1. Click **"Connect to WebSocket"** (establishes persistent connection)
2. Click **"Add Symbol(s)"** (subscribes to AAPL by default, supports multiple symbols)
3. Wait for live trades to stream with full exchange names from **ALL exchanges** (NYSE, NASDAQ, FINRA ADF, Cboe EDGX, IEX, etc.)
4. Try adding more symbols (e.g., TSLA, MSFT) to track multiple stocks simultaneously
5. Click **"Remove Symbol(s)"** to stop specific symbols or **"Clear All"** to unsubscribe from all
6. Click **"Disconnect"** when done

**Note:** Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R) to ensure you're seeing the latest version.

---

## Architecture

```
Browser (test_live.html)
    ↓ WebSocket ws://localhost:8766
Browser Proxy (services/browser_proxy.py)
    ↓ WebSocket ws://localhost:8765
Trade Stream Server (services/trade_stream_server.py)
    ↓ WebSocket wss://stream.data.alpaca.markets/v2/sip
Alpaca Market Data Stream (SIP Feed - ALL Exchanges)
```

**Data Feed:** Using **SIP (Securities Information Processor)** feed which includes real-time data from all U.S. stock exchanges (requires paid Alpaca subscription $99/month).

---

## Problem That Was Fixed

### Issue
After restarting the computer, Test 4 would:
- ✅ Connect and receive trade data ONCE
- ❌ **FAIL** on disconnect and reconnect (timeout after a few seconds)

### Root Cause
**Timing issue** in `services/trade_stream_server.py`:

1. Backend set `alpaca_connected = True` after only **2 seconds**
2. Alpaca WebSocket needed **~5 seconds** to fully authenticate
3. When clients disconnected during the authentication window, the backend sent **invalid unsubscribe commands**
4. This caused `ERROR: invalid syntax (400)` which **corrupted the Alpaca connection**

### Symptoms
- First connection worked
- Reconnection failed with timeout
- Backend logs showed: `alpaca.data.live.websocket - ERROR - error: invalid syntax (400)`
- Subsequent connections failed because Alpaca stream was broken

---

## Fixes Applied

### File Modified: `services/trade_stream_server.py`

#### Fix 1: Increased Connection Wait Time (Line 104)
```python
# BEFORE: 2 seconds
await asyncio.sleep(2)

# AFTER: 5 seconds
await asyncio.sleep(5)  # Give it time to establish connection and authenticate
```

**Why**: Alpaca needs time to: Connect → Authenticate → Be ready for subscriptions

#### Fix 2: Guard Against Premature Subscription (Line 270-272)
```python
async def _subscribe_alpaca_symbol(self, symbol: str):
    """Subscribe to Alpaca trade stream for a symbol."""
    # Only subscribe if Alpaca is connected
    if not self.alpaca_connected:
        logger.warning(f"Cannot subscribe to {symbol} - Alpaca stream not connected yet")
        return
```

**Why**: Prevents sending subscribe commands before Alpaca is ready

#### Fix 3: Guard Against Premature Unsubscription (Line 294-297)
```python
# CRITICAL FIX: Don't unsubscribe if Alpaca isn't connected yet
# This prevents sending invalid commands during authentication
if not self.alpaca_connected:
    logger.warning(f"Alpaca stream not connected, deferring unsubscribe for {symbol}")
    self.alpaca_subscribed.discard(symbol)  # Remove from tracking but don't send command
    return
```

**Why**: Prevents sending unsubscribe commands during authentication window, which was causing the 400 error

### File Modified: `public_html/test_live.html`

#### Enhancement 1: Symbol Input Uppercase (Line 439)
```html
<input type="text" id="ws-symbol" value="AAPL" placeholder="e.g., AAPL" style="text-transform: uppercase">
```

**Why**: Symbol input now automatically displays in uppercase as you type

#### Enhancement 2: Clear Stats on Disconnect (Line 848-853)
```javascript
// Clear the stats (keep Connection, clear Trades Received and Last Price)
document.getElementById('ws-trades-count').textContent = '0';
document.getElementById('ws-last-price').textContent = '-';

// Reset trades count
wsTradesCount = 0;
```

**Why**: Provides cleaner UI when disconnecting. Symbol field is preserved, other stats are cleared.

---

## Test Results

All tests passing ✅:

### Test 1: Rapid Reconnection
```bash
python3 test_rapid_reconnect.py
# ✅ All 5 rapid reconnections successful!
```

### Test 2: Sustained Connection
```bash
python3 test_sustained_connection.py
# ✅ SUSTAINED CONNECTION TEST PASSED
# First connection: {'connecting', 'subscribed'}
# Reconnection: {'connecting', 'subscribed'}
```

### Test 3: Live Trade Streaming
```bash
python3 test_live_trades.py
# ✅ Successfully received 5 live trades!
# AAPL @ $269.35 x 100 shares
```

### Test 4: Symbol Switching
```bash
python3 test_symbol_switching.py
# ✅ SYMBOL SWITCHING TEST PASSED
# AAPL Test: ✅ PASSED
# SGBX Test: ✅ PASSED
```

### Test 5: Backend Health
```bash
python3 test_health.py
# ✅ Alpaca stream is connected and healthy
```

### Error Check
- **Before Fix**: `ERROR: invalid syntax (400)` in Alpaca WebSocket
- **After Fix**: No errors ✅

---

## Service Management

### Start Services
```bash
./services/restart_websocket_services.sh
```

### Check Status
```bash
ps aux | grep -E "(trade_stream|browser_proxy)" | grep -v grep
```

Expected output:
```
wilsonb    XXXXX  python3 services/trade_stream_server.py
wilsonb    XXXXX  python3 services/browser_proxy.py
```

### View Logs
```bash
# Backend logs
tail -f /tmp/trade_stream.log

# Proxy logs
tail -f /tmp/browser_proxy.log
```

### Stop Services
```bash
pkill -f "trade_stream_server.py"
pkill -f "browser_proxy.py"
```

---

## Testing Tools Created

All test scripts are in the repository root:

1. **`test_health.py`** - Check backend health and Alpaca connection status
2. **`test_rapid_reconnect.py`** - Stress test with 5 rapid connect/disconnect cycles
3. **`test_sustained_connection.py`** - Realistic browser-like connection test
4. **`test_live_trades.py`** - Test live market data streaming
5. **`test_symbol_switching.py`** - Test changing symbols across reconnections (AAPL → SGBX)
6. **`test_websocket_reconnect.py`** - Basic reconnection test

### Quick Health Check
```bash
python3 test_health.py
```

### Full Test Suite
```bash
python3 test_rapid_reconnect.py
python3 test_sustained_connection.py
python3 test_symbol_switching.py
```

---

## Troubleshooting

### Problem: Services not running after restart
**Solution**:
```bash
./services/restart_websocket_services.sh
```

### Problem: Backend not responding (health check timeout)
**Symptoms**: `python3 test_health.py` times out

**Solution**: Restart services
```bash
pkill -f "trade_stream_server.py"
pkill -f "browser_proxy.py"
./services/restart_websocket_services.sh
```

### Problem: "Alpaca Connected: False"
**Symptoms**: Health check shows Alpaca not connected

**Solution**: Wait 5-10 seconds after starting services, Alpaca needs time to authenticate
```bash
sleep 10
python3 test_health.py
```

### Problem: Browser shows "Connection timeout"
**Check**:
1. Are services running? `ps aux | grep trade_stream`
2. Is backend healthy? `python3 test_health.py`
3. Check logs: `tail -50 /tmp/trade_stream.log`

**Solution**: Restart services
```bash
./services/restart_websocket_services.sh
```

### Problem: No trades appearing (market hours)
**Check**:
1. Is market open? Market hours: 9:30 AM - 4:00 PM ET
2. Check backend logs: `grep "trade from Alpaca" /tmp/trade_stream.log`
3. Try a liquid symbol like AAPL or SPY

**Note**: During market hours, you should see trades within 10-15 seconds. After hours, connection will work but no trades will appear (expected behavior).

---

## Files in This Repository

### Main Application Files
- **`public_html/test_live.html`** - Browser test interface with 4 tests
- **`services/trade_stream_server.py`** - Backend WebSocket server (FIXED)
- **`services/browser_proxy.py`** - Browser-compatible WebSocket proxy
- **`services/restart_websocket_services.sh`** - Service startup script

### Test Files
- **`test_health.py`** - Health check utility
- **`test_rapid_reconnect.py`** - Reconnection stress test
- **`test_sustained_connection.py`** - Realistic connection test
- **`test_live_trades.py`** - Live market data test
- **`test_symbol_switching.py`** - Symbol switching test
- **`test_websocket_reconnect.py`** - Basic reconnection test

### Documentation Files
- **`README_test_live.md`** - This file (comprehensive guide)
- **`WEBSOCKET_RECONNECTION_FIX.md`** - Detailed technical explanation
- **`TESTING_SUMMARY.md`** - Test results and browser instructions

---

## How the Fix Works

### Before Fix
```
Timeline:
0s:  Client connects
0s:  Backend starts Alpaca stream
2s:  Backend sets alpaca_connected = True (TOO EARLY!)
2.5s: Client disconnects
2.5s: Backend tries to unsubscribe from Alpaca (INVALID - not authenticated yet!)
3s:  Alpaca returns ERROR 400 "invalid syntax"
     → Alpaca connection corrupted
5s:  Alpaca finishes authentication (but connection is broken)
     → All future connections fail
```

### After Fix
```
Timeline:
0s:  Client connects
0s:  Backend starts Alpaca stream
2s:  Client disconnects
2s:  Backend checks: alpaca_connected = False
     → Skip unsubscribe (SAFE!)
     → Just clean up local state
5s:  Backend sets alpaca_connected = True (SAFE - authenticated)
     → Future operations can now safely talk to Alpaca
10s: Client reconnects
     → Backend correctly subscribes to Alpaca
     ✅ Works!
```

The key insight: **Never send commands to Alpaca during the authentication window (0-5 seconds).**

---

## Market Hours Note

### Market Open
When the market is **open** (9:30 AM - 4:00 PM ET weekdays):
- ✅ Connection works
- ✅ Subscription confirms
- ✅ **Live trades stream** (AAPL typically has trades every few seconds)

### Market Closed
When the market is **closed**:
- ✅ Connection still works
- ✅ Subscription still confirms
- ⚠️  No trades (expected - market is closed)
- ✅ **Reconnection still works** (connection mechanics are independent of market hours)

**Testing Tip**: The best time to test is during market hours when you can see live trades flowing. But reconnection functionality works 24/7.

---

## Expected Behavior (Test 4)

### First Connection
1. Click **"Connect & Subscribe"**
2. Status changes: `PENDING` → `CONNECTING...` → `CONNECTED`
3. After ~10 seconds, see:
   - **Connection**: Connected ✅
   - **Trades Received**: Incrementing (market hours)
   - **Last Price**: Updates with each trade
   - Live trades appear in the stream below

### Disconnect
1. Click **"Disconnect"**
2. Status changes to: `DISCONNECTED`
3. Stats cleared:
   - **Connection**: Disconnected
   - **Trades Received**: 0
   - **Last Price**: -
4. Symbol preserved (e.g., still shows "AAPL")

### Reconnection
1. Click **"Connect & Subscribe"** again (same or different symbol)
2. Should work exactly like first connection ✅
3. Can repeat disconnect/reconnect multiple times

---

## Recent Updates

### 2025-11-20: SIP Feed Configuration for All Exchanges ✅

**Critical Fix:**
- **Configured SIP (Securities Information Processor) feed** to utilize paid Alpaca subscription ($99/month)
- Previously system was using free IEX feed (single exchange only)
- Now receiving trades from **ALL U.S. stock exchanges** including NYSE, NASDAQ, FINRA ADF, Cboe EDGX, Cboe EDGA, IEX, and 17 others

**What Changed:**
- Updated `services/trade_stream_server.py` to use `DataFeed.SIP` instead of default `DataFeed.IEX`
- Added exchange name display instead of single-letter codes
- Implemented comprehensive exchange code lookup table (23 exchanges)
- Updated both stats panel and individual trade items to show full exchange names

**Example Display:**
- Before: `AAPL @ 9:30:15 AM [V] - $269.35 x 100` (IEX only)
- After: `AAPL @ 9:30:15 AM [FINRA ADF] - $269.35 x 100` (all exchanges)

**Exchange Codes Supported:**
- **D** = FINRA ADF (FINRA Alternative Display Facility)
- **K** = Cboe EDGX
- **V** = IEX (Investors Exchange)
- **N** = New York Stock Exchange
- **Q** = NASDAQ OMX
- **A** = NYSE American (AMEX)
- And 17 more exchanges (full list in code)

**Files Modified:**
- `services/trade_stream_server.py` - Added `feed=DataFeed.SIP` parameter and import
- `public_html/test_live.html` - Added `EXCHANGE_NAMES` lookup table and `getExchangeName()` helper function
- `test_live_trades.py` - Added exchange name mapping for Python test script

### 2025-11-19: Core WebSocket Functionality ✅

**Completed:**
1. **Fixed reconnection issue** - Backend keeps Alpaca subscriptions active when clients disconnect
2. **Multi-symbol support** - Track multiple stocks simultaneously (AAPL, TSLA, MSFT, etc.)
3. **Improved UI/UX** - Separate Connect/Subscribe buttons for better control
4. **Smart button states** - Auto enable/disable based on connection status
5. **Backend optimizations**:
   - Persistent Alpaca subscriptions for instant reconnection
   - Prevents double-subscription when reconnecting
   - Symbol reuse detection

**Current Status:**
- ✅ **All automated tests passing**
- ✅ **Multi-symbol subscription working**
- ✅ **Connect/Subscribe/Unsubscribe working**
- ✅ **Exchange names displayed**
- ✅ **Backend stable and efficient**

---

## Contact / Issues

If you encounter issues after restart:
1. Check this README first (Troubleshooting section)
2. Review logs: `/tmp/trade_stream.log` and `/tmp/browser_proxy.log`
3. Run health check: `python3 test_health.py`
4. Try restarting services: `./services/restart_websocket_services.sh`

---

**Last Updated**: 2025-11-20
**Status**: ✅ Production Ready - SIP Feed Configured for All Exchanges
**Tested**: All tests passing with SIP feed providing real-time data from all U.S. stock exchanges
