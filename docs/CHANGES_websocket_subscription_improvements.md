# WebSocket Subscription Improvements for Fake Trade Tracking

## Summary

Enhanced the squeeze alerts system to ensure fake trades always receive price updates by:
1. Automatically subscribing symbols when fake trades are created
2. Properly unsubscribing from symbols that are no longer needed (while respecting active fake trades)

## Changes Made

### 1. Store WebSocket Connection as Instance Variable

**File:** `cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py`

- **Line 121:** Added `self.ws = None` initialization in `__init__()`
- **Line 553:** Store websocket connection when connected: `self.ws = ws`
- **Line 587:** Clear websocket connection when disconnecting: `self.ws = None`

**Purpose:** Makes the websocket connection accessible to other methods (particularly `_start_fake_trade()`) without having to pass it through the entire call chain.

### 2. Auto-Subscribe Symbols When Creating Fake Trades

**File:** `cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py`

- **Line 2193:** Made `_start_fake_trade()` async
- **Lines 2214-2223:** Added subscription logic that:
  - Checks if symbol is already in `active_subscriptions`
  - If not subscribed, sends subscribe message to websocket
  - Adds symbol to `active_subscriptions`
  - Logs auto-subscription
  - Handles errors gracefully (continues even if subscription fails)

**Updated call sites:**
- **Line 2415:** Made `_report_squeeze()` async
- **Line 1223:** Added `await` when calling `_report_squeeze()`
- **Line 2725:** Added `await` when calling `_start_fake_trade()`

**Purpose:** Guarantees that fake trades will receive price updates, even if:
- The fake trade is created manually/programmatically
- The symbol wasn't previously subscribed
- Fake trades are restored after a restart (future enhancement)

### 3. Unsubscribe from Removed Symbols (Respecting Active Fake Trades)

**File:** `cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py`

- **Lines 668-689:** Added unsubscribe logic in `_refresh_existing_symbols()` that:
  - For each symbol removed from the backend's active list:
    - Checks if there are active fake trades for that symbol
    - If NO active fake trades: sends unsubscribe message and removes from `active_subscriptions`
    - If YES active fake trades: keeps subscription and logs the reason
  - Handles errors gracefully

**Purpose:**
- Prevents accumulation of unnecessary websocket subscriptions
- Respects active fake trades by maintaining subscriptions they depend on
- Provides clear logging for debugging

## Benefits

### 1. **Defensive Programming**
Ensures fake trades receive price updates regardless of how they're created or when they're created.

### 2. **Future-Proof**
Enables potential future features:
- Manual fake trade creation via API
- Fake trade persistence and restoration after restarts
- Fake trades for symbols not in the main monitoring list

### 3. **Resource Efficiency**
- Only subscribes when needed (checks before subscribing)
- Unsubscribes from symbols no longer needed
- Backend websocket server handles duplicate subscriptions gracefully

### 4. **Clear Dependencies**
Makes the relationship between fake trades and websocket subscriptions explicit in the code.

### 5. **Robust Error Handling**
- Continues operation even if subscription/unsubscription fails
- Logs all operations for debugging
- Won't break existing functionality

## Testing Recommendations

### Manual Testing Scenarios:

1. **Normal Operation (Symbol Already Subscribed):**
   - Start squeeze alerts monitor with a symbol list
   - Wait for a squeeze alert to fire
   - Verify fake trade is created
   - Check logs - should NOT see auto-subscribe message (symbol already subscribed)

2. **Auto-Subscribe on Fake Trade Creation:**
   - Modify code to create a fake trade for a non-subscribed symbol
   - Verify auto-subscribe message appears in logs
   - Verify fake trade receives price updates

3. **Unsubscribe Removed Symbols:**
   - Use `--use-existing` mode
   - Start with symbols subscribed by another client
   - Stop the other client (symbols become "removed")
   - Wait for refresh cycle (60 seconds)
   - Verify unsubscribe messages in logs
   - Create a fake trade for one of the removed symbols
   - Verify that symbol is NOT unsubscribed (kept for fake trade)

4. **Fake Trade Completion and Cleanup:**
   - Create fake trade for a symbol
   - Remove symbol from monitoring list (using --use-existing)
   - Wait for fake trade to complete (hit stop loss or take profit)
   - Wait for next refresh cycle (60 seconds)
   - Verify symbol is unsubscribed after fake trade completes

### Automated Testing:

Consider adding unit tests for:
- `_start_fake_trade()` subscription logic
- Unsubscribe logic in `_refresh_existing_symbols()`
- Edge cases (websocket disconnected, duplicate subscriptions, etc.)

## Log Messages to Watch For

### New Log Messages:

1. **Auto-subscription:**
   ```
   📡 Auto-subscribed to AAPL for fake trade tracking
   ```

2. **Keeping subscription for fake trade:**
   ```
   📌 Keeping AAPL subscribed (has active fake trades)
   ```

3. **Unsubscribing removed symbol:**
   ```
   🔕 Unsubscribed from AAPL
   ```

4. **Unsubscribe errors:**
   ```
   ❌ Failed to unsubscribe from AAPL: [error message]
   ```

## Backward Compatibility

✅ **Fully backward compatible** - all changes are additive:
- Existing functionality unchanged
- No breaking changes to method signatures (only made some async, which is compatible)
- No changes to external APIs or configuration
- Fails gracefully if websocket is unavailable

## Known Limitations

1. **Deferred Unsubscription:** Symbols with completed fake trades are unsubscribed on the next 60-second refresh cycle, not immediately. This is acceptable and keeps the code simpler.

2. **No Fake Trade Persistence:** Fake trades are not currently saved/restored across restarts. If this is implemented in the future, the auto-subscribe logic will handle it automatically.

3. **Websocket Required:** If `self.ws` is None, the subscription logic is skipped. Fake trades will only receive updates if the symbol is already subscribed by other means.

## Files Modified

- `cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py`

## Lines Changed

Approximately 70 lines added/modified across 6 locations in the file.
