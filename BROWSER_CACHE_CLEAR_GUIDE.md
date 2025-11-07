# How to Clear Browser Cache for Market Sentinel

## The Problem
Your browser cached the OLD version of index.html (before the live SIP data changes), so it's still showing stale Time & Sales data even though the backend is returning fresh data.

## Verification That Backend is Working
The backend API is correctly returning fresh data:
```
12:00:00 ET: $3.32
11:59:00 ET: $3.30
11:58:00 ET: $3.31
```

But your browser is displaying old cached data from 10:25-10:39 ET at $2.30-$2.47.

## Solution: Completely Clear Browser Cache

### Method 1: Force Reload (Try This First)

**Chrome/Edge/Brave:**
1. Open the page (Market Sentinel)
2. Open DevTools: Press `F12` or `Ctrl+Shift+I` (Windows/Linux) / `Cmd+Option+I` (Mac)
3. With DevTools open, **right-click** the refresh button (🔄)
4. Select **"Empty Cache and Hard Reload"**
5. Keep DevTools open and check the Console tab for "[V2.0]" messages

**Firefox:**
1. Press `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
2. If that doesn't work, press `Ctrl+F5` (Windows/Linux)

**Safari:**
1. Enable Developer menu: Safari → Preferences → Advanced → "Show Develop menu in menu bar"
2. Develop → Empty Caches
3. Then: `Cmd+R` to reload

### Method 2: Clear All Browser Cache (Nuclear Option)

**Chrome/Edge/Brave:**
1. Press `Ctrl+Shift+Delete` (Windows/Linux) or `Cmd+Shift+Delete` (Mac)
2. Select "Cached images and files"
3. Time range: "All time"
4. Click "Clear data"
5. Close and reopen the browser
6. Navigate back to Market Sentinel

**Firefox:**
1. Press `Ctrl+Shift+Delete` (Windows/Linux) or `Cmd+Shift+Delete` (Mac)
2. Select "Cache"
3. Time range: "Everything"
4. Click "Clear Now"
5. Close and reopen Firefox

**Safari:**
1. Safari → Preferences → Privacy
2. Click "Manage Website Data..."
3. Click "Remove All"
4. Confirm
5. Close and reopen Safari

### Method 3: Private/Incognito Window (Quick Test)

Open Market Sentinel in a private/incognito window to test if it works without cache:

- **Chrome/Edge/Brave**: `Ctrl+Shift+N` (Windows) or `Cmd+Shift+N` (Mac)
- **Firefox**: `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
- **Safari**: `Cmd+Shift+N`

If it works in private mode, the issue is definitely browser cache.

## How to Verify It's Fixed

After clearing cache, you should see:

### 1. Console Messages (F12 → Console tab)
Look for:
```
[V2.0] loadTimeSales called for MTC at 12:05:23
Loading trades from: ./cgi-bin/api/market_data_api.py?action=trades&symbol=MTC&limit=100&_t=1730995523000
```

The "[V2.0]" prefix means you're running the new code.

### 2. Time & Sales Panel
- Timestamps should be within 1-2 minutes of current time
- Prices should match current chart prices ($3.15-$3.32 range)
- "Updated:" time should refresh every ~10 seconds
- Exchange column shows "BAR" (synthetic from bars)

### 3. Network Tab (F12 → Network)
- Each API request should have a different `_t=` parameter (timestamp)
- Response should show fresh data with recent timestamps

## Still Not Working?

If after clearing cache multiple ways you still see old data:

### Check 1: Verify You're on the Right Version
Look at the browser tab title or page source (Ctrl+U). You should see:
```html
<!-- Version: 2.0 - Live SIP Data - Cache Bust 20251107 -->
```

### Check 2: Check for JavaScript Errors
1. Open Console (F12)
2. Look for red error messages
3. If you see errors related to `loadTimeSales` or `fetch`, share them

### Check 3: Check Network Requests
1. Open DevTools → Network tab
2. Filter by "market_data_api.py"
3. Click on a request
4. Check the Response tab - does it show fresh timestamps?
5. If Response shows fresh data but page doesn't update, there's a rendering issue

### Check 4: Disable Browser Extensions
Some ad blockers or privacy extensions can interfere:
1. Try disabling all extensions temporarily
2. Reload the page
3. Re-enable extensions one by one to find the culprit

### Check 5: Try a Different Browser
If nothing else works, try accessing Market Sentinel from:
- Chrome (if you were using Firefox)
- Firefox (if you were using Chrome)
- Edge
- Safari

If it works in a different browser, the issue is browser-specific cache/settings.

## What Changed in Version 2.0

1. **Backend**: Switched from paper API to live account for real-time SIP data
2. **API Fallback**: Uses 1-minute bar data when tick data is stale
3. **Cache Busting**: Added timestamps to all API requests
4. **Fetch Options**: Added `cache: 'no-store'` to prevent browser caching

These changes ensure you always get fresh, accurate market data.
