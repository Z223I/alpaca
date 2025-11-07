# Watch List Changes - index.html

## Overview
This document details the Watch List-specific changes made to `public_html/index.html` in the current feature branch (`feature/market_sentinel_chart`).

## Watch List Enhancements

### 1. Oracle File Change Detection (Lines 2722-2740)

**Feature**: Automatic detection of Oracle CSV file changes with smart deleted symbols management.

**Implementation**:
- Added `currentOracleFile` property to `watchListState` to track the active Oracle data file
- When Oracle file changes (e.g., from `20251105.csv` to `20251107.csv`):
  - System automatically detects the file change
  - Clears all previously deleted symbols (`deletedSymbols.clear()`)
  - Allows fresh Oracle symbols from new file to appear
  - Logs the file change and cleared symbol count for debugging

**Purpose**:
- Prevents old deleted symbols from blocking new Oracle symbols when a new day's data is loaded
- Ensures users see fresh momentum data from the latest Oracle file
- Gives users ability to re-evaluate and delete unwanted symbols from the new file

**Code Location**: `public_html/index.html:2722-2740`

```javascript
// Check if Oracle file has changed
const newOracleFile = data.oracle_file || '';
if (watchListState.currentOracleFile && newOracleFile &&
    watchListState.currentOracleFile !== newOracleFile) {
    console.log(`🔄 Oracle file changed: ${watchListState.currentOracleFile} -> ${newOracleFile}`);
    console.log(`   Clearing deleted symbols to allow new Oracle symbols to appear`);

    // Clear ALL deleted symbols when Oracle file changes
    const previousDeletedCount = watchListState.deletedSymbols.size;
    watchListState.deletedSymbols.clear();
    console.log(`   Cleared ${previousDeletedCount} deleted symbols for fresh Oracle data`);
}

// Update current Oracle file
watchListState.currentOracleFile = newOracleFile;
```

### 2. State Management Enhancement (Line 2697)

**Feature**: Added tracking for current Oracle file.

**Implementation**:
- Added `currentOracleFile: null` property to `watchListState` object
- Tracks which Oracle CSV file (e.g., `data/20251107.csv`) is currently loaded
- Used to detect when Oracle data source changes between API calls

**Code Location**: `public_html/index.html:2697`

## Related Watch List Features (No Changes)

The following Watch List features remain unchanged but are important context:

### Manual Symbol Management
- `manualSymbols` Set: Tracks user-added symbols
- `deletedSymbols` Set: Tracks symbols user has removed (now cleared on Oracle file change)
- Auto-refresh interval: 2 minutes (120000ms)

### Symbol Filtering Logic
- Filters out deleted symbols unless manually re-added
- Preserves manual symbols even if not in API response
- Alphabetical sorting of all symbols

### Symbol Source Badges
- Oracle (🔮): From daily Oracle CSV file
- Gainers (📈): From top gainers list
- Surge (🚀): From volume surge detection

## Technical Notes

### Integration with Watch List API
- Expects `oracle_file` field in API response from `watch_list_api.py`
- API should return the current Oracle file path in response data
- Format example: `"oracle_file": "data/20251107.csv"`

### User Experience Benefits
1. **Fresh Daily Data**: Users automatically see new Oracle symbols when market data updates
2. **No Stale Blocks**: Previously deleted symbols don't prevent new symbols from appearing
3. **Seamless Transition**: File changes are transparent with informative console logging
4. **Flexibility**: Users can still delete unwanted symbols from the new day's data

### Debugging
Console logs provide clear visibility:
- `🔄 Oracle file changed: [old] -> [new]`
- `Clearing deleted symbols to allow new Oracle symbols to appear`
- `Cleared [N] deleted symbols for fresh Oracle data`

## Files Modified
- `public_html/index.html` - Watch List functionality (lines 2692-2777)

## Testing Recommendations
1. Test Oracle file transition (e.g., `20251105.csv` → `20251107.csv`)
2. Verify deleted symbols are cleared on file change
3. Confirm new Oracle symbols appear after file change
4. Test manual symbol persistence across file changes
5. Verify console logging for debugging
