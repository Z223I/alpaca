# Symbol Polling System - WebSocket Real-Time Streaming

## Overview

**UPDATED**: This module now uses **WebSocket streaming** instead of REST API polling for real-time market data.

## High Level Requirements

Real-time price monitoring system using Alpaca WebSocket API for streaming trade data (time and sales).

## Implementation Status

**✅ COMPLETED** - Updated to use WebSocket subscriptions with the following improvements:
- Real-time trade data via WebSocket (not polling)
- Sub-second latency for price updates
- Single WebSocket connection for all symbols (no rate limits)
- Automatic subscription updates when symbol list changes
- Async/await architecture for efficient concurrent operations

## Architecture

### Components

1. **SymbolManager**: Loads symbols from CSV files
   - `data/{YYYYMMDD}.csv` - Daily symbol list
   - `historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv` - Gainers file

2. **PricePoller**: WebSocket streaming manager
   - Uses `AlpacaStreamClient` from `atoms/websocket/alpaca_stream.py`
   - Subscribes to minute bars for all symbols
   - Handles dynamic subscription updates
   - Processes real-time market data

3. **File Monitoring**: Watchdog-based file system monitoring
   - Detects gainers file updates
   - Triggers automatic symbol list refresh

### WebSocket vs REST Comparison

| Feature | Old (REST Polling) | New (WebSocket Streaming) |
|---------|-------------------|---------------------------|
| **Latency** | 5-second intervals | Sub-second real-time |
| **API Calls** | N symbols × 12/min | 1 connection total |
| **Rate Limits** | Risk hitting limits | No issue |
| **Data Type** | Latest trade only | Full time & sales |
| **Scalability** | Poor (100+ symbols) | Excellent (1000+ symbols) |

## Usage

### Basic Usage
```bash
# Live WebSocket streaming (paper trading, today's data from files)
python atoms/api/symbol_polling.py

# Monitor specific symbol (single)
python atoms/api/symbol_polling.py --symbol AAPL --verbose

# Monitor multiple symbols (comma-separated, auto-uppercase)
python atoms/api/symbol_polling.py --symbol aapl,googl,msft,tsla --verbose

# Test mode with specific symbols (no API calls)
python atoms/api/symbol_polling.py --symbol AAPL --test

# Live streaming with verbose output (loads symbols from files)
python atoms/api/symbol_polling.py --verbose

# Use historical data from specific date (YYYYMMDD format)
python atoms/api/symbol_polling.py --date 20251017 --verbose

# Combine --symbol with --date for historical testing
python atoms/api/symbol_polling.py --symbol AAPL,TSLA --date 20251017 --test

# Test mode (simulated data, no API calls, loads from files)
python atoms/api/symbol_polling.py --test

# Different account configuration
python atoms/api/symbol_polling.py --account-name Dale --account live
```

### Command Line Arguments

- `--account-name` - Account name for API credentials (default: Bruce)
- `--account` - Account type: paper, live, cash (default: paper)
- `--verbose` - Enable verbose output
- `--test` - Test mode with simulated WebSocket data
- `--date YYYYMMDD` - Use historical data from specific date (e.g., 20251017). If not specified, uses today's date.
- `--symbol SYMBOL` - Monitor specific symbol(s). Comma-separated for multiple (e.g., `AAPL,GOOGL,MSFT`). Symbols are automatically converted to uppercase. **When specified, symbols are NOT loaded from CSV files.**

## Technical Details

### Real-Time Data Format

The system now receives **real-time trade data** (time and sales):
```
TRADE: AAPL | Price: $178.25 | Volume: 1,500 | Time: 2025-10-17 09:45:32 ET
```

Each update contains:
- **Symbol**: Stock ticker
- **Price**: Trade execution price (formatted to 2 decimal places)
- **Volume**: Number of shares traded (individual shares with comma formatting)
  - **Units**: 1 share = 1 (NOT thousands)
  - Example: `1,500` means 1,500 individual shares, not 1,500,000
- **Time**: Full execution timestamp with date (YYYY-MM-DD HH:MM:SS ET)

**IMPORTANT - Timestamps:**
- **All timestamps are in Eastern Time (ET)**, not local time
- WebSocket data is automatically converted from UTC to ET by `AlpacaStreamClient`
- Format: `YYYY-MM-DD HH:MM:SS ET` (includes date, time, and timezone label)
- Test mode also uses ET (via `pytz.timezone('US/Eastern')`)
- This ensures consistency with market hours (NYSE/NASDAQ operate in ET)

### Dynamic Subscription Management

The system automatically:
1. Monitors symbol list changes every 10 seconds
2. Updates WebSocket subscriptions when symbols are added/removed
3. Maintains continuous streaming connection
4. Handles reconnection on connection failures

### Data Sources

1. **Daily Symbols**: `data/{YYYYMMDD}.csv`
   - Contains Symbol column
   - Loaded at startup
   - Refreshed periodically
   - Use `--date YYYYMMDD` to specify historical date

2. **Gainers List**: `historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv`
   - Contains symbol column (lowercase)
   - Updated multiple times per day
   - Monitored via file system watchers
   - Automatically uses same date as `--date` parameter

### Symbol Override Mode (`--symbol`)

Use the `--symbol` parameter to monitor specific symbols without loading from CSV files:

```bash
# Monitor single symbol
python atoms/api/symbol_polling.py --symbol AAPL --verbose

# Monitor multiple symbols (NO SPACES after commas - recommended)
python atoms/api/symbol_polling.py --symbol aapl,googl,msft,tsla --verbose

# Monitor multiple symbols (with spaces - must use quotes)
python atoms/api/symbol_polling.py --symbol "aapl, googl, msft, tsla" --verbose
```

**Important Syntax Notes:**
- **NO SPACES** after commas: `aapl,googl,msft` ✅ (recommended)
- **With spaces requires quotes**: `"aapl, googl, msft"` ✅
- **Without quotes and spaces**: `aapl, googl` ❌ (shell error)

**How it works:**
- Symbols are automatically converted to **UPPERCASE** (e.g., `aapl` → `AAPL`)
- Comma-separated lists supported: `AAPL,GOOGL,MSFT`
- Whitespace is automatically trimmed via `.strip()`
- **Bypasses CSV file reading entirely** when `--symbol` is specified
- Useful for:
  - Testing specific symbols
  - Focused monitoring without managing CSV files
  - Quick ad-hoc price tracking
  - Development and debugging

**Important:** When `--symbol` is used, the system does NOT read symbols from:
- `data/{YYYYMMDD}.csv` files
- `historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv` files

### Historical Data Mode (`--date`)

Use the `--date` parameter to analyze historical trading days:

```bash
# Stream historical data from October 17, 2025
python atoms/api/symbol_polling.py --date 20251017 --verbose
```

**How it works:**
- Loads symbols from `data/20251017.csv`
- Loads gainers from `historical_data/2025-10-17/market/gainers_nasdaq_amex.csv`
- Subscribes to WebSocket for these symbols
- Useful for testing strategies with known symbol lists
- Can be combined with `--test` mode for offline simulation

**Date format:** YYYYMMDD (e.g., 20251017 for October 17, 2025)

**Note:** Without `--date`, the system uses today's date and waits for today's data files to appear.

### Combining `--symbol` and `--date`

You can combine both parameters for flexible testing:

```bash
# Test specific symbols with historical date
python atoms/api/symbol_polling.py --symbol AAPL,TSLA --date 20251017 --test

# Live WebSocket for specific symbols (ignores --date files)
python atoms/api/symbol_polling.py --symbol AAPL --verbose
```

**Note:** When `--symbol` is specified, `--date` has no effect on symbol loading (since files are not read). However, `--date` may still be useful for other date-dependent features.

### Async Architecture

The system uses Python `asyncio` for concurrent operations:
- **Main loop**: WebSocket message processing
- **Update loop**: Symbol list monitoring
- **File watcher**: Background file system monitoring

All operations run concurrently without blocking.

## Configuration

### Constants

- `SYMBOL_UPDATE_INTERVAL = 10` # seconds between symbol list checks

### Dependencies

- `atoms/websocket/alpaca_stream.py` - WebSocket client
- `atoms/api/init_alpaca_client.py` - API credential management
- `watchdog` - File system monitoring
- `asyncio` - Async/await support

## Testing

### Test Mode
```bash
python atoms/api/symbol_polling.py --test
```

Test mode:
- Uses most recent data files (not today's)
- Simulates WebSocket stream with random data
- No API calls made
- Runs for 5 seconds in functionality test

### Output Example
```
TEST MODE: Simulating WebSocket stream for 49 symbols
First 5 symbols: ['WWR', 'RPGL', 'HYPD', 'MAGH', 'DGXX']
TRADE: WWR | Price: $365.84 | Volume: 9533 | Time: 13:41:23
TRADE: RPGL | Price: $467.63 | Volume: 6490 | Time: 13:41:23
...
```

## Performance Benefits

1. **Reduced API Load**: Single WebSocket vs hundreds of REST calls/minute
2. **Lower Latency**: Real-time updates vs 5-second polling
3. **Better Scalability**: Handles 1000+ symbols efficiently
4. **True Time & Sales**: Receives actual trade executions, not just latest quote

## Migration Notes

**Breaking Changes from Previous Version:**
- No longer uses REST API `get_latest_trade()` calls
- Requires async/await in calling code
- Different output format (TRADE vs ALERT)
- Runs continuously (not in background thread)

## Standards Compliance

- ✅ Linting: flake8 compliant
- ✅ Type hints: Optional type annotations
- ✅ Testing: Test mode available
- ✅ Documentation: Comprehensive docstrings
- ✅ Error handling: Try/except with logging

## Future Enhancements

Potential improvements:
- Add quote subscriptions (bid/ask data)
- Support for trade-level data (message type "t" instead of "b")
- Database storage for historical time & sales
- Alert triggers based on volume/price patterns
