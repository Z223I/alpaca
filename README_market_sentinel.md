# Market Sentinel - Project Documentation

## Overview

Market Sentinel is a real-time stock market monitoring web application designed for deployment on GoDaddy hosting. It provides professional-grade charting capabilities with candlestick charts, technical indicators, and time & sales data.

**Status**: Initial implementation complete (Phase 1)

## Project Structure

```
.
├── public_html/
│   └── index.html           # Main web interface
├── cgi-bin/
│   ├── atoms/
│   │   └── alpaca/          # Alpaca API atoms (future)
│   └── molecules/
│       └── alpaca/
│           ├── alpaca_config.py  # API configuration (copied from code/)
│           └── alpaca.py         # Market data collection module
└── logs/                    # Log files directory (existing)
```

## Architecture

### Frontend (`public_html/index.html`)

**Technology Stack:**
- Vanilla JavaScript (no framework dependencies)
- TradingView Lightweight Charts 4.1.0 for professional candlestick charts
  - Replaced Chart.js due to compatibility issues with chartjs-chart-financial plugin
  - Lightweight Charts is specifically designed for financial data visualization
  - Better performance and native candlestick support

**Features Implemented:**
- 🔍 **Search Bar**: Enter stock symbols with magnifying glass icon
- 📑 **Tabbed Interface**: Multiple charts in separate tabs with close buttons
- 📊 **Candlestick Charts**: Professional-grade financial charts
- ⚙️ **Chart Controls**:
  - Intervals: 10s, 20s, 30s, 1m, 5m, 30m, 1h, 1d, 1w, 1mo
  - Ranges: 1D, 2D, 5D, 1M, 1Y
  - Indicators: EMA(9), EMA(20), EMA(21), EMA(50), EMA(200), VWAP, MACD, Volume
- 📈 **Time & Sales Panel**: Real-time trade data display
- 🎨 **Dark Theme**: Professional trading terminal aesthetic
- ♻️ **Auto-refresh**: Charts update every 30 seconds, trades every 10 seconds
- 📏 **Resizable Panels**: Drag-to-resize functionality (planned)

### Backend (`cgi-bin/molecules/alpaca/`)

**Technology Stack:**
- Python 3.x
- Flask (for CGI/web API)
- Alpaca Trade API
- pandas for data manipulation

**Components:**

#### `alpaca_config.py`
- Exact copy of `code/alpaca_config.py`
- Configuration dataclasses for API credentials
- Supports multiple accounts (Bruce, Dale Wilson, Janice)
- Environment types: paper, live, cash

#### `alpaca.py` (AlpacaMarketData class)
Market data collection module with NO trading functionality.

**Key Methods:**
```python
class AlpacaMarketData:
    def __init__(provider, account_name, account)
    def get_latest_quote_data(symbol) -> Dict
    def get_bar_data(symbol, timeframe, start_date, end_date) -> DataFrame
    def get_chart_data(symbol, interval, range_str) -> Dict
    def get_time_and_sales(symbol, start_date, limit) -> List[Dict]
    def to_json(data) -> str
```

**Data Access:**
- Uses SIP (Securities Information Processor) data via Alpaca API
- Supports historical and real-time data retrieval
- Returns data in JSON format for web consumption

**CRITICAL Implementation Note:**
The code correctly uses Alpaca Bar object single-letter attributes:
- `bar.o` (open), `bar.h` (high), `bar.l` (low), `bar.c` (close), `bar.v` (volume), `bar.t` (timestamp)
- NOT `bar.open`, `bar.high`, etc. (these don't exist and cause AttributeError)

## Phase 1: Completed Items ✅

1. ✅ Created GoDaddy-compliant directory structure
2. ✅ Copied `alpaca_config.py` to `cgi-bin/molecules/alpaca/`
3. ✅ Created data-collection-focused `alpaca.py` (no trading)
4. ✅ Built comprehensive HTML interface with:
   - Search functionality
   - Tabbed chart panels
   - Candlestick chart display
   - Time & sales window
   - Chart controls (intervals, ranges, indicators)
   - Professional dark theme UI
5. ✅ Created project documentation

## Testing Findings

During testing, discovered that the codebase uses `alpaca-trade-api` (legacy package) rather than `alpaca-py` (newer package). The `market_data.py` module was initially written for the newer API and needs to be adapted to use `alpaca-trade-api` classes.

**Directory Naming Issue Resolved:**
- Renamed `cgi-bin/atoms/alpaca/` → `cgi-bin/atoms/alpaca_api/`
- Renamed `cgi-bin/molecules/alpaca/` → `cgi-bin/molecules/alpaca_molecules/`
- This avoids Python package name collisions with the `alpaca` import from alpaca-trade-api

**API Compatibility Updates Needed:**
- Replace `alpaca.data.timeframe.TimeFrame` with `alpaca_trade_api.TimeFrame`
- Replace `alpaca.data.requests.StockBarsRequest` with native `api.get_bars()` calls
- Replace `alpaca.data.requests.StockTradesRequest` with `api.get_trades()` calls
- Replace `StockHistoricalDataClient` with `alpaca_trade_api.REST` client

## Phase 2: Next Steps 🚧

### A. Backend API Integration
Create Flask CGI endpoint to connect frontend to backend:

**File:** `cgi-bin/api/market_data.py`
```python
#!/usr/bin/env python3
# Flask CGI endpoint for market data

# Endpoints needed:
# GET /api/quote/{symbol}
# GET /api/chart/{symbol}?interval=1m&range=1d
# GET /api/trades/{symbol}?limit=100
```

**Integration Tasks:**
1. Create Flask app in `cgi-bin/api/`
2. Configure CGI handling for GoDaddy
3. Implement REST endpoints
4. Update `index.html` to call actual API instead of mock data
5. Add error handling and status codes

### B. Real-time Updates
Implement WebSocket or Server-Sent Events for live data streaming:
- Replace polling with push-based updates
- Reduce API call frequency
- Improve data freshness

### C. Technical Indicators
Implement calculation functions for:
- EMAs (9, 20, 21, 50, 200)
- VWAP (Volume Weighted Average Price)
- MACD (Moving Average Convergence Divergence)

**Atom Locations:**
- Copy/adapt from `atoms/utils/calculate_macd.py`
- Use existing `atoms/display/generate_chart_from_df.py` patterns

### D. Panel Resizing
Implement drag-to-resize for time & sales panel:
- Add mouse event handlers
- Store panel width in localStorage
- Maintain responsive behavior

### E. Chart Enhancements
1. Add volume bars below candlesticks
2. Add MACD histogram panel
3. Implement zoom and pan controls
4. Add drawing tools (trend lines, support/resistance)

### F. Data Caching
Implement caching to reduce API calls:
- Cache bar data for 30-60 seconds
- Use Redis or file-based cache
- Implement cache invalidation

### G. User Preferences
Add localStorage for:
- Default interval/range settings
- Preferred indicators
- Theme customization
- Saved watchlists

## Technical Considerations

### GoDaddy Hosting Requirements

**CGI Configuration:**
```apache
# .htaccess for Python CGI
Options +ExecCGI
AddHandler cgi-script .py
```

**Python Environment:**
- Ensure Python 3.x is available
- Install dependencies via pip or requirements.txt
- Use virtual environment if supported

**Dependencies to Install:**
```bash
pip3 install alpaca-trade-api flask pandas pytz
```

### API Rate Limits

**Alpaca API Limits:**
- 200 requests per minute per API key
- Consider caching and request batching
- Implement exponential backoff for rate limit errors

### Security Considerations

1. **API Keys**: Never expose in frontend code
2. **Input Validation**: Sanitize all symbol inputs
3. **CORS**: Configure appropriately for domain
4. **Authentication**: Consider adding user auth for production

## Testing Plan

### Unit Tests
- Test `AlpacaMarketData` methods with mock data
- Test date range calculations
- Test timeframe conversions

### Integration Tests
- Test Flask API endpoints
- Test data flow from Alpaca → Backend → Frontend
- Test error handling and edge cases

### Browser Testing
- Chrome/Edge (Chromium)
- Firefox
- Safari
- Mobile browsers (responsive design)

## Known Issues / Limitations

1. **Mock Data**: Frontend currently uses mock data (see TODO comments)
2. **No Backend API**: Flask endpoint not yet implemented
3. **No Indicator Calculations**: EMAs, VWAP, MACD not calculated
4. **No WebSocket**: Using polling instead of push
5. **No Persistence**: Chart settings not saved
6. **No Error UI**: Need better error message display
7. **No Loading States**: Need skeleton screens for better UX

## Development Notes

### Code Style
- Follow existing repo patterns in `atoms/` and `code/`
- Use type hints for Python functions
- Document all public methods
- Keep functions focused and testable

### Linting
```bash
# Run flaking on new code
~/miniconda3/envs/alpaca/bin/python -m flake8 cgi-bin/
```

### Testing
```bash
# Test alpaca.py module directly
~/miniconda3/envs/alpaca/bin/python cgi-bin/molecules/alpaca/alpaca.py
```

## References

### Existing Codebase
- **Chart Generation**: `atoms/display/generate_chart_from_df.py`
- **Plot Functions**: `atoms/display/plot_candle_chart.py`
- **MACD Calculation**: `atoms/utils/calculate_macd.py`
- **Main Trading**: `code/alpaca.py` (trading features NOT included in web version)

### External Resources
- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [TradingView Lightweight Charts Documentation](https://tradingview.github.io/lightweight-charts/)
- [Flask CGI Deployment](https://flask.palletsprojects.com/en/2.3.x/deploying/cgi/)

## Changelog

### 2025-10-30 - Phase 1 Initial Implementation + Candlestick Fix
- Created GoDaddy directory structure
- Implemented `AlpacaMarketData` class for data collection
- Built comprehensive HTML/CSS/JS frontend
- Added tabbed interface with chart controls
- Implemented time & sales panel
- Created project documentation
- **FIX**: Replaced Chart.js + chartjs-chart-financial with TradingView Lightweight Charts
  - Chart.js 4.4.0 incompatible with chartjs-chart-financial@0.2.1 (designed for Chart.js 3.x)
  - Lightweight Charts provides superior candlestick rendering
  - Native financial chart support with better performance
  - Updated data format: Unix timestamps in seconds, proper OHLC structure

---

**Last Updated**: 2025-10-30
**Branch**: `feature/market_sentinel_chart`
**Status**: Phase 1 Complete, Ready for Phase 2 Backend Integration
