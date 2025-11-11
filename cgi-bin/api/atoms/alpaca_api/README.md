# Alpaca API Utilities

## Fundamental Data Module

### Overview

The `fundamental_data.py` module provides fundamental stock data using **Yahoo Finance** via the `yfinance` library.

**Important Note**: Alpaca-py does NOT provide fundamental data. The `alpaca.trading.models.Asset` class only contains trading-related information (tradable, marginable, shortable, etc.), not fundamental financial data.

### Available Data

This module fetches the following fundamental data:
- **Float Shares**: Number of shares available for public trading
- **Shares Outstanding**: Total number of shares issued by the company
- **Market Cap**: Total market capitalization

### Installation

```bash
pip3 install yfinance
```

### Usage

```python
from cgi_bin.api.atoms.alpaca_api.fundamental_data import FundamentalDataFetcher

# Initialize fetcher
fetcher = FundamentalDataFetcher(verbose=True)

# Get fundamental data
data = fetcher.get_fundamental_data("AAPL")

# Access the data
float_shares = data['float_shares']           # 14,750,198,855
shares_outstanding = data['shares_outstanding']  # 14,776,353,000
market_cap = data['market_cap']               # $3,981,192,724,480
source = data['source']                       # 'yahoo'
```

### Convenience Function

```python
from cgi_bin.api.atoms.alpaca_api.fundamental_data import get_fundamental_data

data = get_fundamental_data("TSLA", verbose=True)
```

### Testing

```bash
# Test the module with AAPL, TSLA, NVDA
python3 cgi-bin/api/atoms/alpaca_api/fundamental_data.py
```

### Return Format

```python
{
    'float_shares': int or None,           # Float shares from Yahoo Finance
    'shares_outstanding': int or None,     # Total shares outstanding
    'market_cap': int or None,             # Market capitalization in USD
    'source': str                          # 'yahoo', 'yahoo-no-data', 'yahoo-error', etc.
}
```

### Integration with Momentum Alerts

To use this module in the momentum alerts system:

```python
# In momentum_alerts.py
from cgi_bin.api.atoms.alpaca_api.fundamental_data import FundamentalDataFetcher

# Initialize in __init__
self.fundamental_fetcher = FundamentalDataFetcher(verbose=verbose)

# Fetch data when loading symbols
data = self.fundamental_fetcher.get_fundamental_data(symbol)

# Store float shares in symbol metadata
symbols_dict[symbol] = {
    'source': 'gainers_csv',
    'market_open_price': price,
    'float_shares': data['float_shares'],
    'shares_outstanding': data['shares_outstanding'],
    'market_cap': data['market_cap']
}

# Calculate Float Rotation
if float_shares and float_shares > 0:
    float_rotation = total_volume / float_shares
```

### Error Handling

The module handles errors gracefully:
- Returns `None` values if data not available
- Returns `source: 'yahoo-no-data'` if symbol not found
- Returns `source: 'yahoo-error'` if request fails
- Returns `source: 'error-yfinance-not-installed'` if yfinance not installed

### Data Source: Yahoo Finance

Data is retrieved from Yahoo Finance using the `yfinance` library, which provides:
- Real-time fundamental data
- No API key required
- Free to use
- Covers most US and international stocks

### Why Not Alpaca-py?

Based on research conducted in January 2025:

1. **Alpaca API Documentation**: The `/v2/assets` endpoint does not include fundamental data fields
2. **alpaca-py SDK**: The `Asset` model source code confirms no fundamental data fields exist
3. **Community Forums**: Multiple users have reported that Alpaca does not provide float or shares outstanding data
4. **Historical Context**: Alpaca previously partnered with Polygon.io but discontinued that integration in 2021

### Alternative Data Sources

If you need alternative data sources:
- **Polygon.io API**: Provides `share_class_shares_outstanding`, `weighted_shares_outstanding`, `market_cap` (requires API key)
- **Alpha Vantage**: Free API with fundamental data
- **Financial Modeling Prep**: Comprehensive fundamental data API
- **atoms.api.fundamental_data**: Existing module with Polygon (primary) and Yahoo (fallback)

### Performance Notes

- Yahoo Finance requests are relatively fast (< 1 second per symbol)
- Consider caching results to avoid repeated requests
- Batch symbol requests if fetching for many symbols
- Data is updated daily by Yahoo Finance
