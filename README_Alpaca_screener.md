# Alpaca Stock Screener

A comprehensive stock screener built with the Alpaca Trading API v2, featuring volume surge detection, exchange filtering, and traditional screening metrics. Designed to integrate seamlessly with existing Alpaca trading infrastructure.

## Features

### 🔍 **Core Screening Capabilities**
- **Price Filtering**: Min/max price ranges
- **Volume Analysis**: Daily volume and 5-day average volume filtering
- **Percent Change**: Filter by daily price movement
- **Exchange Filtering**: NYSE, NASDAQ, and AMEX only (for safety)
- **Technical Indicators**: Simple Moving Averages (SMA)

### 📊 **Volume Surge Detection**
- **N Times Volume**: Detect when current volume is N times higher than historical average
- **Configurable Periods**: Analyze volume surges over 5, 10, 20+ day periods
- **Real-time Analysis**: Compare current trading activity to historical patterns

### 🏛️ **Exchange Support**
- **Safe Exchanges Only**: NYSE, NASDAQ, and AMEX filtering
- **Asset Discovery**: Automatic discovery of tradable symbols by exchange
- **Compliance Focus**: Avoids risky or illiquid exchanges

### 💾 **Export Options**
- **CSV Export**: Spreadsheet-compatible results
- **JSON Export**: Structured data with metadata and scan criteria
- **Console Display**: Formatted table output with key metrics

## Installation

### Prerequisites
- Python 3.8+
- Alpaca Trading Account (Paper or Live)
- Required Python packages (installed via conda environment)

### Setup
1. **Environment Setup**:
   ```bash
   # Activate the conda environment
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca
   ```

2. **API Credentials**:
   - Credentials are managed via the existing `code/alpaca_config.py` configuration
   - Supports multiple accounts (Bruce, Dale Wilson, Janice)
   - Supports multiple environments (paper, live, cash)

## Usage

### Basic Commands

```bash
# Basic screening with price and volume filters
python code/alpaca_screener.py --min-price 0.75 --min-volume 1000000

# Volume surge detection (2x volume over 5 days)
python code/alpaca_screener.py --volume-surge 2.0 --surge-days 5

# Exchange filtering (NYSE, NASDAQ, and AMEX)
python code/alpaca_screener.py --exchanges NYSE NASDAQ AMEX --min-price 1.0 --max-price 50.0
python code/alpaca_screener.py --exchanges AMEX --min-price 1.0 --max-price 10.0

# Analyze specific symbols
python code/alpaca_screener.py --symbols AAPL TSLA NVDA --volume-surge 1.5 --verbose

# Get top gainers and losers
python code/alpaca_screener.py --top-gainers 10 --min-volume 100000 --verbose
python code/alpaca_screener.py --top-losers 5 --exchanges NYSE NASDAQ --verbose

# Export results
python code/alpaca_screener.py --min-volume 500000 --export-csv results.csv --export-json results.json
```

### Advanced Examples

```bash
# Complex screening with multiple criteria
python code/alpaca_screener.py \
  --exchanges NYSE NASDAQ AMEX \
  --min-price 5.0 \
  --max-price 200.0 \
  --min-volume 500000 \
  --volume-surge 3.0 \
  --surge-days 10 \
  --min-percent-change 2.0 \
  --verbose

# Account-specific screening
python code/alpaca_screener.py \
  --account-name "Dale Wilson" \
  --account paper \
  --min-volume 1000000 \
  --export-json daily_screen.json

# Technical analysis with SMA
python code/alpaca_screener.py \
  --min-price 10.0 \
  --sma-periods 20 50 200 \
  --verbose
```

## Command Line Options

### Account Configuration
- `--provider`: API provider (default: alpaca)
- `--account-name`: Account name (Bruce, Dale Wilson, Janice)
- `--account`: Account type (paper, live, cash)

### Screening Criteria
- `--min-price`: Minimum stock price (USD)
- `--max-price`: Maximum stock price (USD)
- `--min-volume`: Minimum daily volume (shares)
- `--min-avg-volume-5d`: Minimum 5-day average volume (shares)
- `--min-percent-change`: Minimum percent change (%)
- `--max-percent-change`: Maximum percent change (%)
- `--min-trades`: Minimum number of trades

### Volume Surge Detection
- `--volume-surge`: Volume surge multiplier (e.g., 2.0 for 2x)
- `--surge-days`: Days for volume surge calculation (default: 5)

### Exchange Filtering
- `--exchanges`: Filter by exchanges (NYSE, NASDAQ, AMEX only)

### Specific Symbol Analysis
- `--symbols`: Analyze specific symbols only (e.g., AAPL MSFT TSLA)

### Top Performers
- `--top-gainers`: Get top N gainers sorted by percent change (e.g., --top-gainers 10)
- `--top-losers`: Get top N losers sorted by percent change (e.g., --top-losers 5)

### Technical Analysis
- `--sma-periods`: SMA periods to calculate (e.g., 20 50 200)

### Data Source & Limits
- `--max-symbols`: Maximum symbols to analyze (default: 3000)
- `--feed`: Data feed to use (iex, sip, boats - default: iex)

### Output Options
- `--export-csv`: Export results to CSV file
- `--export-json`: Export results to JSON file
- `--verbose`, `-v`: Verbose output with progress tracking

## Output Formats

### Console Output
```
Alpaca Stock Screener Results
================================================================================
Scan completed at: 2025-08-27 15:34:31
Results found: 4 stocks

Symbol   Price    Volume       %Change  $Volume      Range    Surge
--------------------------------------------------------------------------
AAPL     $230.50  952,464       +0.52% $219.5M      $2.60    No
GOOGL    $207.42  778,382       +0.12% $161.5M      $3.19    No
MSFT     $506.69  568,567       +0.92% $288.1M      $7.23    No
NVDA     $181.57  2.6M          -0.07% $474.9M      $3.38    Yes (3.2x)
```

### JSON Export Structure
```json
{
  "scan_metadata": {
    "timestamp": "2025-08-27T15:34:56.188520",
    "total_symbols_scanned": 5,
    "results_count": 4,
    "account": "Bruce",
    "environment": "paper",
    "criteria": {
      "min_price": 1.0,
      "min_volume": 500000,
      "volume_surge_multiplier": 2.0,
      "volume_surge_days": 5
    }
  },
  "results": [
    {
      "symbol": "AAPL",
      "price": 230.5,
      "volume": 952464,
      "percent_change": 0.52,
      "dollar_volume": 219542952.0,
      "day_range": 2.60,
      "volume_surge_detected": false,
      "volume_surge_ratio": null,
      "avg_volume_5d": 936784.2,
      "avg_range_5d": 3.35
    }
  ]
}
```

## Architecture

### Integration with Existing Infrastructure
The screener leverages existing Alpaca trading infrastructure:

- **Client Initialization**: Uses `atoms/api/init_alpaca_client.py`
- **Configuration Management**: Integrates with `code/alpaca_config.py`
- **Multi-Account Support**: Bruce, Dale Wilson, Janice configurations
- **Environment Flexibility**: Paper, live, cash trading environments

### Core Components

#### Data Classes
- **`ScreeningCriteria`**: Configuration for screening parameters
- **`StockResult`**: Individual stock screening results
- **`VolumeSurge`**: Volume surge analysis data

#### Main Class
- **`AlpacaScreener`**: Core screening engine with rate limiting and error handling

### Volume Surge Algorithm
```python
# Detect if current volume is N times higher than M-day average
current_volume = today's_volume
avg_volume = sum(last_M_days_volume) / M_days
surge_ratio = current_volume / avg_volume
volume_surge_detected = surge_ratio >= N_multiplier
```

## Safety & Compliance

### Exchange Restrictions
- **NYSE, NASDAQ, and AMEX Only**: Prevents trading on risky or illiquid exchanges
- **Validation**: Command line and runtime validation
- **Tradable Assets**: Filters for `tradable=True` and `status='active'`

### Rate Limiting
- **API Protection**: Built-in rate limiting (200 calls/minute)
- **Batch Processing**: Efficient symbol processing in batches
- **Error Handling**: Graceful degradation on API errors

### Data Quality
- **Multiple Fallbacks**: Fallback symbol lists if API unavailable
- **Data Validation**: Comprehensive error checking
- **Historical Data**: Configurable lookback periods

## Common Use Cases

### 1. **Daily Market Screen**
```bash
python code/alpaca_screener.py \
  --min-volume 1000000 \
  --min-price 1.0 \
  --max-price 100.0 \
  --export-json daily_scan.json
```

### 2. **Volume Breakout Detection**
```bash
python code/alpaca_screener.py \
  --volume-surge 2.5 \
  --surge-days 5 \
  --exchanges NYSE NASDAQ AMEX \
  --verbose
```

### 3. **High-Volume Momentum Plays**
```bash
python code/alpaca_screener.py \
  --min-volume 2000000 \
  --min-percent-change 3.0 \
  --max-percent-change 15.0 \
  --volume-surge 1.5
```

### 4. **Technical Analysis Setup**
```bash
python code/alpaca_screener.py \
  --min-price 10.0 \
  --sma-periods 20 50 \
  --min-volume 500000 \
  --export-csv technical_screen.csv
```

### 5. **Specific Symbol Analysis**
```bash
# Analyze single symbol with volume surge detection
python code/alpaca_screener.py \
  --symbols AAPL \
  --volume-surge 2.0 \
  --surge-days 5 \
  --sma-periods 20 50 200 \
  --export-json aapl_analysis.json

# Compare multiple symbols
python code/alpaca_screener.py \
  --symbols AAPL MSFT GOOGL TSLA \
  --verbose
```

### 6. **Top Performers Scanning**
```bash
# Get top 10 gainers with decent volume
python code/alpaca_screener.py \
  --top-gainers 10 \
  --min-volume 100000 \
  --exchanges NYSE NASDAQ \
  --export-json top_gainers.json

# Get top 5 losers from all exchanges
python code/alpaca_screener.py \
  --top-losers 5 \
  --min-price 5.0 \
  --verbose

# Combine with other filters for quality movers
python code/alpaca_screener.py \
  --top-gainers 20 \
  --min-volume 500000 \
  --volume-surge 1.5 \
  --export-csv quality_gainers.csv

# Comprehensive NASDAQ scan (covers all ~5,000 NASDAQ symbols)
python code/alpaca_screener.py --top-gainers 50 --exchanges NASDAQ --max-symbols 6000 --min-volume 10000 --verbose
```

## Performance Optimization

### Efficient Data Collection
- **Batch Processing**: Processes symbols in configurable batches
- **Rate Limiting**: Stays within API limits automatically
- **Parallel Processing**: Concurrent API requests where possible

### Caching Strategy
- **Historical Data**: Reuses historical bars for multiple calculations
- **Symbol Universe**: Caches active symbol lists
- **Configuration**: Persistent account configurations

## Troubleshooting

### Common Issues

1. **No symbols found**:
   - Check API credentials in `code/alpaca_config.py`
   - Verify account has market data permissions
   - Try increasing `--max-symbols` limit

2. **Rate limiting errors**:
   - Reduce batch size or add delays
   - Check API subscription limits
   - Use `--verbose` to monitor API calls

3. **Exchange filtering returns no results**:
   - Verify exchange names (NYSE, NASDAQ only)
   - Check if symbols are tradable on specified exchanges
   - Try removing exchange filter to test

### Debug Mode
```bash
# Enable verbose logging
python code/alpaca_screener.py --verbose --max-symbols 10 --min-volume 100000
```

## Integration Examples

### Automated Screening Script
```bash
#!/bin/bash
# Daily automated screening
DATE=$(date +%Y-%m-%d)
python code/alpaca_screener.py \
  --min-volume 1000000 \
  --volume-surge 2.0 \
  --export-json "screens/daily_${DATE}.json" \
  --verbose
```

### Custom Analysis Pipeline
```python
from code.alpaca_screener import AlpacaScreener, ScreeningCriteria

# Initialize screener
screener = AlpacaScreener(account="Bruce", environment="paper", verbose=True)

# Define criteria
criteria = ScreeningCriteria(
    min_price=5.0,
    min_volume=500000,
    volume_surge_multiplier=2.0,
    volume_surge_days=5,
    exchanges=['NYSE', 'NASDAQ']
)

# Run screening
results = screener.screen_stocks(criteria)

# Process results
for stock in results:
    if stock.volume_surge_detected:
        print(f"{stock.symbol}: {stock.volume_surge_ratio:.1f}x volume surge!")
```

## License

This project is part of the Alpaca trading system and follows the same licensing terms as the parent project.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the existing Alpaca trading documentation
3. Ensure API credentials and permissions are properly configured
4. Use `--verbose` mode to diagnose API-related issues

---

**⚠️ Risk Disclaimer**: This tool is for screening and analysis purposes only. Always perform your own due diligence before making trading decisions. Past performance does not guarantee future results.