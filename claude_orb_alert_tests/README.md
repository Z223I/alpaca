# Claude ORB Alert Tests

This directory contains testing tools for analyzing ORB (Opening Range Breakout) patterns in historical stock data.

## Files

- `orb_analyzer.py` - Main analysis script that can process any symbol's historical data
- `README.md` - This documentation file

## Usage

### Basic Usage
```bash
# Analyze a symbol (auto-detects CSV file)
python3 orb_analyzer.py BSLK

# Analyze with specific CSV file
python3 orb_analyzer.py AAPL /path/to/AAPL_data.csv

# Custom ORB period
python3 orb_analyzer.py BSLK --orb-minutes 30
```

### Examples

#### Analyze BSLK data
```bash
cd claude_orb_alert_tests
python3 orb_analyzer.py BSLK
```

#### Analyze any symbol with auto-detection
```bash
python3 orb_analyzer.py WOLF
python3 orb_analyzer.py PROK
python3 orb_analyzer.py ZVSA
```

## Features

### Comprehensive ORB Analysis
- **Opening Range Metrics**: High, low, range, midpoint, volume
- **Breakout Detection**: Identifies when price breaks above/below ORB with configurable thresholds
- **Trading Assessment**: Evaluates setup quality and potential
- **Timezone Handling**: Properly converts to Eastern Time for market hours

### Auto-Detection
The script automatically searches for CSV files in these locations:
1. `historical_data/*/market_data/{SYMBOL}_*.csv`
2. `tmp/{SYMBOL}_*.csv`
3. `data/{SYMBOL}_*.csv`
4. `{SYMBOL}_*.csv`

### Output Analysis
- ORB period analysis (default 15 minutes)
- Breakout detection with timestamps and percentages
- Volume analysis and quality assessment
- Trading opportunity rating
- Session summary with key statistics

## CSV File Format

Expected CSV columns:
- `timestamp` - ISO format timestamp
- `symbol` - Trading symbol
- `high` - High price for the period
- `low` - Low price for the period  
- `close` - Closing price for the period
- `volume` - Volume traded
- `vwap` (optional) - Volume weighted average price
- `trade_count` (optional) - Number of trades

## Configuration

The analyzer uses the same configuration as the main ORB alert system:
- `breakout_threshold` - Percentage above/below ORB for breakout (default 0.2%)
- `orb_period_minutes` - ORB period length (default 15 minutes)
- `volume_multiplier` - Volume requirements for alerts

## Example Output

```
🔍 ORB ALERT ANALYZER
Analyzing symbol: BSLK
✅ Loaded 29 data points from historical_data/2025-07-08/market_data/BSLK_20250708_090100.csv

================================================================================
📊 ORB ANALYSIS FOR BSLK
================================================================================
📈 OPENING RANGE ANALYSIS (First 15 minutes)
ORB High: $3.880
ORB Low: $3.135
ORB Range: $0.745 (21.24%)

🎯 BREAKOUT ANALYSIS
🚨 2 BREAKOUT(S) DETECTED:
  13:59:00 ET: HIGH BREAKOUT
    Price: $4.040 (+4.12%)
    Volume: 10,309

💡 TRADING ASSESSMENT
🎯 OVERALL: EXCELLENT ORB TRADING OPPORTUNITY
```