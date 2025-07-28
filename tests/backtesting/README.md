# ORB Alerts Backtesting Framework

## Overview

This directory contains comprehensive backtesting tools for the ORB (Opening Range Breakout) alerts system. The framework enables historical analysis, strategy validation, and performance evaluation using real market data.

## Files

### `alerts_backtest.py`
Main backtesting script for the basic ORB alerts system (`code/orb_alerts.py`):
- Chronological market data replay simulation
- ORB level calculation (fixed 15-minute period)
- Breakout detection and confidence scoring
- Alert generation with proper timing
- Comparison with existing alerts
- Comprehensive statistics and metrics

### `superduper_alerts_backtest.py`
Backtesting framework for the superduper alerts system:
- Single symbol backtesting
- Batch processing for multiple symbols
- Analysis-only mode (no alert generation)
- Telegram integration with configurable limits
- Comprehensive statistics and reporting

## Usage Examples

## Basic ORB Alerts Backtesting (`alerts_backtest.py`)

### Basic Usage
```bash
# Basic backtest for specific symbol and date
python3 tests/backtesting/alerts_backtest.py --symbol MCVT --date 2025-07-25

# Verbose output with detailed logging
python3 tests/backtesting/alerts_backtest.py --symbol AAPL --date 2025-07-23 --verbose

# Compare generated alerts with existing alerts
python3 tests/backtesting/alerts_backtest.py --symbol MCVT --date 2025-07-25 --compare

# Analysis mode - only analyze existing data without generating new alerts
python3 tests/backtesting/alerts_backtest.py --symbol AAPL --date 2025-07-23 --analysis-only
```

### Arguments for `alerts_backtest.py`
| Argument | Description | Required |
|----------|-------------|----------|
| `--symbol` | Stock symbol to backtest | Yes |
| `--date` | Date in YYYY-MM-DD format | Yes |
| `--verbose` | Enable detailed logging | No |
| `--compare` | Compare generated alerts with existing alerts | No |
| `--analysis-only` | Only analyze existing data without generating new alerts | No |

### Output Example for `alerts_backtest.py`
```
================================================================================
📊 ORB ALERTS BACKTEST RESULTS
================================================================================
🔍 Symbol: MCVT
📅 Date: 2025-07-25
⏱️ ORB Period: 15 minutes
📊 Market Data Loaded: ✅
📈 ORB Calculated: ✅

🚨 Alert Generation:
  Generated Alerts: 3
  Existing Alerts: 2

📈 Breakout Types:
  Generated:
    bullish_breakout: 2
    bearish_breakdown: 1
  Existing:
    bullish_breakout: 2

🎯 Confidence Statistics:
  Generated: Mean=0.742, Min=0.650, Max=0.850
  Existing: Mean=0.768, Min=0.720, Max=0.816

🔍 Comparison Results:
  Matches: 2
  Missing in Generated: 0
  Extra in Generated: 1
  Recall: 1.000
  Precision: 0.667
  F1 Score: 0.800
================================================================================
```

### Metrics Explained for `alerts_backtest.py`

- **Recall**: Percentage of existing alerts that were reproduced (existing alerts found / total existing alerts)
- **Precision**: Percentage of generated alerts that match existing ones (matches / total generated alerts)
- **F1 Score**: Harmonic mean of precision and recall
- **Matches**: Alerts that appear in both generated and existing sets
- **Missing in Generated**: Existing alerts that weren't reproduced
- **Extra in Generated**: New alerts not in the existing set

### Technical Details for `alerts_backtest.py`

#### ORB Calculation
The framework calculates Opening Range Breakout levels using the first 15 minutes of trading data (9:30-9:45 AM ET):
- **ORB High**: Highest price during the opening range
- **ORB Low**: Lowest price during the opening range
- **ORB Range**: Difference between high and low

#### Breakout Detection
Breakouts are detected when:
- Price moves above ORB high (bullish breakout)
- Price moves below ORB low (bearish breakdown)
- Volume and confidence criteria are met

#### Confidence Scoring
Alerts are scored based on:
- Volume ratio vs average volume
- Technical indicators (EMAs, momentum)
- Breakout strength and sustainability
- Market context and timing

### Data Structure for `alerts_backtest.py`

Expected directory structure:
```
historical_data/
└── YYYY-MM-DD/
    ├── market_data/
    │   └── SYMBOL_YYYYMMDD_HHMMSS.csv
    └── alerts/
        ├── bullish/
        │   └── alert_SYMBOL_YYYYMMDD_HHMMSS.json
        └── bearish/
            └── alert_SYMBOL_YYYYMMDD_HHMMSS.json
```

#### Market Data Format
CSV files with columns:
- `timestamp`: ISO format timestamp
- `symbol`: Stock symbol
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume
- `vwap`: Volume-weighted average price
- `trade_count`: Number of trades

#### Alert Data Format
JSON files with alert information:
- `symbol`: Stock symbol
- `timestamp`: Alert generation time
- `current_price`: Price at alert time
- `orb_high`: ORB high level
- `orb_low`: ORB low level
- `breakout_type`: Type of breakout (bullish_breakout/bearish_breakdown)
- `confidence_score`: Alert confidence score
- `volume_ratio`: Volume ratio vs average
- Additional technical indicators

## Superduper Alerts Backtesting (`superduper_alerts_backtest.py`)

### Basic Symbol Backtest
```bash
# Test MCVT for July 25, 2025
python3 tests/backtesting/superduper_alerts_backtest.py --symbol MCVT --date 2025-07-25

# Custom timeframe and alert limits
python3 tests/backtesting/superduper_alerts_backtest.py --symbol AAPL --date 2025-07-23 --timeframe 60 --max-alerts 5
```

### Dry Run Mode
```bash
# Analysis without Telegram notifications
python3 tests/backtesting/superduper_alerts_backtest.py --symbol VWAV --date 2025-07-28 --dry-run
```

### Batch Processing
```bash
# Backtest all symbols for a specific date
python3 tests/backtesting/superduper_alerts_backtest.py --date 2025-07-25 --batch-mode

# Batch with limited alerts per symbol
python3 tests/backtesting/superduper_alerts_backtest.py --date 2025-07-25 --batch-mode --max-alerts 3
```

### Analysis-Only Mode
```bash
# Trend analysis without creating alerts
python3 tests/backtesting/superduper_alerts_backtest.py --symbol MCVT --date 2025-07-25 --analysis-only
```

## Command Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--symbol` | Stock symbol to analyze | None | Yes (unless batch) |
| `--date` | Date in YYYY-MM-DD format | None | Yes |
| `--timeframe` | Analysis window in minutes | 45 | No |
| `--max-alerts` | Max Telegram alerts to send | 10 | No |
| `--dry-run` | Skip Telegram notifications | False | No |
| `--batch-mode` | Process all symbols for date | False | No |
| `--analysis-only` | Only perform analysis | False | No |
| `--verbose` | Enable detailed logging | False | No |

## Output Examples

### Single Symbol Results
```
================================================================================
📊 BACKTEST SUMMARY
================================================================================
📅 Date: 2025-07-25
🔍 Symbol: MCVT
📁 Super alerts found: 30
⚙️ Files processed: 30
🎯 Superduper alerts created: 10
📱 Telegram notifications sent: 10
🚫 Alerts filtered: 19
💰 Price range: $3.4900 → $4.2350
🎯 Penetration range: 48.9% → 100.0%

🔧 Overall Statistics:
  Symbols processed: 1
  Super alerts processed: 30
  Superduper alerts created: 10
  Telegram notifications sent: 10
  Alerts filtered: 19
  Processing errors: 0
================================================================================
```

### Batch Results
```
================================================================================
📊 BACKTEST SUMMARY
================================================================================
📅 Date: 2025-07-25
🔍 Symbols found: 15
⚙️ Symbols processed: 15

📈 Per-Symbol Results:
  AAPL:
    📁 Super alerts: 25
    🎯 Superduper alerts: 8
    📱 Telegram sent: 5
  MCVT:
    📁 Super alerts: 30
    🎯 Superduper alerts: 10
    📱 Telegram sent: 5
  ...

🔧 Overall Statistics:
  Symbols processed: 15
  Super alerts processed: 425
  Superduper alerts created: 85
  Telegram notifications sent: 75
  Alerts filtered: 340
  Processing errors: 0
================================================================================
```

## Features

### Chronological Processing
- Processes super alerts in timestamp order
- Implements backtesting-compatible filtering
- Prevents future data leakage

### Comprehensive Analysis
- Trend type classification (rising/consolidating)
- Momentum scoring and strength calculation
- Price progression tracking
- Penetration analysis

### Flexible Configuration
- Configurable timeframes (15-120 minutes)
- Alert quantity limits
- Urgency classification
- Test modes and dry runs

### Statistical Reporting
- Processing statistics
- Success/failure rates
- Price movement summaries
- Trend analysis metrics

## Integration

### Data Requirements
Requires existing super alerts in the standard directory structure:
```
historical_data/YYYY-MM-DD/super_alerts/bullish/
└── super_alert_SYMBOL_YYYYMMDD_HHMMSS.json
```

### Output Structure
Generates superduper alerts in:
```
historical_data/YYYY-MM-DD/superduper_alerts/bullish/
└── superduper_alert_SYMBOL_YYYYMMDD_HHMMSS.json
```

### Telegram Integration
- Respects rate limits and user preferences
- Configurable urgency thresholds
- Rich message formatting
- Success/failure tracking

## Performance Characteristics

### Processing Speed
- ~100ms per super alert analysis
- Efficient chronological filtering
- Minimal memory footprint

### Scalability
- Handles 1000+ alerts per run
- Batch processing for multiple symbols
- Configurable resource limits

### Accuracy
- Historically accurate backtesting
- No future data leakage
- Realistic trend analysis

## Advanced Usage

### Custom Analysis Scripts
```python
from tests.backtesting.superduper_alerts_backtest import SuperduperAlertsBacktester

# Initialize backtester
backtester = SuperduperAlertsBacktester(timeframe_minutes=30, dry_run=True)

# Run custom analysis
results = backtester.backtest_symbol('AAPL', '2025-07-25', max_alerts=5)

# Access detailed data
for analysis in results['trend_analysis']:
    print(f"Time: {analysis['timestamp']}")
    print(f"Trend: {analysis['trend_type']} ({analysis['trend_strength']:.2f})")
    print(f"Price: ${analysis['current_price']:.4f}")
```

### Strategy Validation
```bash
# Test different timeframes
for timeframe in 30 45 60; do
    python3 tests/backtesting/superduper_alerts_backtest.py \
        --symbol MCVT --date 2025-07-25 \
        --timeframe $timeframe --analysis-only
done

# Compare batch results across dates
for date in 2025-07-23 2025-07-24 2025-07-25; do
    python3 tests/backtesting/superduper_alerts_backtest.py \
        --date $date --batch-mode --dry-run
done
```

### Troubleshooting `alerts_backtest.py`

#### Common Issues

1. **No market data found**
   - Ensure historical data exists in `historical_data/YYYY-MM-DD/market_data/`
   - Check that CSV files follow the expected naming convention
   - Verify the date format is YYYY-MM-DD

2. **ORB calculation failed**
   - Verify market data has sufficient records during ORB period (9:30-9:45 AM)
   - Check data quality and completeness
   - Ensure timestamp format is correct

3. **No alerts generated**
   - This may be normal if no significant breakouts occurred
   - Use `--verbose` to see detailed processing logs
   - Verify confidence thresholds are appropriate

4. **Interface errors**
   - Ensure all atom dependencies are properly installed
   - Check that the ORB calculator interface matches expected parameters
   - Verify breakout detector configuration

#### Debug Mode
Enable verbose logging to see detailed processing information:
```bash
python3 tests/backtesting/alerts_backtest.py --symbol MCVT --date 2025-07-25 --verbose
```

This will show:
- Data loading progress
- ORB calculation details
- Breakout detection logic
- Confidence scoring steps
- Alert generation process

### Integration for `alerts_backtest.py`

The backtesting framework integrates with the main ORB alerts system components:
- `atoms/indicators/orb_calculator.py`: ORB level calculations
- `atoms/alerts/breakout_detector.py`: Breakout signal detection
- `atoms/alerts/confidence_scorer.py`: Alert confidence scoring
- `atoms/alerts/alert_formatter.py`: Alert formatting and output
- `atoms/websocket/data_buffer.py`: Market data management

### Performance for `alerts_backtest.py`

The framework is optimized for:
- **Memory efficiency**: Processes data chronologically without loading entire datasets
- **Accuracy**: Prevents future data leakage through chronological filtering
- **Speed**: Efficient data structures and minimal I/O operations

Typical performance:
- ~1000 data points per second
- Memory usage scales with data buffer size
- Processing time proportional to market session length

## Troubleshooting

### Common Issues

**No super alerts found**
- Verify date format (YYYY-MM-DD)
- Check if super alerts exist for the date
- Ensure correct symbol spelling

**Import errors**
- Activate conda environment: `conda activate alpaca`
- Verify project path configuration
- Check dependencies installation

**Telegram failures**
- Verify bot configuration
- Check network connectivity
- Review rate limiting settings

### Debug Mode
```bash
# Enable verbose logging
python3 tests/backtesting/superduper_alerts_backtest.py \
    --symbol MCVT --date 2025-07-25 --verbose
```

## Future Enhancements

### Planned Features
- Multi-date range backtesting
- Performance metric calculations
- Strategy comparison tools
- Export to CSV/JSON formats
- Visualization integration

### Extension Points
- Custom trend classifiers
- Alternative scoring algorithms
- Portfolio-level analysis
- Risk-adjusted returns calculation

## Contributing

When adding new backtesting features:
1. Maintain chronological accuracy
2. Add comprehensive error handling
3. Include statistical tracking
4. Update documentation
5. Add usage examples

## License

This backtesting framework is part of the ORB trading system and follows the same licensing terms as the main project.