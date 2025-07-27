# Superduper Alerts Backtesting Framework

## Overview

This directory contains comprehensive backtesting tools for the Superduper Alerts system. The framework enables historical analysis, strategy validation, and performance evaluation using real market data.

## Files

### `superduper_alerts_backtest.py`
Main backtesting framework with multiple analysis modes:
- Single symbol backtesting
- Batch processing for multiple symbols
- Analysis-only mode (no alert generation)
- Telegram integration with configurable limits
- Comprehensive statistics and reporting

## Usage Examples

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
python3 tests/backtesting/superduper_alerts_backtest.py --symbol MCVT --date 2025-07-25 --dry-run
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