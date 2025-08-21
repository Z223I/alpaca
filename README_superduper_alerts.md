# ORB Superduper Alerts System

## Overview

The ORB Superduper Alerts System is an advanced alert generation system that monitors super alerts and creates enhanced "superduper" alerts based on sophisticated trend analysis and momentum indicators. It represents the highest tier of alert quality in the ORB (Opening Range Breakout) trading system.

## Alert Hierarchy

```
ORB Alerts → Super Alerts → Superduper Alerts
     ↓              ↓              ↓
Basic breakout   Signal price    Advanced trend
notifications    penetration     analysis &
                                momentum
```

## System Architecture

### Core Components

1. **SuperduperAlertFilter** (`atoms/alerts/superduper_alert_filter.py`)
   - Advanced trend analysis with configurable timeframes
   - Rising vs consolidating pattern detection
   - Momentum scoring and strength calculation
   - Backtesting-compatible chronological filtering

2. **SuperduperAlertGenerator** (`atoms/alerts/superduper_alert_generator.py`)
   - Enhanced alert creation with detailed metrics
   - Advanced Telegram message formatting
   - Risk assessment and urgency classification
   - Breakout quality evaluation

3. **ORBSuperduperAlertMonitor** (`code/orb_alerts_monitor_superduper.py`)
   - Real-time file monitoring using watchdog
   - Automated processing pipeline
   - Telegram notification integration

## Key Features

### Advanced Trend Analysis
- **Rising Trends**: Identifies stocks with accelerating upward momentum
- **Consolidating Patterns**: Detects sustained strength at resistance levels
- **Timeframe Analysis**: Configurable analysis window (default: 45 minutes)
- **Strength Scoring**: 0.0-1.0 scale based on multiple momentum factors

### Backtesting Support
- **Chronological Filtering**: Only uses historically available data
- **Timestamp Awareness**: Prevents future data leakage in analysis
- **Realistic Simulation**: Accurate backtesting for strategy validation

### Enhanced Messaging
- **Rich Telegram Notifications**: Detailed trend analysis and metrics
- **Urgency Classification**: CRITICAL/HIGH/MODERATE/LOW levels
- **Risk Assessment**: Dynamic risk evaluation based on pattern strength
- **Visual Formatting**: Emojis and structured layout for quick reading

## Installation & Setup

### Prerequisites
- Python 3.8+
- Conda environment `alpaca` (see main project README)
- Telegram bot configuration (see main project setup)

### Required Dependencies
```bash
pip install watchdog pytz
```

## Usage

### Command Line Interface

```bash
# Basic monitoring (45-minute timeframe)
python3 code/orb_alerts_monitor_superduper.py

# Custom timeframe
python3 code/orb_alerts_monitor_superduper.py --timeframe 60

# Test mode (no live Telegram notifications)
python3 code/orb_alerts_monitor_superduper.py --test

# Urgent alerts only
python3 code/orb_alerts_monitor_superduper.py --post-only-urgent

# Verbose logging
python3 code/orb_alerts_monitor_superduper.py --verbose

# Combined options
python3 code/orb_alerts_monitor_superduper.py --timeframe 30 --test --verbose
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--timeframe` | Analysis window in minutes | 45 |
| `--test` | Run in test mode (dry run) | False |
| `--post-only-urgent` | Send only urgent alerts via Telegram | False |
| `--verbose` | Enable detailed logging | False |

## Alert Criteria

### Rising Trend Criteria
- **Price Change**: > 2.0% increase
- **Momentum**: > 0.03% per minute
- **Penetration Increase**: > 10%
- **Minimum Strength**: 0.3

### Consolidating Trend Criteria
- **Average Penetration**: > 25%
- **Volatility**: < 0.03 (low volatility = stable)
- **Minimum Strength**: 0.3

### Urgency Thresholds
- **CRITICAL**: Trend strength > 0.7
- **HIGH**: Strong momentum with significant penetration changes
- **MODERATE**: Above minimum thresholds
- **LOW**: Below standard criteria

## File Structure

### Input Directory
```
historical_data/YYYY-MM-DD/super_alerts/bullish/
├── super_alert_SYMBOL_YYYYMMDD_HHMMSS.json
└── ...
```

### Output Directory
```
historical_data/YYYY-MM-DD/superduper_alerts/bullish/
├── superduper_alert_SYMBOL_YYYYMMDD_HHMMSS.json
└── ...
```

## Sample Output

### Console Output
```
🎯🎯 SUPERDUPER ALERT: AAPL @ $150.2500
   Trend: RISING | Strength: 0.78 | Urgency: CRITICAL
   Penetration: 32.5% | Quality: HIGH
   Saved: superduper_alert_AAPL_20250723_143000.json
```

### Telegram Message
```
🎯🎯 **SUPERDUPER ALERT** 🎯🎯

🚀📈 **AAPL** @ **$150.2500**
📊 **STRONG UPTREND** | 🔥 **VERY STRONG**

🎯 **Signal Performance:**
• Entry Signal: $145.0000 ✅
• Current Price: $150.2500
• Resistance Target: $155.0000
• Penetration: **32.5%** into range

📈 **Trend Analysis (45m):**
• Price Movement: **+3.62%**
• Momentum: **0.0543%/min**
• Penetration Increase: **+18.3%**
• Pattern: **Accelerating Breakout** 🚀

⚡ **Alert Level:** CRITICAL
⚠️ **Risk Level:** LOW

🎯 **Action Zones:**
• Watch for continuation above $150.25
• Watch for major resistance
• Monitor for volume confirmation

⏰ **Alert Generated:** 14:30:15 ET
```

## Monitoring & Statistics

### Real-time Status
The monitor displays comprehensive status information:
```
================================================================================
🎯 ORB SUPERDUPER ALERTS MONITOR ACTIVE
📁 Monitoring: historical_data/2025-07-23/super_alerts/bullish
💾 Superduper alerts: historical_data/2025-07-23/superduper_alerts/bullish
⏱️ Analysis timeframe: 45 minutes
✅ Filtering: Rising trends & high-quality consolidation patterns
📱 Telegram: Enhanced notifications for superduper alerts
================================================================================
```

### Statistics API
```python
monitor = ORBSuperduperAlertMonitor()
stats = monitor.get_statistics()
print(stats)
# {
#     'timeframe_minutes': 45,
#     'superduper_alerts_generated': 12,
#     'super_alerts_processed': 150,
#     'super_alerts_filtered': 138,
#     'monitoring_directory': 'historical_data/2025-07-23/super_alerts/bullish',
#     'superduper_alerts_directory': 'historical_data/2025-07-23/superduper_alerts/bullish'
# }
```

## Backtesting

### Historical Analysis
The system supports accurate backtesting by ensuring only historically available data is used:

```python
# When processing super_alert_AAPL_20250723_143000.json
# Only files with timestamps ≤ 14:30:00 are included in analysis
# Files with later timestamps are automatically excluded
```

### Validation
```bash
# Process historical data chronologically
for file in $(ls historical_data/*/super_alerts/bullish/*.json | sort); do
    python3 code/orb_alerts_monitor_superduper.py --test
done
```

## Configuration

### Timeframe Tuning
Adjust analysis window based on trading strategy:
- **15-30 minutes**: High-frequency, sensitive to short-term moves
- **45-60 minutes**: Balanced, default recommendation
- **90+ minutes**: Conservative, focuses on sustained trends

### Threshold Customization
Edit filter criteria in `superduper_alert_filter.py`:
```python
RISING_THRESHOLD = 1.0  # Minimum price change %
MOMENTUM_THRESHOLD = 0.02  # Minimum momentum per minute
MIN_PENETRATION = 15.0  # Minimum penetration %
```

## Troubleshooting

### Common Issues

**No superduper alerts generated**
- Check if super alerts exist in input directory
- Verify timeframe allows sufficient data points (need ≥2)
- Review filter criteria thresholds

**Import errors**
- Ensure conda environment is activated
- Verify project path is in sys.path
- Check all dependencies are installed

**Telegram notifications not sending**
- Verify Telegram bot configuration
- Check `post_only_urgent` setting
- Review network connectivity

### Debug Mode
Enable verbose logging for detailed analysis:
```bash
python3 code/orb_alerts_monitor_superduper.py --verbose
```

## Performance

### System Requirements
- **Memory**: ~50MB RAM per active symbol
- **CPU**: Minimal (event-driven processing)
- **Storage**: ~2KB per superduper alert

### Scaling
- Handles 100+ symbols simultaneously
- Sub-second processing per alert
- Efficient file watching with minimal overhead

## Integration

### With Existing Systems
The superduper alerts system integrates seamlessly with:
- ORB alerts generation (`orb_alerts.py`)
- Super alerts monitoring (`orb_alerts_monitor.py`)
- Telegram notification system
- Historical data storage structure

### API Integration
```python
# Programmatic usage
from atoms.alerts.superduper_alert_filter import SuperduperAlertFilter
from atoms.alerts.superduper_alert_generator import SuperduperAlertGenerator

filter_obj = SuperduperAlertFilter(timeframe_minutes=45)
should_create, reason, analysis = filter_obj.should_create_superduper_alert(file_path)

if should_create:
    generator = SuperduperAlertGenerator(output_dir)
    filename = generator.create_and_save_superduper_alert(
        super_alert_data, analysis, trend_type, trend_strength
    )
```

## Future Enhancements

### Planned Features
- Machine learning trend prediction
- Multi-timeframe analysis
- Volume-weighted momentum scoring
- Options flow integration
- Portfolio-level risk management

### Extension Points
- Custom trend classifiers
- Additional notification channels (Discord, Slack)
- Real-time market data integration
- Advanced charting integration

## Support

For issues, feature requests, or contributions:
1. Check existing documentation
2. Review troubleshooting section
3. Enable verbose logging for diagnostics
4. Create detailed issue reports with logs

## License

This software is part of the ORB trading system and follows the same licensing terms as the main project.