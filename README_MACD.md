# MACD Analysis System

This directory contains a comprehensive MACD (Moving Average Convergence Divergence) analysis system for evaluating trading alerts and momentum indicators.

## Overview

The MACD system consists of three interconnected components that provide both real-time analysis capabilities and historical performance evaluation:

```
atoms/utils/calculate_macd.py          ← Core MACD calculations
    ↓ (used by)
atoms/utils/macd_alert_scorer.py       ← Real-time alert scoring
    ↓ (used by)  
atoms/utils/analyze_macd_scores.py     ← Historical analysis tool
```

## File Descriptions

### 1. `atoms/utils/calculate_macd.py` - Core MACD Engine
**Purpose**: Foundational MACD calculation utility

**Features**:
- Standard MACD calculation (12, 26, 9 periods)
- Returns MACD line, signal line, and histogram
- Supports multiple price sources (close, open, high, low, typical)
- Both pandas and manual calculation methods
- Handles insufficient data gracefully

**Usage**:
```python
from atoms.utils.calculate_macd import calculate_macd
success, macd_data = calculate_macd(df, fast_length=12, slow_length=26, signal_length=9)
```

**Real-time Suitability**: ✅ **Excellent** - Optimized for real-time use with JIT compilation support

### 2. `atoms/utils/macd_alert_scorer.py` - Alert Scoring System
**Purpose**: MACD-based scoring system for trading alerts using Green/Yellow/Red classification

**Scoring Methodology**:
- **GREEN (3-4/4 conditions)**: Excellent MACD setup
  - MACD > Signal Line (bullish crossover)
  - MACD > 0 (positive momentum)
  - Histogram > 0 (increasing momentum)
  - MACD Rising (5-period upward trend)

- **YELLOW (2/4 conditions)**: Moderate MACD setup
- **RED (0-1/4 conditions)**: Poor MACD setup

**Usage**:
```python
from atoms.utils.macd_alert_scorer import MACDAlertScorer
scorer = MACDAlertScorer()
scored_alerts = scorer.score_alerts_batch(market_data_df, alerts_list)
```

**Real-time Suitability**: ✅ **Excellent** - Designed for real-time alert evaluation
- Efficient batch processing
- Timezone-aware timestamp matching
- 5-minute tolerance for market data alignment
- Comprehensive error handling

### 3. `atoms/utils/analyze_macd_scores.py` - Historical Analysis Tool
**Purpose**: Comprehensive retrospective analysis of all historical trading alerts

**What It Does**:
1. **Alert Discovery** 📋
   - Scans `historical_data/` directory for all alert files
   - Finds patterns: `superduper_alert_SYMBOL_YYYYMMDD_HHMMSS.json`
   - Identifies unique symbol/date combinations

2. **MACD Scoring** 🎯
   - Applies 4-condition MACD scoring to each historical alert
   - Uses the same Green/Yellow/Red system as real-time scorer

3. **Comprehensive Reporting** 📈
   - Overall summary with percentage breakdowns
   - Symbol-by-symbol performance analysis
   - Date-based performance trends

**Sample Output**:
```
🔍 MACD ALERT ANALYSIS
============================================================
Found 76 alert files
Processing 9 symbol/date combinations:

📊 OVERALL SUMMARY:
🟢 GREEN: 45 alerts (59.2%)
🟡 YELLOW: 20 alerts (26.3%)
🔴 RED: 11 alerts (14.5%)

📈 BY SYMBOL BREAKDOWN:
BTAI: 12 total alerts
  🟢 GREEN: 8 (66.7%)
  🟡 YELLOW: 3 (25.0%)
  🔴 RED: 1 (8.3%)
```

**Usage**:
```python
# As module
from atoms.utils.analyze_macd_scores import analyze_all_historical_alerts
analyze_all_historical_alerts()

# Or direct execution
python atoms/utils/analyze_macd_scores.py
```

**Real-time Suitability**: ❌ **Not Recommended** - Designed for batch historical analysis
- Processes all historical data (computationally expensive)
- Uses mock market data for missing information
- Intended for post-mortem analysis and strategy validation

## Real-Time Integration Recommendations

### ✅ **Recommended for Real-Time Alert System**:

1. **`calculate_macd.py`** - Essential for real-time MACD calculations
   - Fast, optimized calculations
   - Handles live market data streams
   - Minimal computational overhead

2. **`macd_alert_scorer.py`** - Perfect for real-time alert filtering
   - Immediate alert quality assessment
   - Filters alerts based on MACD momentum
   - Helps prioritize high-quality trading opportunities

### 📊 **Integration Example**:
```python
# In your real-time alert system
from atoms.utils.macd_alert_scorer import MACDAlertScorer

scorer = MACDAlertScorer()

# When new alert is generated:
def process_new_alert(alert, market_data):
    # Score the alert using MACD
    macd_analysis = scorer.calculate_macd_conditions(market_data, alert['timestamp'])
    score_result = scorer.score_alert(macd_analysis)
    
    # Only proceed with GREEN alerts
    if score_result['color'] == 'green':
        execute_trade_logic(alert)
    else:
        log_filtered_alert(alert, score_result['reasoning'])
```

### ❌ **Not for Real-Time**:

**`analyze_macd_scores.py`** - Use for periodic analysis only
- Run weekly/monthly for performance review
- Validate alert system effectiveness
- Identify patterns in historical data

## Business Value

- **Quality Control**: Ensures alerts occur during favorable MACD conditions
- **Performance Validation**: Quantifies the percentage of high-quality historical alerts
- **Strategy Optimization**: Identifies which symbols/timeframes produce better MACD setups
- **Risk Management**: Filters out alerts with poor momentum characteristics

## Dependencies

- pandas
- numpy  
- pytz (timezone handling)
- pathlib (file operations)

## Configuration

Default MACD parameters (configurable):
- Fast EMA: 12 periods
- Slow EMA: 26 periods  
- Signal EMA: 9 periods
- Trend lookback: 5 periods

## Getting Started

1. **For real-time integration**: Start with `macd_alert_scorer.py`
2. **For historical analysis**: Run `analyze_macd_scores.py`
3. **For custom calculations**: Use `calculate_macd.py` directly

This MACD system provides both the real-time filtering capabilities needed for your alert system and the analytical tools to validate and optimize your trading strategy over time.