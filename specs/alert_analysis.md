# Product Requirements Document: Alert Performance Analysis

## Document Information
- **Version**: 1.3
- **Date**: 2025-07-10
- **Author**: Development Team
- **Status**: Draft

---

## Executive Summary

This PRD outlines the development of an Alert Performance Analysis system that evaluates the effectiveness of our ORB (Opening Range Breakout) alerts by analyzing historical market data against generated alerts to measure accuracy, profitability, and timing effectiveness.

## Problem Statement

Currently, we generate ORB alerts based on technical indicators, but we lack systematic analysis of how these alerts perform in practice. Without performance metrics, we cannot:

- Validate the effectiveness of our alert algorithms
- Optimize alert parameters and thresholds  
- Provide confidence metrics to users
- Identify which market conditions produce the best alerts
- Continuously improve our trading strategies

## Goals and Objectives

### Primary Goals
1. **Measure Alert Accuracy**: Determine how often alerts correctly predict profitable price movements
2. **Analyze Profitability**: Calculate theoretical P&L if trades were executed based on alerts
3. **Optimize Alert Parameters**: Identify which alert configurations produce the best results
4. **Risk Assessment**: Evaluate risk/reward ratios and maximum drawdowns

### Secondary Goals
1. **Performance Benchmarking**: Compare alert performance across different stocks and market conditions
2. **Timing Analysis**: Understand optimal entry/exit timing relative to alert generation
3. **Continuous Monitoring**: Enable ongoing performance tracking for new alerts

## User Stories

### As a Trader
- I want to see how profitable alerts have been historically so I can decide whether to act on them
- I want to understand the success rate of different alert priorities (High/Medium/Low)
- I want to know the average time it takes for alerts to become profitable

### As a Developer
- I want to identify which alert parameters produce the best results so I can optimize the algorithm
- I want to detect when alert performance degrades so I can investigate and fix issues
- I want to validate new alert features against historical data

### As a Risk Manager
- I want to understand the maximum potential losses from alert-based trades
- I want to see win/loss ratios and average returns per alert type
- I want to identify high-risk alert patterns

## Functional Requirements

### Core Analysis Features

#### 1. Alert Performance Tracking
- **FR-1.1**: Calculate success rate for each alert (percentage of profitable outcomes)
- **FR-1.2**: Measure time-to-profit for successful alerts
- **FR-1.3**: Track maximum favorable/adverse excursion for each alert
- **FR-1.4**: Calculate risk-adjusted returns (Sharpe ratio, Sortino ratio)

#### 2. Profitability Analysis
- **FR-2.1**: Simulate P&L assuming trades executed at alert prices with recommended stop-loss and take-profit
- **FR-2.2**: Calculate cumulative returns over different time periods
- **FR-2.3**: Analyze performance by position size (based on portfolio risk calculations)
- **FR-2.4**: Include transaction costs and slippage estimates

#### 3. Segmentation Analysis
- **FR-3.1**: Performance breakdown by alert priority (High/Medium/Low)
- **FR-3.2**: Performance by confidence level and breakout percentage
- **FR-3.3**: Performance by stock symbol and market capitalization
- **FR-3.4**: Performance by time of day and market session
- **FR-3.5**: Performance by market volatility conditions

#### 4. Risk Metrics
- **FR-4.1**: Calculate maximum drawdown periods
- **FR-4.2**: Analyze consecutive loss streaks
- **FR-4.3**: Measure volatility of returns
- **FR-4.4**: Calculate Value at Risk (VaR) metrics

### Data Processing Features

#### 5. Historical Data Integration
- **FR-5.1**: Parse and align alert timestamps with market data
- **FR-5.2**: Filter historical data to normal trading hours (9:30 AM to 4:00 PM ET)
- **FR-5.3**: Handle market gaps (weekends, holidays, after-hours)
- **FR-5.4**: Validate data quality and identify missing periods
- **FR-5.5**: Support multiple timeframes (1min, 5min, 15min, 1hour, daily)

#### 6. Alert Matching
- **FR-6.1**: Filter alerts to valid alert hours (9:30 AM to 3:30 PM ET)
- **FR-6.2**: Match alerts to corresponding market data timestamps
- **FR-6.3**: Define alert expiration criteria (e.g., end of trading day)
- **FR-6.4**: Handle multiple alerts for the same symbol
- **FR-6.5**: Track alert modifications or cancellations

### Reporting Features

#### 7. Performance Reports
- **FR-7.1**: Generate comprehensive performance summaries
- **FR-7.2**: Create visual charts and graphs for key metrics
- **FR-7.3**: Export results to CSV/JSON for further analysis
- **FR-7.4**: Support date range filtering and custom periods

#### 8. Real-time Monitoring
- **FR-8.1**: Track performance of recent alerts (last 24 hours, week, month)
- **FR-8.2**: Alert on performance degradation
- **FR-8.3**: Compare current performance to historical baselines

## Technical Requirements

### Data Sources
- **Historic market data**: `historical_data/YYYY-MM-DD/market_data/` CSV files
- **Alert data**: `alerts/` JSON files (bullish, bearish, and root directory)
- **Market metadata**: Summary files with trading session information

### Trading Hours Filter
- **Regular Trading Hours**: 9:30 AM to 4:00 PM Eastern Time (ET)
- **Valid Alert Hours**: 9:30 AM to 3:30 PM Eastern Time (ET)
- **Timezone Handling**: All timestamps must be converted to ET for consistency
- **Data Exclusion**: Pre-market (before 9:30 AM) and after-hours (after 4:00 PM) data will be filtered out
- **Alert Exclusion**: Alerts generated outside of valid alert hours (9:30 AM - 3:30 PM) will be excluded from analysis
- **Holiday Handling**: Market holidays will be excluded from analysis

### Analysis Timeframes
- **Immediate**: 5, 15, 30 minutes post-alert
- **Short-term**: 1, 2, 4 hours post-alert  
- **Intraday**: Until market close
- **Multi-day**: 1, 2, 3 trading days post-alert

### Performance Metrics Calculation

#### Success Criteria
- **Bullish Alert Success**: Price reaches take-profit before stop-loss
- **Bearish Alert Success**: Price reaches take-profit before stop-loss
- **Partial Success**: Price moves favorably but doesn't reach full target
- **Failure**: Price hits stop-loss before take-profit

#### Key Metrics
```
Success Rate = (Successful Alerts / Total Alerts) * 100
Average Return = Sum(Individual Returns) / Total Alerts
Profit Factor = Gross Profit / Gross Loss
Win/Loss Ratio = Average Win / Average Loss
Sharpe Ratio = (Average Return - Risk Free Rate) / Standard Deviation
Maximum Drawdown = Max(Peak - Trough) / Peak
```

### Data Pipeline Architecture
1. **Data Ingestion**: Load historic market data and alerts
2. **Time Filtering**: Filter market data to regular trading hours (9:30 AM - 4:00 PM ET)
3. **Alert Filtering**: Filter alerts to valid alert hours (9:30 AM - 3:30 PM ET)
4. **Timezone Conversion**: Ensure all timestamps are in Eastern Time
5. **Alignment**: Match alerts to market data timestamps
6. **Simulation**: Execute virtual trades based on alert parameters
7. **Calculation**: Compute performance metrics
8. **Aggregation**: Summarize results across multiple dimensions
9. **Visualization**: Generate charts and reports

## Implementation Specifications

### Architecture Design

This implementation follows the **atoms-molecules architecture pattern** used throughout this repository, emphasizing modularity and reusability.

#### Atoms (Reusable Components)
Small, focused functions that perform single responsibilities:

- **`atoms/analysis/`** - Core analysis functions
  - `filter_trading_hours.py` - Trading hours filtering utility (9:30 AM - 4:00 PM ET)
  - `filter_alert_hours.py` - Valid alert hours filtering utility (9:30 AM - 3:30 PM ET)
  - `calculate_returns.py` - Return calculation functions
  - `align_timestamps.py` - Timestamp alignment utilities
  - `validate_data.py` - Data quality validation functions

- **`atoms/metrics/`** - Financial metrics calculations
  - `success_rate.py` - Alert success rate calculations
  - `sharpe_ratio.py` - Risk-adjusted return metrics
  - `drawdown.py` - Maximum drawdown calculations
  - `profit_factor.py` - Profit factor and win/loss ratios

- **`atoms/simulation/`** - Trade simulation components
  - `trade_executor.py` - Individual trade simulation logic
  - `position_sizer.py` - Position sizing calculations
  - `cost_calculator.py` - Transaction cost and slippage models

#### Molecules (Combined Functionality)
Higher-level components that combine multiple atoms:

- **`molecules/alert_analyzer.py`** - Main analysis orchestrator
- **`molecules/performance_reporter.py`** - Report generation and visualization
- **`molecules/data_pipeline.py`** - Data loading and processing pipeline

### Core Components

#### 1. AlertAnalyzer Molecule
```python
# molecules/alert_analyzer.py
from atoms.api.get_cash import get_cash
from atoms.analysis.filter_trading_hours import filter_trading_hours
from atoms.analysis.filter_alert_hours import filter_alert_hours
from atoms.analysis.align_timestamps import align_alerts_to_market_data
from atoms.simulation.trade_executor import simulate_trade
from atoms.metrics.success_rate import calculate_success_rate

class AlertAnalyzer:
    def load_alerts(self, date_range)
    def load_market_data(self, date_range)
    def filter_trading_hours(self, start_time="09:30", end_time="16:00", timezone="US/Eastern")
    def filter_alert_hours(self, start_time="09:30", end_time="15:30", timezone="US/Eastern")
    def align_data(self)
    def simulate_trades(self)
    def calculate_metrics(self)
    def generate_report(self)
```

#### 2. Reusable Atoms Design
Each atom should be independently testable and reusable:

```python
# atoms/analysis/filter_trading_hours.py
def filter_trading_hours(df, start_time="09:30", end_time="16:00", timezone="US/Eastern"):
    """Reusable function to filter any DataFrame to trading hours"""
    pass

# atoms/analysis/filter_alert_hours.py
def filter_alert_hours(df, start_time="09:30", end_time="15:30", timezone="US/Eastern"):
    """Reusable function to filter alerts to valid alert hours"""
    pass

# atoms/metrics/success_rate.py  
def calculate_success_rate(trades_df):
    """Calculate success rate from any trades DataFrame"""
    pass

# atoms/simulation/trade_executor.py
def simulate_trade(alert, market_data, stop_loss, take_profit):
    """Simulate a single trade execution"""
    pass
```

#### 3. Benefits of Atoms-Molecules Architecture
- **Reusability**: Atoms can be used across different analysis scripts
- **Testability**: Each atom can be unit tested independently
- **Maintainability**: Changes to core logic only affect specific atoms
- **Composability**: New analysis types can combine existing atoms
- **Consistency**: Shared atoms ensure consistent calculations across features

### Command Line Interface
```bash
# Analyze specific date range (regular trading hours and valid alert hours only)
python3 alert_analyzer.py --start-date 2025-07-08 --end-date 2025-07-10

# Analyze specific symbols
python3 alert_analyzer.py --symbols PROK,STKH,CGTX

# Generate detailed report
python3 alert_analyzer.py --detailed --export-csv

# Real-time monitoring mode
python3 alert_analyzer.py --monitor --threshold 0.5

# Include extended hours (optional override)
python3 alert_analyzer.py --include-extended-hours --start-date 2025-07-08
```

### Output Format

#### Summary Report
```
ALERT PERFORMANCE ANALYSIS
==========================
Analysis Period: 2025-07-08 to 2025-07-10
Total Alerts: 252
Symbols Analyzed: 16

OVERALL PERFORMANCE
Success Rate: 68.5%
Average Return: +2.34%
Profit Factor: 1.8
Max Drawdown: -12.3%
Sharpe Ratio: 1.42

BREAKDOWN BY PRIORITY
High Priority: 72.1% success, +3.1% avg return
Medium Priority: 65.8% success, +1.9% avg return
Low Priority: 58.3% success, +1.2% avg return

TOP PERFORMING SYMBOLS
STKH: 85.0% success, +4.2% avg return (10 alerts)
PROK: 71.4% success, +2.8% avg return (33 alerts)
CGTX: 66.7% success, +2.1% avg return (15 alerts)
```

#### Detailed Trade Log
```csv
Alert_ID,Symbol,Timestamp,Type,Entry_Price,Exit_Price,Stop_Loss,Take_Profit,Duration,Return,Status
STKH_20250710_105100,STKH,2025-07-10T10:51:00,bullish,3.50,3.64,3.24,3.64,45min,+4.0%,SUCCESS
CGTX_20250710_110000,CGTX,2025-07-10T11:00:00,bearish,0.7038,0.6756,0.7565,0.6756,23min,-7.5%,STOPPED_OUT
```

## Success Metrics

### Quantitative Metrics
- **Alert Accuracy**: >65% success rate for High priority alerts
- **Profitability**: Positive cumulative returns over 30-day periods
- **Risk Management**: Maximum drawdown <15% of capital
- **Timing**: Average time-to-profit <2 hours for successful alerts

### Qualitative Metrics
- Clear identification of best-performing alert configurations
- Actionable insights for algorithm optimization
- Reliable performance monitoring for live trading

## Dependencies

### Existing Atoms (Reusable Components)
- **`atoms/api/`** - Existing API interaction functions
  - `get_cash.py` - Cash balance retrieval
  - `get_positions.py` - Position data access
  - `get_latest_quote.py` - Market quote functions
- **`atoms/utils/`** - Existing utility functions
  - `parse_args.py` - Command line argument parsing
  - `delay.py` - Timing utilities
  - `read_csv.py` - CSV file reading functions

### Existing Systems
- Historical market data collection system
- Alert generation system (`molecules/orb_alert_engine.py`)
- Data storage infrastructure (CSV files, JSON alerts)

### New Dependencies
- Statistical libraries (pandas, numpy, scipy)
- Visualization libraries (matplotlib, plotly)
- Financial metrics libraries (quantlib, pyfolio)

### New Atoms to Develop
- **`atoms/analysis/`** - New analysis functions (as specified above)
- **`atoms/metrics/`** - New financial metrics calculations
- **`atoms/simulation/`** - New trade simulation components

## Risks and Mitigations

### Technical Risks
- **Data Quality Issues**: Implement robust data validation and cleaning
- **Performance Overhead**: Use efficient data structures and caching
- **Timestamp Alignment**: Careful handling of timezone conversion and market hours filtering
- **Timezone Complexity**: Ensure proper ET conversion across different data sources and daylight saving time transitions

### Business Risks
- **Overfitting**: Use out-of-sample testing and rolling analysis
- **Market Regime Changes**: Include market condition analysis
- **Survivorship Bias**: Include delisted stocks and failed alerts

## Timeline and Milestones

### Phase 1: Core Atoms Development (2 weeks)
- **Atoms**: Develop reusable analysis, metrics, and simulation atoms
- **Molecules**: Create basic AlertAnalyzer molecule
- **Integration**: Data loading, alignment, and trading hours filtering
- **Testing**: Unit tests for all atoms

### Phase 2: Advanced Analytics Atoms (2 weeks)
- **Atoms**: Advanced risk metrics and statistical analysis functions
- **Molecules**: Enhanced performance reporting and visualization
- **Reusability**: Ensure atoms can be used in other analysis scenarios
- **Testing**: Integration tests for molecule combinations

### Phase 3: Reporting and Monitoring (1 week)
- **Molecules**: Complete report generation and export functionality
- **CLI**: Command line interface for easy usage
- **Reusability**: Export atoms that can be used in other reporting contexts

### Phase 4: Optimization and Deployment (1 week)
- **Performance**: Optimize atom and molecule performance
- **Documentation**: Document all atoms for reusability
- **Testing**: Comprehensive testing suite
- **Deployment**: Production-ready alert analyzer

## Future Enhancements

### Advanced Atoms for Reusability
- **`atoms/ml/`** - Machine learning atoms for alert optimization
- **`atoms/portfolio/`** - Real-time portfolio tracking atoms
- **`atoms/backtesting/`** - Strategy backtesting framework atoms
- **`atoms/regimes/`** - Market regime detection atoms

### Additional Reusable Analytics
- **Market Regime Atoms**: Detect different market conditions (trending, sideways, volatile)
- **Correlation Atoms**: Analyze relationships between alerts and market factors
- **Seasonality Atoms**: Identify seasonal performance patterns
- **Clustering Atoms**: Group similar alerts and identify patterns
- **Risk Management Atoms**: Advanced risk calculation functions

### Composable Molecules
- **`molecules/strategy_optimizer.py`** - Combines ML and backtesting atoms
- **`molecules/market_analyzer.py`** - Combines regime and correlation atoms
- **`molecules/risk_monitor.py`** - Combines risk management and portfolio atoms

### Architecture Benefits for Future Development
- **Rapid Prototyping**: New analyses can quickly combine existing atoms
- **Consistent Calculations**: Shared atoms ensure methodology consistency
- **Easy Testing**: Isolated atoms simplify debugging and validation
- **Knowledge Sharing**: Well-documented atoms enable team collaboration

---

## Appendix

### Sample Data Structures

#### Alert Data Schema
```json
{
  "symbol": "STKH",
  "timestamp": "2025-07-10T10:51:00",
  "current_price": 3.5,
  "breakout_type": "bullish_breakout",
  "priority": "MEDIUM",
  "confidence_score": 0.889,
  "recommended_stop_loss": 3.24,
  "recommended_take_profit": 3.64
}
```

#### Market Data Schema
```csv
timestamp,symbol,open,high,low,close,volume,vwap
2025-07-10T10:51:00,STKH,3.48,3.52,3.47,3.50,12500,3.495
```

#### Trading Hours Filter Implementation
```python
def filter_trading_hours(df, start_time="09:30", end_time="16:00", timezone="US/Eastern"):
    """
    Filter DataFrame to include only regular trading hours (9:30 AM - 4:00 PM ET)
    
    Args:
        df: DataFrame with timestamp column
        start_time: Trading day start time (default: "09:30")
        end_time: Trading day end time (default: "16:00") 
        timezone: Target timezone (default: "US/Eastern")
    
    Returns:
        Filtered DataFrame containing only regular trading hours data
    """
    # Convert timestamps to Eastern Time
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(timezone)
    
    # Filter to trading hours
    mask = (df['timestamp'].dt.time >= pd.to_datetime(start_time).time()) & \
           (df['timestamp'].dt.time <= pd.to_datetime(end_time).time())
    
    # Exclude weekends
    mask &= df['timestamp'].dt.dayofweek < 5
    
    return df[mask]

def filter_alert_hours(df, start_time="09:30", end_time="15:30", timezone="US/Eastern"):
    """
    Filter DataFrame to include only valid alert hours (9:30 AM - 3:30 PM ET)
    
    Args:
        df: DataFrame with timestamp column
        start_time: Alert hours start time (default: "09:30")
        end_time: Alert hours end time (default: "15:30") 
        timezone: Target timezone (default: "US/Eastern")
    
    Returns:
        Filtered DataFrame containing only valid alert hours data
    """
    # Convert timestamps to Eastern Time
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(timezone)
    
    # Filter to alert hours
    mask = (df['timestamp'].dt.time >= pd.to_datetime(start_time).time()) & \
           (df['timestamp'].dt.time <= pd.to_datetime(end_time).time())
    
    # Exclude weekends
    mask &= df['timestamp'].dt.dayofweek < 5
    
    return df[mask]
```

### Performance Calculation Examples

#### Trade Simulation Logic
```python
def simulate_trade(alert, market_data):
    entry_price = alert['current_price']
    stop_loss = alert['recommended_stop_loss']
    take_profit = alert['recommended_take_profit']
    
    for timestamp, price_data in market_data.items():
        if alert['breakout_type'] == 'bullish_breakout':
            if price_data['high'] >= take_profit:
                return calculate_return(entry_price, take_profit)
            elif price_data['low'] <= stop_loss:
                return calculate_return(entry_price, stop_loss)
        # Similar logic for bearish alerts
    
    # Alert expired
    return calculate_return(entry_price, price_data['close'])
```