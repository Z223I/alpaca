# Exit Strategy Optimization Specification

## Overview

This document outlines the statistical methodology for optimizing exit strategies for superduper alert-based trading positions. The goal is to maximize risk-adjusted profit while handling the inherent overlap between competing exit strategies.

## Problem Statement

### Two-Stage Optimization Problem

#### Stage 1: Alert Generation Parameters
1. **Timeframe**: 10, 15, 20, 25, 30 minutes (affects alert sensitivity)
2. **Green_Threshold**: 0.60, 0.65, 0.70, 0.75 (affects alert selectivity)

#### Stage 2: Exit Strategy Parameters  
1. **Stop Loss**: Fixed percentage below entry (existing baseline)
2. **Take Profit**: Fixed percentage above entry
3. **Trailing Stop**: Dynamic stop that trails price by fixed percentage
4. **MACD Exit**: Technical indicator-based exit signal
5. **End-of-Day**: Automatic liquidation at 15:40 ET (not optimized)

### Key Challenge
This is a **nested optimization problem**: Alert parameters determine which opportunities you see, then exit parameters determine how you capitalize on them. The optimal exit strategy may depend on the alert sensitivity (e.g., more selective alerts might justify wider stops).

## Methodology

### 1. Data Structure Requirements

#### Alert Data Location Pattern
Historical alert data is stored in organized directory structure:
```
runs/YYYY-MM-DD/[SYMBOL]/run_YYYY-MM-DD_tf[TIMEFRAME]_th[THRESHOLD]_[HASH]/
```

**Examples:**
- `runs/2025-07-29/STAI/run_2025-07-29_tf10_th0.65_5c51065c/` → 10-minute timeframe, 0.65 threshold
- `runs/2025-08-04/BTAI/run_2025-08-04_tf30_th0.7_f7bcf731/` → 30-minute timeframe, 0.70 threshold

**Alert Parameter Extraction:**
```python
def parse_run_directory(run_path):
    """
    Extract timeframe and threshold from directory name
    Example: 'run_2025-07-29_tf15_th0.65_5c51065c' → tf=15, th=0.65
    """
    import re
    pattern = r'run_\d{4}-\d{2}-\d{2}_tf(\d+)_th([\d\.]+)_[a-f0-9]+'
    match = re.search(pattern, run_path)
    if match:
        return {
            'timeframe_minutes': int(match.group(1)),
            'green_threshold': float(match.group(2))
        }
    return None
```

#### Existing Data Loading Methods

**1. ORBPipelineSimulator._load_historical_data() (code/orb_pipeline_simulator.py:190)**
```python
def _load_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
    """Load historical data for a symbol - tries CSV first, then API"""
    csv_data = self._load_from_csv(symbol)  # Loads from historical_data/{date}/market_data/{symbol}_*.csv
    if csv_data is not None:
        return csv_data
    return self._fetch_from_api(symbol)  # Fallback to API
```

**2. load_candlestick_data() (plot_comprehensive_ema_divergence.py:18)**
```python
def load_candlestick_data(symbol='VWAV', date='20250728'):
    """Load minute-by-minute candlestick data from CSV files"""
    csv_files = glob.glob(f'historical_data/2025-07-28/market_data/{symbol}_*.csv')
    # Combines multiple CSV files, handles timezone conversion to Eastern Time
    return combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
```

**3. AlertsBacktest.load_historical_market_data() (tests/backtesting/alerts_backtest.py:110)**
```python
def load_historical_market_data(self) -> Optional[pd.DataFrame]:
    """Load historical market data for backtesting"""
    pattern = f"{self.symbol}_*.csv"
    data_files = list(self.market_data_dir.glob(pattern))
    # Combines all files for symbol, handles errors gracefully
```

#### Recommended Data Loading Implementation with Caching

**Create cached data loader for exit strategy optimization:**
```python
class HistoricalDataLoader:
    def __init__(self):
        self._cache = {}  # Cache dataframes for reuse
        
    def get_symbol_data(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """
        Load and cache historical data for a specific symbol and date.
        Uses existing codebase patterns with caching for optimization testing.
        """
        cache_key = f"{symbol}_{date}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Use existing pattern from plot_comprehensive_ema_divergence.py
        formatted_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
        csv_files = glob.glob(f'historical_data/{formatted_date}/market_data/{symbol}_*.csv')
        
        if not csv_files:
            return None
            
        # Combine all CSV files for the symbol (existing logic)
        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        # Handle timezone (Eastern Time for trading)
        if combined_df['timestamp'].dt.tz is None:
            combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('US/Eastern')
        
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        
        # Cache for reuse during optimization
        self._cache[cache_key] = combined_df
        return combined_df
        
    def clear_cache(self):
        """Clear cache to free memory if needed"""
        self._cache.clear()
```

#### Collecting Sent Superduper Alerts from ./runs Directory

**Alert Data Collection Strategy:**
```python
def collect_superduper_alerts_from_runs() -> List[Dict]:
    """
    Collect all sent superduper alerts from existing runs directory structure.
    Returns list of alerts with extracted parameters and entry data.
    """
    
    alerts = []
    runs_dir = Path('./runs')
    
    # Iterate through date directories: 2025-07-29, 2025-08-04, etc.
    for date_dir in runs_dir.glob('20*'):
        if not date_dir.is_dir():
            continue
            
        date_str = date_dir.name  # e.g., '2025-07-29'
        
        # Iterate through symbol directories: STAI, BTAI, etc.  
        for symbol_dir in date_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
                
            symbol = symbol_dir.name  # e.g., 'STAI'
            
            # Iterate through run directories with parameters
            for run_dir in symbol_dir.glob('run_*_tf*_th*_*'):
                # Extract parameters from directory name
                params = parse_run_directory(run_dir.name)
                if not params:
                    continue
                
                # Look for alert files in logs directory
                alert_files = find_alert_files(run_dir)
                
                for alert_file in alert_files:
                    # Parse alerts from log files
                    file_alerts = parse_superduper_alerts(alert_file)
                    
                    for alert in file_alerts:
                        alert.update({
                            'symbol': symbol,
                            'date': date_str,
                            'timeframe_minutes': params['timeframe_minutes'],
                            'green_threshold': params['green_threshold'],
                            'run_directory': str(run_dir)
                        })
                        alerts.append(alert)
    
    return alerts

def find_alert_files(run_dir: Path) -> List[Path]:
    """Find alert files in run directory logs."""
    alert_files = []
    
    # Check multiple possible locations
    locations = [
        run_dir / 'logs' / 'orb_superduper',      # orb_superduper_*.log
        run_dir / 'logs' / 'orb_trades',          # orb_trades_*.log  
        run_dir / 'logs',                         # Direct in logs
    ]
    
    for location in locations:
        if location.exists():
            # Look for superduper alert files
            alert_files.extend(location.glob('*superduper*.log'))
            alert_files.extend(location.glob('*alert*.log'))
            alert_files.extend(location.glob('*trades*.log'))
    
    return alert_files

def parse_superduper_alerts(log_file: Path) -> List[Dict]:
    """
    Parse superduper alerts from log files.
    Extract entry timestamp, price, and alert details.
    """
    
    alerts = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for alert patterns in logs
                if 'superduper' in line.lower() or 'alert' in line.lower():
                    alert = parse_alert_line(line)
                    if alert:
                        alerts.append(alert)
                        
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
    
    return alerts

def parse_alert_line(line: str) -> Optional[Dict]:
    """
    Parse individual alert line from log file.
    Extract timestamp, symbol, price, and other alert data.
    """
    
    # This will need to be customized based on actual log format
    # Example patterns to look for:
    # - Timestamp patterns
    # - Price data  
    # - Alert trigger information
    # - Entry conditions met
    
    # Placeholder - implement based on actual log format
    try:
        # Extract timestamp, price, alert details from log line
        # Return structured alert data
        return {
            'alert_timestamp': None,  # Parse from log
            'entry_price': None,      # Parse from log
            'entry_time': None,       # Parse from log
            'alert_details': line.strip()
        }
    except:
        return None
```

**Data Validation and Quality Checks:**
```python
def validate_collected_alerts(alerts: List[Dict]) -> Dict[str, Any]:
    """
    Validate collected alerts and report data quality issues.
    """
    
    validation_report = {
        'total_alerts': len(alerts),
        'unique_symbols': len(set(alert['symbol'] for alert in alerts)),
        'date_range': None,
        'parameter_coverage': {},
        'missing_data_issues': [],
        'data_quality_score': 0.0
    }
    
    if alerts:
        dates = [alert['date'] for alert in alerts]
        validation_report['date_range'] = (min(dates), max(dates))
        
        # Check parameter space coverage
        param_combinations = set()
        for alert in alerts:
            param_key = (alert['timeframe_minutes'], alert['green_threshold'])
            param_combinations.add(param_key)
        
        validation_report['parameter_coverage'] = {
            'combinations_found': len(param_combinations),
            'expected_combinations': 4,   # 1 timeframe × 4 thresholds (FOCUSED)
            'coverage_percentage': len(param_combinations) / 4 * 100
        }
        
        # Check for missing data
        missing_entry_prices = sum(1 for alert in alerts if not alert.get('entry_price'))
        missing_timestamps = sum(1 for alert in alerts if not alert.get('alert_timestamp'))
        
        if missing_entry_prices > 0:
            validation_report['missing_data_issues'].append(f"{missing_entry_prices} alerts missing entry prices")
        if missing_timestamps > 0:
            validation_report['missing_data_issues'].append(f"{missing_timestamps} alerts missing timestamps")
    
    return validation_report
```

#### Required Data Structure per Alert
```python
# Required data per historical alert/position (collected from runs)
{
    'alert_timestamp': datetime,        # Parsed from log files
    'entry_price': float,              # Parsed from log files  
    'entry_time': datetime,            # Parsed from log files
    'symbol': str,                     # From directory structure
    'date': str,                       # From directory structure (YYYY-MM-DD)
    'timeframe_minutes': int,          # Extracted from run directory name
    'green_threshold': float,          # Extracted from run directory name
    'run_directory': str,              # Source directory path
    'alert_details': str,              # Raw log line for debugging
    'price_series': pd.DataFrame,      # From HistoricalDataLoader.get_symbol_data()
    'macd_series': [(timestamp, macd, signal, histogram), ...],  # Calculated
    'market_conditions': {
        'volatility': float,
        'trend': str,  # 'bullish', 'bearish', 'sideways' 
        'sector': str,
        'market_cap': str
    }
}
```

### 2. Parameter Space Definition

```python
# Two-stage parameter space (FOCUSED TESTING)
ALERT_PARAMETERS = {
    'timeframe_minutes': [20],                          # 1 value (focused testing)
    'green_threshold': [0.60, 0.65, 0.70, 0.75]        # 4 values
}

EXIT_PARAMETERS = {
    'take_profit_pct': [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],  # 10 values
    'trailing_stop_pct': [5, 7.5, 10, 12.5, 15],            # 5 values
    'stop_loss_pct': [5, 7.5, 10, 12.5, 15, 17.5, 20],     # 7 values
    'macd_sensitivity': ['conservative', 'normal', 'aggressive'],  # 3 values
    'macd_enabled': [True, False]                            # 2 values
}

# Total combinations (FOCUSED): 
# Alert combinations: 1 × 4 = 4
# Exit combinations per alert set: 10 × 5 × 7 × 3 × 2 = 2,100
# Total parameter sets: 4 × 2,100 = 8,400 (80% reduction!)
```

### 3. Two-Stage Simulation Framework

#### Stage 1: Alert Generation Simulation
```python
def generate_alerts(historical_data, alert_params):
    """
    Generate alerts based on timeframe and green_threshold parameters.
    Returns: list of alert opportunities with entry points
    """
    
    alerts = []
    timeframe = alert_params['timeframe_minutes']
    threshold = alert_params['green_threshold']
    
    # Apply superduper alert logic with these parameters
    for timestamp, market_data in historical_data.items():
        if superduper_criteria_met(market_data, timeframe, threshold):
            alerts.append({
                'timestamp': timestamp,
                'symbol': market_data['symbol'],
                'entry_price': market_data['price'],
                'price_series': get_future_price_data(timestamp),
                'macd_series': get_future_macd_data(timestamp)
            })
    
    return alerts
```

#### Stage 2: Exit Strategy Simulation  
```python
def simulate_exit(alert, exit_params):
    """
    Simulate which exit strategy would trigger first for a given alert.
    Returns: (exit_time, exit_price, exit_reason, profit_pct)
    """
    
    entry_price = alert['entry_price']
    
    for timestamp, price in alert['price_series']:
        # Check exits in order of typical trigger speed
        
        # 1. Stop Loss (fastest, protects capital)
        if price <= entry_price * (1 - exit_params['stop_loss_pct']/100):
            return timestamp, price, 'stop_loss', calculate_profit(price, entry_price)
            
        # 2. Take Profit (if set and triggered)
        if price >= entry_price * (1 + exit_params['take_profit_pct']/100):
            return timestamp, price, 'take_profit', calculate_profit(price, entry_price)
            
        # 3. Trailing Stop (dynamic, can trigger on pullbacks)
        if trailing_stop_triggered(price, highest_price_so_far, exit_params):
            return timestamp, price, 'trailing_stop', calculate_profit(price, entry_price)
            
        # 4. MACD Signal (if enabled and triggered)
        if exit_params['macd_enabled'] and macd_exit_signal(alert['macd_series'], timestamp, exit_params):
            return timestamp, price, 'macd_exit', calculate_profit(price, entry_price)
    
    # 5. End of day liquidation (15:40 ET)
    return eod_time, eod_price, 'eod_liquidation', calculate_profit(eod_price, entry_price)

#### Combined Strategy Testing
def test_complete_strategy(historical_data, alert_params, exit_params):
    """
    Test complete strategy: generate alerts, then test exits on each alert
    """
    
    # Stage 1: Generate alerts with these parameters
    alerts = generate_alerts(historical_data, alert_params)
    
    # Stage 2: Test exit strategy on each alert
    results = []
    for alert in alerts:
        exit_result = simulate_exit(alert, exit_params)
        results.append(exit_result)
    
    return calculate_strategy_metrics(results)
```

### 4. Optimization Objectives

#### Primary Metrics
- **Risk-Adjusted Return**: Sharpe ratio or Sortino ratio
- **Total Return**: Sum of all position profits/losses
- **Win Rate**: Percentage of profitable positions
- **Average Win/Loss Ratio**: Mean profit of winners / Mean loss of losers

#### Secondary Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss
- **Trade Frequency**: Number of positions per time period
- **Market Condition Performance**: Returns segmented by volatility/trend

### 5. Statistical Testing Framework

#### A. Grid Search with Cross-Validation
```python
# Split historical data chronologically
train_data = alerts[:'2024-06-01']
validation_data = alerts['2024-06-01':'2024-09-01'] 
test_data = alerts['2024-09-01':]

# Optimize on training data
best_params = grid_search_optimization(train_data, PARAMETER_SPACE)

# Validate on unseen data
validation_results = backtest(validation_data, best_params)

# Final test on holdout set
final_results = backtest(test_data, best_params)
```

#### B. Walk-Forward Analysis
```python
# Use expanding window to avoid look-ahead bias
for start_date in monthly_intervals:
    train_end = start_date + training_period
    test_start = train_end + timedelta(days=1)
    test_end = test_start + testing_period
    
    # Optimize parameters on training window
    optimal_params = optimize(alerts[start_date:train_end])
    
    # Test on out-of-sample period
    results = backtest(alerts[test_start:test_end], optimal_params)
    
    walk_forward_results.append(results)
```

#### C. Statistical Significance Testing

```python
# Bootstrap resampling for confidence intervals
def bootstrap_performance(results, n_iterations=1000):
    """
    Generate confidence intervals for performance metrics
    """
    bootstrap_returns = []
    for _ in range(n_iterations):
        sample = np.random.choice(results['returns'], 
                                 size=len(results['returns']), 
                                 replace=True)
        bootstrap_returns.append(np.mean(sample))
    
    return {
        'mean': np.mean(bootstrap_returns),
        'ci_lower': np.percentile(bootstrap_returns, 2.5),
        'ci_upper': np.percentile(bootstrap_returns, 97.5)
    }

# Paired t-test for strategy comparison
from scipy.stats import ttest_rel

def compare_strategies(baseline_returns, optimized_returns):
    """
    Test if optimized strategy significantly outperforms baseline
    """
    statistic, p_value = ttest_rel(optimized_returns, baseline_returns)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_improvement': np.mean(optimized_returns) - np.mean(baseline_returns)
    }
```

### 6. Handling Strategy Overlap

#### Exit Strategy Interaction Analysis
```python
def analyze_strategy_interactions(historical_data, parameter_sets):
    """
    Analyze how often each exit strategy would have triggered
    under different parameter combinations
    """
    
    interaction_matrix = {}
    
    for params in parameter_sets:
        exit_counts = {
            'stop_loss': 0,
            'take_profit': 0, 
            'trailing_stop': 0,
            'macd_exit': 0,
            'eod_liquidation': 0
        }
        
        for position in historical_data:
            exit_reason = simulate_exit(position, params)[2]
            exit_counts[exit_reason] += 1
            
        interaction_matrix[str(params)] = exit_counts
    
    return interaction_matrix
```

#### Marginal Contribution Analysis
```python
def marginal_strategy_value(base_params, strategy_to_test):
    """
    Calculate the marginal value of adding a specific exit strategy
    """
    
    # Test with strategy disabled
    params_without = base_params.copy()
    params_without[f"{strategy_to_test}_enabled"] = False
    results_without = backtest(historical_data, params_without)
    
    # Test with strategy enabled
    params_with = base_params.copy()
    params_with[f"{strategy_to_test}_enabled"] = True
    results_with = backtest(historical_data, params_with)
    
    return {
        'marginal_return': results_with['total_return'] - results_without['total_return'],
        'marginal_sharpe': results_with['sharpe'] - results_without['sharpe'],
        'activation_rate': results_with['strategy_usage'][strategy_to_test]
    }
```

### 7. Market Regime Analysis

#### Segmented Performance Testing
```python
def regime_based_optimization(historical_data):
    """
    Optimize parameters separately for different market conditions
    """
    
    regimes = {
        'high_volatility': filter_by_volatility(historical_data, threshold=0.02),
        'low_volatility': filter_by_volatility(historical_data, threshold=0.01),
        'bullish_trend': filter_by_trend(historical_data, 'bullish'),
        'bearish_trend': filter_by_trend(historical_data, 'bearish'),
        'sideways_trend': filter_by_trend(historical_data, 'sideways')
    }
    
    regime_optimal_params = {}
    
    for regime_name, regime_data in regimes.items():
        if len(regime_data) > 30:  # Minimum sample size
            optimal_params = optimize_parameters(regime_data)
            regime_optimal_params[regime_name] = optimal_params
    
    return regime_optimal_params
```

### 8. Implementation Recommendations

#### Phase 1: Data Preparation (Week 1)
1. **Collect existing sent superduper alerts from ./runs directory**
2. Enrich with minute-by-minute price and MACD data for exit simulation  
3. Build two-stage simulation infrastructure
4. Validate data completeness and quality

#### Phase 2: Computational Strategy (Week 2)
**Challenge**: 42,000 parameter combinations (20 alert × 2,100 exit) requires optimization strategy

**Option A - Hierarchical Optimization** (RECOMMENDED):
```python
# Step 1: Find optimal alert parameters using simple exit (fixed stops)
simple_exit = {'stop_loss': 10, 'take_profit': 8, 'trailing': None, 'macd': False}
best_alert_params = test_all_alert_combinations(simple_exit)

# Step 2: Optimize exit parameters for best alert settings
best_exit_params = test_all_exit_combinations(best_alert_params)
```

**Option B - Stratified Sampling**:
- Test representative sample of 1,000 combinations
- Focus on extreme and middle values
- Use results to guide full search in promising regions

#### Phase 3: Joint Optimization Analysis (Week 3)
1. **Alert Quality Analysis**: Compare alert performance by parameters
   - High threshold/long timeframe → Fewer, higher quality alerts
   - Low threshold/short timeframe → More, lower quality alerts
2. **Exit Strategy Matching**: Test if alert quality affects optimal exit strategy
3. **Cross-validation**: Ensure temporal stability of joint parameters

#### Phase 4: Validation & Implementation (Week 4-5)
1. Out-of-sample testing on holdout data  
2. Walk-forward analysis for joint parameter stability
3. Statistical significance testing vs current strategy
4. Final recommendation with confidence intervals

### 9. Risk Management Considerations

#### Overfitting Prevention
- Use at least 3:1 training-to-testing data ratio
- Limit parameter complexity to avoid curve fitting
- Require statistical significance (p < 0.05) for changes
- Test stability across different time periods

#### Transaction Cost Integration
```python
def calculate_net_profit(gross_profit, position_size, symbol):
    """
    Calculate net profit - no transaction costs assumed
    """
    return gross_profit  # Zero transaction costs
```

#### Risk Limit Integration
- Maximum position size limits
- Daily loss limits
- Concentration limits by sector/symbol
- Volatility-adjusted position sizing

### 10. Success Metrics

#### Minimum Performance Thresholds
- **Statistical Significance**: p < 0.05 vs current strategy
- **Sharpe Ratio Improvement**: > 0.2 increase
- **Maximum Drawdown**: < 15% in worst case
- **Win Rate**: > 55% for viability
- **Profit Factor**: > 1.3 for sustainability

#### Deliverables
1. **Optimized Parameter Set**: Best parameters for each exit strategy
2. **Performance Report**: Detailed backtest results with confidence intervals
3. **Market Regime Rules**: When to use different parameter sets
4. **Risk Management Updates**: Revised position sizing and limits
5. **Implementation Guide**: Code changes needed for live trading

### 10. Critical Implementation Considerations

#### Statistical Multiple Testing Correction
```python
# With 42,000 parameter combinations, need correction for multiple testing
from statsmodels.stats.multitest import multipletests

def apply_multiple_testing_correction(p_values, method='holm'):
    """
    Apply multiple testing correction to avoid false discoveries.
    42,000 tests require strict correction to maintain statistical validity.
    """
    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, 
        alpha=0.05, 
        method=method  # 'holm', 'bonferroni', 'fdr_bh'
    )
    return rejected, p_corrected
```

#### Market Microstructure Considerations
```python
# Handle real trading constraints
MARKET_CONSTRAINTS = {
    'trading_hours': ('09:30', '16:00'),  # Eastern Time
    'min_price_increment': 0.01,         # Penny increments
    'market_holidays': [],               # Load from calendar
    'halt_handling': True,               # Handle trading halts
    'gap_handling': 'next_available',    # Price gaps over stop/target
    'execution_delay': 1,                # 1 minute execution delay
}

def adjust_for_market_reality(theoretical_exit_price, market_data, timestamp):
    """
    Adjust theoretical exit prices for real market constraints.
    """
    # Check if market is open
    if not is_market_open(timestamp):
        return None, 'market_closed'
    
    # Handle price gaps (stock gaps over stop loss)
    if theoretical_exit_price < market_data['low'] or theoretical_exit_price > market_data['high']:
        actual_price = market_data['open']  # Gap execution at next open
        return actual_price, 'gap_execution'
    
    # Apply bid-ask spread impact (even with zero commissions)
    spread_impact = estimate_spread_impact(market_data['volume'])
    adjusted_price = theoretical_exit_price - spread_impact
    
    return adjusted_price, 'normal_execution'
```

#### Sample Size and Statistical Power Analysis
```python
def calculate_required_sample_size(effect_size=0.2, power=0.8, alpha=0.05):
    """
    Calculate minimum number of alerts needed for statistical significance.
    Conservative parameter sets may not generate enough alerts for valid testing.
    """
    # Rule of thumb: need 30+ alerts per parameter combination for reliable results
    # With 42,000 combinations, need robust methodology for small sample handling
    
    min_alerts_per_combination = 30
    low_volume_combinations = []
    
    # Flag parameter combinations with insufficient data
    for params in parameter_combinations:
        alert_count = count_alerts_for_params(params)
        if alert_count < min_alerts_per_combination:
            low_volume_combinations.append((params, alert_count))
    
    return low_volume_combinations
```

#### Position Sizing and Portfolio Risk Integration
```python
class PortfolioRiskManager:
    """
    Account for portfolio-level constraints in exit strategy optimization.
    """
    
    def __init__(self, max_portfolio_risk=0.02, max_position_correlation=0.7):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_correlation = max_position_correlation
        
    def calculate_position_size(self, entry_price, stop_loss_pct, account_value):
        """
        Calculate position size based on portfolio risk, not just individual position risk.
        """
        # Risk per share = entry_price * (stop_loss_pct / 100)
        risk_per_share = entry_price * (stop_loss_pct / 100)
        
        # Maximum position risk = account_value * max_portfolio_risk
        max_position_risk = account_value * self.max_portfolio_risk
        
        # Position size = max_position_risk / risk_per_share
        position_size = max_position_risk / risk_per_share
        
        return position_size
        
    def check_correlation_limits(self, new_position, existing_positions):
        """
        Ensure new position doesn't violate correlation limits.
        """
        for existing in existing_positions:
            correlation = calculate_symbol_correlation(new_position['symbol'], existing['symbol'])
            if correlation > self.max_position_correlation:
                return False, f"High correlation ({correlation:.2f}) with {existing['symbol']}"
        return True, "OK"
```

#### Backtesting Bias Prevention
```python
class BiasPreventionFramework:
    """
    Prevent common backtesting biases that invalidate results.
    """
    
    @staticmethod
    def prevent_lookahead_bias(alert_timestamp, market_data):
        """
        Ensure no future data is used in decision making.
        """
        # Only use market data up to alert_timestamp
        valid_data = market_data[market_data['timestamp'] <= alert_timestamp]
        return valid_data
    
    @staticmethod
    def handle_survivorship_bias(symbol_list, date_range):
        """
        Include delisted/merged companies in analysis if they had alerts.
        """
        # Don't exclude symbols that were delisted during test period
        all_symbols = get_all_historical_symbols(date_range)
        return all_symbols
    
    @staticmethod
    def cross_validation_temporal_split(alerts, n_folds=5):
        """
        Ensure temporal ordering in cross-validation splits.
        """
        # Sort by date, split chronologically (not randomly)
        sorted_alerts = sorted(alerts, key=lambda x: x['alert_timestamp'])
        fold_size = len(sorted_alerts) // n_folds
        
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(sorted_alerts)
            folds.append(sorted_alerts[start_idx:end_idx])
        
        return folds
```

#### Alert Quality Metrics
```python
def calculate_alert_quality_metrics(alerts, market_data):
    """
    Assess quality of alerts beyond just count.
    High-quality alerts should justify different exit strategies.
    """
    
    quality_metrics = {}
    
    for alert in alerts:
        symbol = alert['symbol']
        entry_time = alert['alert_timestamp']
        entry_price = alert['entry_price']
        
        # Look at price movement in first 5, 15, 30 minutes after alert
        future_prices = get_future_prices(market_data, entry_time, minutes=[5, 15, 30])
        
        # Calculate immediate momentum (quality indicator)
        momentum_5min = (future_prices[5] - entry_price) / entry_price
        momentum_15min = (future_prices[15] - entry_price) / entry_price
        momentum_30min = (future_prices[30] - entry_price) / entry_price
        
        # Volume surge (another quality indicator)
        avg_volume = get_average_volume(market_data, symbol, lookback_minutes=60)
        entry_volume = get_volume_at_time(market_data, entry_time)
        volume_surge = entry_volume / avg_volume if avg_volume > 0 else 1
        
        alert_quality = {
            'momentum_5min': momentum_5min,
            'momentum_15min': momentum_15min, 
            'momentum_30min': momentum_30min,
            'volume_surge': volume_surge,
            'quality_score': calculate_composite_quality_score(momentum_5min, volume_surge)
        }
        
        quality_metrics[alert['symbol']] = alert_quality
    
    return quality_metrics
```

#### Performance Attribution Analysis
```python
def performance_attribution_analysis(results):
    """
    Decompose performance to understand key drivers.
    Critical for knowing whether success comes from alert quality or exit strategy.
    """
    
    attribution = {
        'alert_selection_contribution': 0,
        'exit_strategy_contribution': 0,
        'market_timing_contribution': 0,
        'symbol_selection_contribution': 0
    }
    
    # Compare performance vs random entry timing
    random_entry_performance = simulate_random_entries(results)
    attribution['alert_selection_contribution'] = results['total_return'] - random_entry_performance
    
    # Compare vs simple fixed exit (e.g., 5% stop, 10% profit)
    simple_exit_performance = simulate_simple_exits(results)
    attribution['exit_strategy_contribution'] = results['total_return'] - simple_exit_performance
    
    # Compare vs market benchmark
    market_return = calculate_market_return(results['date_range'])
    attribution['market_timing_contribution'] = results['total_return'] - market_return
    
    return attribution
```

#### Orin Nano Optimization Strategy

**Jetson Orin Nano Hardware Specifications:**
- **CPU**: 6-core ARM Cortex-A78AE @ 1.5GHz
- **GPU**: 1024-core NVIDIA Ampere GPU @ 625MHz
- **Memory**: 8GB LPDDR5 (shared CPU/GPU)
- **AI Performance**: 40 TOPS (INT8)
- **Power**: 7-15W configurable

**Optimization Strategy for Exit Strategy Testing:**

```python
import cupy as cp  # GPU-accelerated NumPy replacement
import cudf      # GPU-accelerated pandas replacement  
import concurrent.futures
import multiprocessing as mp
from numba import jit, cuda
import numpy as np

# Orin Nano optimized configuration
ORIN_NANO_CONFIG = {
    'cpu_cores': 6,
    'gpu_memory_limit': '6GB',        # Reserve 2GB for system
    'cpu_workers': 4,                 # Leave 2 cores for system/GPU management
    'batch_size': 100,                # Process multiple parameter sets together
    'use_gpu_acceleration': True,
    'memory_management': 'aggressive_caching',
    'checkpoint_frequency': 'every_50_tests',  # More frequent due to power constraints
}

class OrinNanoOptimizer:
    """
    Leverage Orin Nano's ARM CPU + GPU architecture for parameter optimization.
    """
    
    def __init__(self):
        self.gpu_available = self._check_gpu()
        self.cpu_cores = mp.cpu_count()  # Should be 6 on Orin Nano
        
        # Initialize GPU memory pool if available
        if self.gpu_available:
            import cupy
            cupy.cuda.MemoryPool().set_limit(size=6*1024**3)  # 6GB limit
            
    def _check_gpu(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            return True
        except:
            return False
    
    def optimize_parameters_gpu_accelerated(self, alerts_data, parameter_batches):
        """
        GPU-accelerated parameter optimization using CuPy/CuDF.
        Ideal for vectorized calculations across many parameter combinations.
        """
        
        if not self.gpu_available:
            return self.optimize_parameters_cpu_only(alerts_data, parameter_batches)
            
        # Move data to GPU
        gpu_price_data = cp.asarray(alerts_data['prices'])
        gpu_timestamps = cp.asarray(alerts_data['timestamps'])
        
        results = []
        
        for batch in parameter_batches:
            # Vectorize exit strategy calculations across entire batch
            batch_results = self._gpu_batch_exit_simulation(
                gpu_price_data, 
                gpu_timestamps,
                batch
            )
            results.extend(batch_results)
            
            # Periodically clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            
        return results
    
    @cuda.jit
    def _gpu_exit_simulation_kernel(price_data, exit_params, results):
        """
        CUDA kernel for parallel exit strategy simulation.
        Each GPU thread processes one parameter combination.
        """
        idx = cuda.grid(1)
        if idx < exit_params.shape[0]:
            # Simulate exit strategy for this parameter set
            # Vectorized operations run in parallel across GPU cores
            stop_loss = exit_params[idx, 0]
            take_profit = exit_params[idx, 1]
            trailing_stop = exit_params[idx, 2]
            
            # Calculate exit points for this parameter set
            # Store results in results array
            pass  # Implementation depends on specific exit logic
    
    def optimize_parameters_cpu_parallel(self, alerts_data, parameter_combinations):
        """
        CPU-optimized parallel processing for ARM Cortex-A78AE.
        Uses multiprocessing with ARM-specific optimizations.
        """
        
        # ARM-specific optimizations
        chunk_size = len(parameter_combinations) // (self.cpu_cores - 2)  # Reserve cores
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Split work across available ARM cores
            futures = []
            
            for i in range(0, len(parameter_combinations), chunk_size):
                chunk = parameter_combinations[i:i+chunk_size]
                future = executor.submit(
                    self._process_parameter_chunk_optimized, 
                    alerts_data, 
                    chunk
                )
                futures.append(future)
            
            # Collect results as they complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    # Update progress
                    print(f"Completed chunk: {len(results)} total results")
                    
                except Exception as e:
                    print(f"Chunk processing error: {e}")
        
        return results
    
    @jit(nopython=True)  # JIT compilation for ARM optimization
    def _optimized_exit_calculation(self, price_series, entry_price, exit_params):
        """
        Numba JIT-compiled exit calculation optimized for ARM architecture.
        """
        
        stop_loss_pct, take_profit_pct, trailing_stop_pct = exit_params
        
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        take_profit_price = entry_price * (1 + take_profit_pct / 100)
        
        highest_price = entry_price
        
        for i in range(len(price_series)):
            current_price = price_series[i]
            
            # Update trailing stop
            if current_price > highest_price:
                highest_price = current_price
            
            trailing_stop_price = highest_price * (1 - trailing_stop_pct / 100)
            
            # Check exit conditions (order matters)
            if current_price <= stop_loss_price:
                return current_price, i, 'stop_loss'
            elif current_price >= take_profit_price:
                return current_price, i, 'take_profit'
            elif current_price <= trailing_stop_price:
                return current_price, i, 'trailing_stop'
        
        # No exit triggered
        return price_series[-1], len(price_series)-1, 'eod_liquidation'

class OrinNanoMemoryManager:
    """
    Optimized memory management for 8GB shared CPU/GPU memory.
    """
    
    def __init__(self, max_memory_gb=6):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.cache = {}
        self.memory_usage = 0
    
    def get_cached_data(self, symbol, date):
        """Get cached data with memory pressure awareness."""
        cache_key = f"{symbol}_{date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load data
        data = self._load_symbol_data(symbol, date)
        data_size = self._estimate_memory_size(data)
        
        # Check memory pressure
        if self.memory_usage + data_size > self.max_memory_bytes:
            self._evict_old_cache_entries(data_size)
        
        # Cache the data
        self.cache[cache_key] = data
        self.memory_usage += data_size
        
        return data
    
    def _evict_old_cache_entries(self, required_space):
        """Evict cached data using LRU strategy."""
        # Simple LRU implementation for memory management
        while self.memory_usage + required_space > self.max_memory_bytes:
            if not self.cache:
                break
                
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            data_size = self._estimate_memory_size(self.cache[oldest_key])
            del self.cache[oldest_key]
            self.memory_usage -= data_size

# Computational requirements adjusted for Orin Nano (FOCUSED TESTING)
ORIN_NANO_REQUIREMENTS = {
    'estimated_memory_per_test': '30MB',      # Reduced due to ARM efficiency
    'total_memory_estimate': '252GB',        # 8,400 × 30MB (80% reduction)
    'processing_time_per_test': '15 seconds', # ARM + GPU acceleration
    'total_processing_time': '35 hours',     # 8,400 × 15 seconds (80% reduction)
    'gpu_accelerated_time': '17.5 hours',   # With GPU vectorization
    'parallel_cpu_time': '8.75 hours',      # 4-core ARM parallel  
    'hybrid_approach_time': '5 hours',      # CPU + GPU hybrid (FAST!)
    'power_consumption': '10-15W',          # Very efficient
    'thermal_management': 'passive_cooling_sufficient'
}

class ThermalAwareScheduler:
    """
    Schedule intensive computations based on Orin Nano thermal limits.
    """
    
    def __init__(self, thermal_limit=70):  # Celsius
        self.thermal_limit = thermal_limit
        self.current_temp = self._get_cpu_temp()
        
    def _get_cpu_temp(self):
        """Read CPU temperature from thermal zone."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millicelsius = int(f.read().strip())
                return temp_millicelsius / 1000.0
        except:
            return 50  # Default safe temperature
            
    def should_throttle(self):
        """Check if should reduce processing due to thermal limits."""
        self.current_temp = self._get_cpu_temp()
        return self.current_temp > self.thermal_limit
        
    def get_optimal_batch_size(self):
        """Adjust batch size based on thermal state."""
        if self.should_throttle():
            return 25  # Smaller batches when hot
        else:
            return 100  # Normal batch size when cool

# Power-efficient optimization strategy
def orin_nano_optimization_strategy():
    """
    Recommended optimization approach for Jetson Orin Nano.
    """
    
    strategy = {
        'phase_1_data_prep': {
            'approach': 'cpu_parallel',
            'workers': 4,
            'estimated_time': '2 hours',
            'memory_usage': '4GB'
        },
        
        'phase_2_parameter_optimization': {
            'approach': 'hybrid_cpu_gpu',
            'cpu_workers': 3,
            'gpu_batch_size': 200,
            'estimated_time': '4 hours',    # FOCUSED TESTING
            'memory_usage': '6GB',
            'checkpoints': 'every_30_minutes'  # More frequent for shorter run
        },
        
        'phase_3_validation': {
            'approach': 'cpu_optimized',
            'workers': 2,
            'estimated_time': '3 hours',
            'memory_usage': '3GB'
        },
        
        'total_estimated_time': '5 hours',      # FOCUSED TESTING
        'total_power_consumption': '50-75 Wh',  # Very efficient  
        'cooling_requirements': 'passive_only'
    }
    
    return strategy
```

**Key Orin Nano Advantages for This Workload:**

1. **Energy Efficiency**: 25 hours at 10-15W vs 44 hours at 100W+ on desktop
2. **GPU Acceleration**: Vectorized exit calculations across parameter batches  
3. **ARM Optimization**: Numba JIT compilation optimized for ARM architecture
4. **Unified Memory**: Efficient data sharing between CPU and GPU
5. **Always-On Capability**: Can run continuously without thermal concerns
6. **Cost Effective**: $200 device vs expensive server hardware

**Implementation Priority:**
1. Start with CPU-parallel approach (immediate implementation)
2. Add GPU acceleration for vectorized calculations (phase 2)
3. Implement thermal-aware scheduling (for extended runs)
4. Add power management for 24/7 operation capability

### 11. Key Testing Insights

#### The Two-Stage Challenge
- **Alert parameters** determine which opportunities you see (17-59 alerts per parameter set)
- **Exit parameters** determine how you capitalize on each opportunity
- **Joint optimization** is required - optimal exits may depend on alert quality

#### Computational Complexity
- **42,000 total combinations** require strategic testing approach
- **Hierarchical optimization** recommended to manage computational load
- **Statistical significance** harder to achieve with fewer alerts from conservative settings

#### Critical Dependencies
- **Conservative alerts** (high threshold/long timeframe) may justify wider stops
- **Aggressive alerts** (low threshold/short timeframe) may require tighter risk management
- **Sample size** varies dramatically by alert parameters (affects statistical power)

## Conclusion

This methodology provides a robust framework for **joint optimization** of alert generation and exit strategies. Unlike simple exit strategy optimization, this requires:

1. **Two-stage simulation**: Generate alerts first, then test exits on each alert
2. **Hierarchical optimization**: Find optimal alert parameters, then optimal exits for those alerts  
3. **Sample size awareness**: Conservative alert settings generate fewer trading opportunities
4. **Joint validation**: Ensure both alert and exit parameters remain stable over time

The key insight is that **alert quality and exit strategy interact** - higher quality entry signals may justify different risk management approaches than lower quality signals. This joint optimization should yield superior risk-adjusted returns compared to optimizing exit strategies alone.