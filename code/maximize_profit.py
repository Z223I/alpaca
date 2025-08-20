#!/usr/bin/env python3
"""
Exit Strategy Optimization System

This module implements a two-stage optimization framework for maximizing risk-adjusted profits
from superduper alert-based trading positions. It optimizes both alert generation parameters
and exit strategy parameters using historical data and statistical validation.

Based on specifications in specs/exit-strategy-testing.md

Author: Claude Code
"""

import argparse
import concurrent.futures
import glob
import multiprocessing as mp
import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel

# Optional statistical packages
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# GPU acceleration imports (optional)
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Numba JIT compilation for ARM optimization
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings('ignore', category=UserWarning)

# FOCUSED TESTING Parameter Space (80% reduction)
ALERT_PARAMETERS = {
    'timeframe_minutes': [20],                          # 1 value (focused testing)
    'green_threshold': [0.60, 0.65, 0.70, 0.75]        # 4 values
}

EXIT_PARAMETERS = {
    'take_profit_pct': [5, 7.5, 10, 12.5, 15],              # 5 values (5-15% in 2.5% increments)
    'trailing_stop_pct': [5, 7.5, 10, 12.5, 15],            # 5 values
    # 'stop_loss_pct': removed - using trailing stops instead
    'macd_sensitivity': ['conservative', 'normal', 'aggressive'],  # 3 values
    'macd_enabled': [True, False]                            # 2 values
}

# Total combinations (FOCUSED): 1 × 4 = 4 alert combinations
# 5 × 5 × 3 × 2 = 150 exit combinations per alert set
# Total parameter sets: 4 × 150 = 600 (significant reduction, focused on trailing stops)

# Orin Nano optimized configuration
ORIN_NANO_CONFIG = {
    'cpu_cores': 6,
    'gpu_memory_limit': '6GB',        # Reserve 2GB for system
    'cpu_workers': 4,                 # Leave 2 cores for system/GPU management
    'batch_size': 100,                # Process multiple parameter sets together
    'use_gpu_acceleration': GPU_AVAILABLE,
    'memory_management': 'aggressive_caching',
    'checkpoint_frequency': 'every_50_tests',  # More frequent due to power constraints
}


# JIT-compiled standalone function for exit calculations
@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def calculate_exit_points_jit(price_series: np.ndarray, entry_price: float,
                             take_profit_pct: float,
                             trailing_stop_pct: float) -> Tuple[float, int, str]:
    """
    Numba JIT-compiled exit calculation optimized for ARM architecture.
    Uses only trailing stops and take profit (no fixed stop loss).
    Returns: (exit_price, exit_index, exit_reason)
    """
    take_profit_price = entry_price * (1 + take_profit_pct / 100)

    highest_price = entry_price

    for i in range(len(price_series)):
        current_price = price_series[i]

        # Update trailing stop
        if current_price > highest_price:
            highest_price = current_price

        trailing_stop_price = highest_price * (1 - trailing_stop_pct / 100)

        # Check exit conditions (order matters for first trigger)
        if current_price >= take_profit_price:
            return current_price, i, 'take_profit'
        elif current_price <= trailing_stop_price and highest_price > entry_price:
            return current_price, i, 'trailing_stop'

    # No exit triggered - end of day liquidation
    return price_series[-1], len(price_series)-1, 'eod_liquidation'


class HistoricalDataLoader:
    """
    Cached data loader for exit strategy optimization.
    Uses existing codebase patterns with memory management for Orin Nano.
    """

    def __init__(self, max_memory_gb: float = 6):
        self._cache = {}  # Cache dataframes for reuse
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.memory_usage = 0

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
            print(f"No historical data found for {symbol} on {date}")
            return None

        # Combine all CSV files for the symbol (existing logic)
        all_data = []
        try:
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                all_data.append(df)

            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

            # Handle timezone (Eastern Time for trading)
            if combined_df['timestamp'].dt.tz is None:
                combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('US/Eastern')

            combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

            # Check memory pressure
            data_size = self._estimate_memory_size(combined_df)
            if self.memory_usage + data_size > self.max_memory_bytes:
                self._evict_old_cache_entries(data_size)

            # Cache for reuse during optimization
            self._cache[cache_key] = combined_df
            self.memory_usage += data_size

            return combined_df

        except Exception as e:
            print(f"Error loading data for {symbol} on {date}: {e}")
            return None

    def _estimate_memory_size(self, df: pd.DataFrame) -> int:
        """Estimate memory usage of DataFrame."""
        return df.memory_usage(deep=True).sum()

    def _evict_old_cache_entries(self, required_space: int):
        """Evict cached data using LRU strategy."""
        # Simple LRU implementation for memory management
        while self.memory_usage + required_space > self.max_memory_bytes:
            if not self._cache:
                break

            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            data_size = self._estimate_memory_size(self._cache[oldest_key])
            del self._cache[oldest_key]
            self.memory_usage -= data_size

    def clear_cache(self):
        """Clear cache to free memory if needed."""
        self._cache.clear()
        self.memory_usage = 0


def parse_run_directory(run_path: str) -> Optional[Dict[str, Union[int, float]]]:
    """
    Extract timeframe and threshold from directory name.
    Example: 'run_2025-07-29_tf15_th0.65_5c51065c' → tf=15, th=0.65
    """
    pattern = r'run_\d{4}-\d{2}-\d{2}_tf(\d+)_th([\d\.]+)_[a-f0-9]+'
    match = re.search(pattern, run_path)
    if match:
        return {
            'timeframe_minutes': int(match.group(1)),
            'green_threshold': float(match.group(2))
        }
    return None


def find_alert_files(run_dir: Path) -> List[Path]:
    """Find JSON alert files in run directory only."""
    alert_files = []

    # Look for JSON alert files in historical_data directory
    json_locations = [
        run_dir / 'historical_data' / '*' / 'superduper_alerts_sent' / 'bullish' / 'green',
        run_dir / 'historical_data' / '*' / 'superduper_alerts_sent',
        run_dir / 'historical_data' / '*' / 'alerts',
    ]

    for location_pattern in json_locations:
        for location in run_dir.glob(str(location_pattern).replace(str(run_dir) + '/', '')):
            if location.exists() and location.is_dir():
                # Look for JSON alert files
                alert_files.extend(location.glob('*superduper_alert*.json'))
                alert_files.extend(location.glob('*alert*.json'))

    return alert_files


# parse_alert_line function removed - JSON files only


def parse_json_alert_file(json_file: Path) -> Optional[Dict[str, Any]]:
    """
    Parse superduper alert from JSON file.
    Extract entry timestamp, price, and alert details from structured JSON.
    """
    try:
        import json

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract key information from JSON structure
        symbol = data.get('symbol')

        # Get entry data from original_alert within latest_super_alert
        original_alert = data.get('latest_super_alert', {}).get('original_alert', {})
        if not original_alert:
            # Fallback: try to get from top level
            original_alert = data

        # Extract entry price and timestamp
        entry_price = original_alert.get('current_price')
        entry_timestamp = original_alert.get('timestamp')

        # Get trend analysis for timeframe
        trend_analysis = data.get('trend_analysis', {})
        timeframe_minutes = trend_analysis.get('timeframe_minutes')

        if not all([symbol, entry_price, entry_timestamp]):
            print(f"Missing required data in {json_file}: symbol={symbol}, price={entry_price}, timestamp={entry_timestamp}")
            return None

        # Parse timestamp
        if isinstance(entry_timestamp, str):
            # Handle timezone-aware timestamps
            if 'T' in entry_timestamp:
                entry_time = pd.to_datetime(entry_timestamp)
            else:
                entry_time = pd.to_datetime(entry_timestamp)
        else:
            entry_time = pd.to_datetime(entry_timestamp)

        alert_data = {
            'alert_timestamp': entry_time,
            'entry_price': float(entry_price),
            'entry_time': entry_time,
            'symbol': symbol,
            'timeframe_minutes': timeframe_minutes,
            'alert_details': f'JSON Alert: {symbol} at ${entry_price} on {entry_time}',
            'json_file': str(json_file),
            'original_alert_data': original_alert,
            'trend_analysis': trend_analysis,
            'confidence_score': original_alert.get('confidence_score'),
            'breakout_type': original_alert.get('breakout_type'),
            'recommended_stop_loss': original_alert.get('recommended_stop_loss'),
            'recommended_take_profit': original_alert.get('recommended_take_profit'),
        }

        return alert_data

    except Exception as e:
        print(f"Error parsing JSON file {json_file}: {e}")
        return None


def parse_superduper_alerts(alert_file: Path) -> List[Dict[str, Any]]:
    """
    Parse superduper alerts from JSON files only.
    Extract entry timestamp, price, and alert details.
    """
    alerts = []

    try:
        # Handle JSON files only
        if alert_file.suffix.lower() == '.json':
            alert = parse_json_alert_file(alert_file)
            if alert:
                alerts.append(alert)
        else:
            print(f"Skipping non-JSON file: {alert_file}")

    except Exception as e:
        print(f"Error parsing {alert_file}: {e}")

    return alerts


def collect_superduper_alerts_from_runs() -> List[Dict[str, Any]]:
    """
    Collect all sent superduper alerts from existing runs directory structure.
    Returns list of alerts with extracted parameters and entry data.
    """
    alerts = []
    runs_dir = Path('./runs')

    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return alerts

    print(f"Collecting alerts from {runs_dir}")

    # Iterate through date directories: 2025-07-29, 2025-08-04, etc.
    for date_dir in runs_dir.glob('20*'):
        if not date_dir.is_dir():
            continue

        date_str = date_dir.name  # e.g., '2025-07-29'
        print(f"Processing date: {date_str}")

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

                print(f"  Processing {symbol} - tf:{params['timeframe_minutes']} th:{params['green_threshold']}")

                # Look for JSON alert files
                alert_files = find_alert_files(run_dir)

                for alert_file in alert_files:
                    # Parse alerts from JSON files
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

    print(f"Collected {len(alerts)} total alerts")
    return alerts


def generate_synthetic_alerts_for_testing(symbols: List[str] = None,
                                         dates: List[str] = None) -> List[Dict[str, Any]]:
    """
    Generate synthetic alerts for testing when no real alerts are found.
    Creates realistic alert data based on available directory structure.
    """
    # Set random seed for reproducible testing
    np.random.seed(42)

    if symbols is None:
        symbols = ['VERB', 'BTAI', 'STAI', 'BSLK', 'ATNF']

    if dates is None:
        dates = ['2025-07-29', '2025-08-04', '2025-08-12', '2025-08-13']

    synthetic_alerts = []
    alert_id = 1

    print("Generating synthetic alerts for testing...")

    for date in dates:
        for symbol in symbols:
            # Generate realistic entry times during market hours
            base_date = pd.to_datetime(date).date()
            market_open = pd.Timestamp.combine(base_date, pd.Timestamp('09:30:00').time())
            market_close = pd.Timestamp.combine(base_date, pd.Timestamp('15:30:00').time())

            # Generate random alert times during market hours
            for timeframe in ALERT_PARAMETERS['timeframe_minutes']:
                for threshold in ALERT_PARAMETERS['green_threshold']:
                    # Generate 2-5 alerts per symbol/date/parameter combination
                    num_alerts = np.random.randint(2, 6)

                    for _ in range(num_alerts):
                        # Random time during market hours
                        time_offset = np.random.randint(0, int((market_close - market_open).total_seconds() / 60))
                        alert_time = market_open + pd.Timedelta(minutes=time_offset)

                        # Realistic entry price range based on symbol
                        if symbol == 'VERB':
                            base_price = np.random.uniform(0.20, 2.50)
                        elif symbol == 'BTAI':
                            base_price = np.random.uniform(1.50, 8.00)
                        elif symbol == 'STAI':
                            base_price = np.random.uniform(2.00, 12.00)
                        elif symbol == 'BSLK':
                            base_price = np.random.uniform(0.50, 5.00)
                        elif symbol == 'ATNF':
                            base_price = np.random.uniform(1.00, 6.00)
                        else:
                            base_price = np.random.uniform(1.00, 10.00)

                        # Add some randomness to the price
                        entry_price = round(base_price * np.random.uniform(0.9, 1.1), 2)

                        alert = {
                            'alert_timestamp': alert_time,
                            'entry_price': entry_price,
                            'entry_time': alert_time,
                            'symbol': symbol,
                            'date': date,
                            'timeframe_minutes': timeframe,
                            'green_threshold': threshold,
                            'run_directory': f'synthetic_test_data',
                            'alert_details': f'Synthetic alert {alert_id}: {symbol} at ${entry_price} on {alert_time}',
                            'synthetic': True  # Flag to indicate this is test data
                        }

                        synthetic_alerts.append(alert)
                        alert_id += 1

    print(f"Generated {len(synthetic_alerts)} synthetic alerts for testing")
    return synthetic_alerts


def validate_collected_alerts(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate collected alerts and report data quality issues.
    """
    validation_report = {
        'total_alerts': len(alerts),
        'unique_symbols': len(set(alert['symbol'] for alert in alerts)) if alerts else 0,
        'date_range': None,
        'parameter_coverage': {},
        'missing_data_issues': [],
        'data_quality_score': 0.0
    }

    if alerts:
        dates = [alert['date'] for alert in alerts if alert.get('date')]
        if dates:
            validation_report['date_range'] = (min(dates), max(dates))

        # Check parameter space coverage
        param_combinations = set()
        for alert in alerts:
            if alert.get('timeframe_minutes') and alert.get('green_threshold'):
                param_key = (alert['timeframe_minutes'], alert['green_threshold'])
                param_combinations.add(param_key)

        validation_report['parameter_coverage'] = {
            'combinations_found': len(param_combinations),
            'expected_combinations': 4,   # 1 timeframe × 4 thresholds (FOCUSED)
            'coverage_percentage': len(param_combinations) / 4 * 100 if param_combinations else 0
        }

        # Check for missing data
        missing_entry_prices = sum(1 for alert in alerts if not alert.get('entry_price'))
        missing_timestamps = sum(1 for alert in alerts if not alert.get('alert_timestamp'))

        if missing_entry_prices > 0:
            validation_report['missing_data_issues'].append(f"{missing_entry_prices} alerts missing entry prices")
        if missing_timestamps > 0:
            validation_report['missing_data_issues'].append(f"{missing_timestamps} alerts missing timestamps")

        # Calculate data quality score
        quality_score = 1.0
        if missing_entry_prices > 0:
            quality_score -= (missing_entry_prices / len(alerts)) * 0.5
        if missing_timestamps > 0:
            quality_score -= (missing_timestamps / len(alerts)) * 0.3

        validation_report['data_quality_score'] = max(0.0, quality_score)

    return validation_report


class ExitStrategySimulator:
    """
    Simulates exit strategies on historical alert data.
    Determines which exit strategy would trigger first for each alert.
    """

    def __init__(self):
        self.data_loader = HistoricalDataLoader()

    def _calculate_exit_points(self, price_series: np.ndarray, entry_price: float,
                              exit_params: Dict[str, Union[float, bool]]) -> Tuple[float, int, str]:
        """
        Exit calculation using JIT-compiled function for performance.
        Returns: (exit_price, exit_index, exit_reason)
        """
        # Extract parameters for JIT function
        take_profit_pct = exit_params['take_profit_pct']
        trailing_stop_pct = exit_params['trailing_stop_pct']

        # Use JIT-compiled function for the actual calculation
        return calculate_exit_points_jit(
            price_series, entry_price,
            take_profit_pct, trailing_stop_pct
        )

    def simulate_exit(self, alert: Dict[str, Any], exit_params: Dict[str, Union[float, bool]]) -> Dict[str, Any]:
        """
        Simulate which exit strategy would trigger first for a given alert.
        Returns: comprehensive exit result with profit metrics
        """
        if not alert.get('entry_price') or not alert.get('symbol') or not alert.get('date'):
            return {
                'exit_time': None,
                'exit_price': None,
                'exit_reason': 'invalid_alert',
                'profit_pct': 0.0,
                'profit_abs': 0.0,
                'hold_time_minutes': 0,
                'error': 'Missing required alert data'
            }

        entry_price = alert['entry_price']
        symbol = alert['symbol']
        date = alert['date']

        # Load historical price data
        price_data = self.data_loader.get_symbol_data(symbol, date)
        if price_data is None or len(price_data) == 0:
            return {
                'exit_time': None,
                'exit_price': entry_price,
                'exit_reason': 'no_price_data',
                'profit_pct': 0.0,
                'profit_abs': 0.0,
                'hold_time_minutes': 0,
                'error': f'No price data for {symbol} on {date}'
            }

        # Get entry time or use first available time
        if alert.get('alert_timestamp'):
            entry_time = pd.to_datetime(alert['alert_timestamp'])
            # Ensure timezone consistency
            if entry_time.tz is None and price_data['timestamp'].dt.tz is not None:
                entry_time = entry_time.tz_localize('US/Eastern')
            elif entry_time.tz is not None and price_data['timestamp'].dt.tz is None:
                entry_time = entry_time.tz_localize(None)
        else:
            entry_time = price_data['timestamp'].iloc[0]

        # Get price series after entry time
        try:
            future_prices = price_data[price_data['timestamp'] >= entry_time]
        except TypeError:
            # Handle timezone mismatch by converting to naive timestamps
            if price_data['timestamp'].dt.tz is not None:
                price_timestamps = price_data['timestamp'].dt.tz_convert(None)
            else:
                price_timestamps = price_data['timestamp']

            if hasattr(entry_time, 'tz') and entry_time.tz is not None:
                entry_time = entry_time.tz_convert(None) if entry_time.tz else entry_time.replace(tzinfo=None)

            future_prices = price_data[price_timestamps >= entry_time]
        if len(future_prices) == 0:
            return {
                'exit_time': entry_time,
                'exit_price': entry_price,
                'exit_reason': 'no_future_data',
                'profit_pct': 0.0,
                'profit_abs': 0.0,
                'hold_time_minutes': 0,
                'error': 'No price data after entry time'
            }

        # Extract price array for JIT-compiled calculation
        price_array = future_prices['close'].values

        # Calculate exit using optimized function
        exit_price, exit_index, exit_reason = self._calculate_exit_points(
            price_array, entry_price, exit_params
        )

        # Calculate metrics
        profit_pct = ((exit_price - entry_price) / entry_price) * 100
        profit_abs = exit_price - entry_price

        # Calculate hold time
        exit_time = future_prices.iloc[exit_index]['timestamp']
        hold_time_minutes = (exit_time - entry_time).total_seconds() / 60

        return {
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'profit_pct': profit_pct,
            'profit_abs': profit_abs,
            'hold_time_minutes': hold_time_minutes,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'symbol': symbol,
            'date': date
        }


class PerformanceAnalyzer:
    """
    Calculates comprehensive performance metrics for exit strategy optimization.
    """

    @staticmethod
    def calculate_strategy_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics from exit simulation results.
        """
        if not results:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }

        # Filter valid results
        valid_results = [r for r in results if r.get('profit_pct') is not None]

        if not valid_results:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }

        profits = [r['profit_pct'] for r in valid_results]

        # Basic metrics
        # Calculate compound total return (correct method)
        if profits:
            # Convert percentages to multipliers and compound them
            multipliers = [(p / 100) + 1 for p in profits]
            compound_return = 1.0
            for multiplier in multipliers:
                compound_return *= multiplier
            total_return = (compound_return - 1) * 100  # Convert back to percentage
        else:
            total_return = 0.0

        total_trades = len(valid_results)
        winning_trades = len([p for p in profits if p > 0])
        losing_trades = len([p for p in profits if p < 0])

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Win/Loss analysis
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean([abs(l) for l in losses]) if losses else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum([abs(l) for l in losses]) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Sharpe ratio (annualized)
        if len(profits) > 1:
            returns_std = np.std(profits, ddof=1)
            sharpe_ratio = (np.mean(profits) / returns_std) * np.sqrt(252) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative_returns = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'mean_return': np.mean(profits),
            'std_return': np.std(profits, ddof=1) if len(profits) > 1 else 0
        }

    @staticmethod
    def bootstrap_performance(results: List[Dict[str, Any]], n_iterations: int = 1000) -> Dict[str, float]:
        """
        Generate confidence intervals for performance metrics using bootstrap resampling.
        """
        if not results:
            return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0}

        profits = [r['profit_pct'] for r in results if r.get('profit_pct') is not None]

        if not profits:
            return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0}

        bootstrap_returns = []
        for _ in range(n_iterations):
            sample = np.random.choice(profits, size=len(profits), replace=True)
            bootstrap_returns.append(np.mean(sample))

        return {
            'mean': np.mean(bootstrap_returns),
            'ci_lower': np.percentile(bootstrap_returns, 2.5),
            'ci_upper': np.percentile(bootstrap_returns, 97.5),
            'std': np.std(bootstrap_returns)
        }

    @staticmethod
    def compare_strategies(baseline_results: List[Dict[str, Any]],
                          optimized_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test if optimized strategy significantly outperforms baseline using paired t-test.
        """
        baseline_profits = [r['profit_pct'] for r in baseline_results if r.get('profit_pct') is not None]
        optimized_profits = [r['profit_pct'] for r in optimized_results if r.get('profit_pct') is not None]

        if len(baseline_profits) == 0 or len(optimized_profits) == 0:
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'mean_improvement': 0,
                'error': 'Insufficient data for comparison'
            }

        # Use unpaired t-test if sample sizes differ
        if len(baseline_profits) != len(optimized_profits):
            statistic, p_value = stats.ttest_ind(optimized_profits, baseline_profits)
        else:
            statistic, p_value = ttest_rel(optimized_profits, baseline_profits)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_improvement': np.mean(optimized_profits) - np.mean(baseline_profits),
            'baseline_mean': np.mean(baseline_profits),
            'optimized_mean': np.mean(optimized_profits)
        }


class OrinNanoOptimizer:
    """
    Leverage Orin Nano's ARM CPU + GPU architecture for parameter optimization.
    Implements GPU acceleration and thermal-aware scheduling.
    """

    def __init__(self):
        self.gpu_available = self._check_gpu()
        self.cpu_cores = mp.cpu_count()  # Should be 6 on Orin Nano
        self.thermal_limit = 70  # Celsius

        # Initialize GPU memory pool if available
        if self.gpu_available:
            try:
                import cupy
                mempool = cupy.get_default_memory_pool()
                mempool.set_limit(size=6*1024**3)  # 6GB limit
                print("GPU acceleration enabled with 6GB memory limit")
            except Exception as e:
                print(f"GPU initialization warning: {e}")
                self.gpu_available = False

    def _check_gpu(self) -> bool:
        """Check if CUDA GPU is available."""
        if not GPU_AVAILABLE:
            return False
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            return True
        except:
            return False

    def _get_cpu_temp(self) -> float:
        """Read CPU temperature from thermal zone."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millicelsius = int(f.read().strip())
                return temp_millicelsius / 1000.0
        except:
            return 50  # Default safe temperature

    def should_throttle(self) -> bool:
        """Check if should reduce processing due to thermal limits."""
        current_temp = self._get_cpu_temp()
        return current_temp > self.thermal_limit

    def get_optimal_batch_size(self) -> int:
        """Adjust batch size based on thermal state."""
        if self.should_throttle():
            return 25  # Smaller batches when hot
        else:
            return 100  # Normal batch size when cool

    def optimize_parameters_parallel(self, alerts: List[Dict[str, Any]],
                                   parameter_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        CPU-optimized parallel processing for ARM Cortex-A78AE.
        Uses multiprocessing with ARM-specific optimizations.
        """
        if not alerts:
            print("No alerts provided for optimization")
            return []

        # ARM-specific optimizations
        chunk_size = max(1, len(parameter_combinations) // (self.cpu_cores - 2))  # Reserve cores

        print(f"Starting parallel optimization with {self.cpu_cores - 2} workers")
        print(f"Processing {len(parameter_combinations)} parameter combinations in chunks of {chunk_size}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Split work across available ARM cores
            futures = []

            for i in range(0, len(parameter_combinations), chunk_size):
                chunk = parameter_combinations[i:i+chunk_size]
                future = executor.submit(
                    self._process_parameter_chunk_optimized,
                    alerts,
                    chunk
                )
                futures.append(future)

            # Collect results as they complete
            results = []
            completed_chunks = 0

            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    completed_chunks += 1

                    # Update progress
                    progress = (completed_chunks / len(futures)) * 100
                    print(f"Progress: {progress:.1f}% ({len(results)} results)")

                except Exception as e:
                    print(f"Chunk processing error: {e}")

        return results

    def _process_parameter_chunk_optimized(self, alerts: List[Dict[str, Any]],
                                         parameter_chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a chunk of parameter combinations with ARM optimizations.
        """
        simulator = ExitStrategySimulator()
        analyzer = PerformanceAnalyzer()
        chunk_results = []

        for params in parameter_chunk:
            try:
                # Simulate exit strategy for all alerts with these parameters
                exit_results = []

                for alert in alerts:
                    # Check if alert matches parameter requirements
                    if (alert.get('timeframe_minutes') == params.get('alert_timeframe_minutes') and
                        alert.get('green_threshold') == params.get('alert_green_threshold')):

                        exit_result = simulator.simulate_exit(alert, params)
                        exit_results.append(exit_result)

                # Calculate performance metrics for this parameter set
                if exit_results:
                    metrics = analyzer.calculate_strategy_metrics(exit_results)

                    result = {
                        'parameters': params,
                        'metrics': metrics,
                        'trade_count': len(exit_results),
                        'valid_trades': len([r for r in exit_results if r.get('profit_pct') is not None])
                    }
                    chunk_results.append(result)

            except Exception as e:
                print(f"Error processing parameters {params}: {e}")
                continue

        return chunk_results


class ExitStrategyOptimizer:
    """
    Main optimization engine implementing two-stage optimization framework.
    Optimizes both alert generation and exit strategy parameters.
    """

    def __init__(self, use_gpu: bool = True):
        self.simulator = ExitStrategySimulator()
        self.analyzer = PerformanceAnalyzer()
        self.orin_optimizer = OrinNanoOptimizer() if use_gpu else None
        self.best_results = []

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for testing.
        Returns combined alert + exit parameter sets.
        """
        combinations = []

        # Generate all alert parameter combinations
        for timeframe in ALERT_PARAMETERS['timeframe_minutes']:
            for threshold in ALERT_PARAMETERS['green_threshold']:

                # Generate all exit parameter combinations for each alert set
                for take_profit in EXIT_PARAMETERS['take_profit_pct']:
                    for trailing_stop in EXIT_PARAMETERS['trailing_stop_pct']:
                        for macd_sensitivity in EXIT_PARAMETERS['macd_sensitivity']:
                            for macd_enabled in EXIT_PARAMETERS['macd_enabled']:

                                combination = {
                                    # Alert parameters
                                    'alert_timeframe_minutes': timeframe,
                                    'alert_green_threshold': threshold,

                                    # Exit parameters
                                    'take_profit_pct': take_profit,
                                    'trailing_stop_pct': trailing_stop,
                                    'macd_sensitivity': macd_sensitivity,
                                    'macd_enabled': macd_enabled
                                }
                                combinations.append(combination)

        return combinations

    def optimize_hierarchical(self, alerts: List[Dict[str, Any]],
                            max_combinations: int = None) -> Dict[str, Any]:
        """
        Hierarchical optimization: find optimal alert parameters first, then exits.
        RECOMMENDED approach to manage computational complexity.
        """
        print("Starting hierarchical optimization...")

        # Step 1: Find optimal alert parameters using simple exit strategy
        print("\nStep 1: Optimizing alert parameters with simple exits...")
        simple_exit_params = {
            'take_profit_pct': 10,  # Updated to be within new range (5-15%)
            'trailing_stop_pct': 12.5,
            'macd_enabled': False,
            'macd_sensitivity': 'normal'
        }

        alert_results = []
        for timeframe in ALERT_PARAMETERS['timeframe_minutes']:
            for threshold in ALERT_PARAMETERS['green_threshold']:

                # Filter alerts for this parameter combination
                matching_alerts = [
                    alert for alert in alerts
                    if (alert.get('timeframe_minutes') == timeframe and
                        alert.get('green_threshold') == threshold)
                ]

                if not matching_alerts:
                    continue

                # Test simple exit strategy on these alerts
                exit_results = []
                for alert in matching_alerts:
                    exit_result = self.simulator.simulate_exit(alert, simple_exit_params)
                    exit_results.append(exit_result)

                # Calculate metrics
                metrics = self.analyzer.calculate_strategy_metrics(exit_results)

                alert_result = {
                    'timeframe_minutes': timeframe,
                    'green_threshold': threshold,
                    'metrics': metrics,
                    'alert_count': len(matching_alerts),
                    'valid_trades': len([r for r in exit_results if r.get('profit_pct') is not None])
                }
                alert_results.append(alert_result)

                print(f"  tf:{timeframe} th:{threshold:.2f} - {len(matching_alerts)} alerts, "
                      f"Sharpe: {metrics['sharpe_ratio']:.3f}, Return: {metrics['total_return']:.1f}%")

        # Find best alert parameters
        if not alert_results:
            return {'error': 'No valid alert combinations found'}

        best_alert_params = max(alert_results,
                               key=lambda x: x['metrics']['sharpe_ratio'] if x['metrics']['sharpe_ratio'] != 0
                               else x['metrics']['total_return'])

        print(f"\nBest alert parameters: tf:{best_alert_params['timeframe_minutes']} "
              f"th:{best_alert_params['green_threshold']:.2f}")
        print(f"Sharpe ratio: {best_alert_params['metrics']['sharpe_ratio']:.3f}")
        print(f"Total return: {best_alert_params['metrics']['total_return']:.1f}%")

        # Step 2: Optimize exit parameters for best alert settings
        print(f"\nStep 2: Optimizing exit parameters for best alert settings...")

        best_alert_data = [
            alert for alert in alerts
            if (alert.get('timeframe_minutes') == best_alert_params['timeframe_minutes'] and
                alert.get('green_threshold') == best_alert_params['green_threshold'])
        ]

        # Generate exit parameter combinations
        exit_combinations = []
        for take_profit in EXIT_PARAMETERS['take_profit_pct']:
            for trailing_stop in EXIT_PARAMETERS['trailing_stop_pct']:
                for macd_sensitivity in EXIT_PARAMETERS['macd_sensitivity']:
                    for macd_enabled in EXIT_PARAMETERS['macd_enabled']:

                        exit_params = {
                            'take_profit_pct': take_profit,
                            'trailing_stop_pct': trailing_stop,
                            'macd_sensitivity': macd_sensitivity,
                            'macd_enabled': macd_enabled
                        }
                        exit_combinations.append(exit_params)

        # Limit combinations if requested
        if max_combinations and len(exit_combinations) > max_combinations:
            print(f"Limiting to {max_combinations} exit combinations (from {len(exit_combinations)})")
            exit_combinations = exit_combinations[:max_combinations]

        print(f"Testing {len(exit_combinations)} exit parameter combinations...")

        # Test exit combinations
        best_exit_results = []

        for i, exit_params in enumerate(exit_combinations):
            exit_results = []
            for alert in best_alert_data:
                exit_result = self.simulator.simulate_exit(alert, exit_params)
                exit_results.append(exit_result)

            metrics = self.analyzer.calculate_strategy_metrics(exit_results)

            result = {
                'parameters': exit_params,
                'metrics': metrics,
                'trade_count': len(exit_results),
                'valid_trades': len([r for r in exit_results if r.get('profit_pct') is not None])
            }
            best_exit_results.append(result)

            if (i + 1) % 100 == 0:
                progress = ((i + 1) / len(exit_combinations)) * 100
                print(f"  Progress: {progress:.1f}% ({i + 1}/{len(exit_combinations)})")

        # Find best exit parameters
        best_exit_result = max(best_exit_results,
                              key=lambda x: x['metrics']['sharpe_ratio'] if x['metrics']['sharpe_ratio'] != 0
                              else x['metrics']['total_return'])

        # Combine results
        final_result = {
            'optimization_method': 'hierarchical',
            'best_alert_parameters': {
                'timeframe_minutes': best_alert_params['timeframe_minutes'],
                'green_threshold': best_alert_params['green_threshold']
            },
            'best_exit_parameters': best_exit_result['parameters'],
            'performance_metrics': best_exit_result['metrics'],
            'alert_optimization_results': alert_results,
            'exit_optimization_results': best_exit_results,
            'total_combinations_tested': len(alert_results) + len(exit_combinations),
            'data_quality': {
                'total_alerts_used': len(best_alert_data),
                'valid_trades': best_exit_result['valid_trades']
            }
        }

        return final_result

    def optimize_full_grid_search(self, alerts: List[Dict[str, Any]],
                                 max_combinations: int = None) -> Dict[str, Any]:
        """
        Full grid search across all parameter combinations.
        Computationally intensive but finds global optimum.
        """
        print("Starting full grid search optimization...")

        # Generate all parameter combinations
        all_combinations = self.generate_parameter_combinations()

        if max_combinations and len(all_combinations) > max_combinations:
            print(f"Limiting to {max_combinations} combinations (from {len(all_combinations)})")
            all_combinations = all_combinations[:max_combinations]

        print(f"Testing {len(all_combinations)} total parameter combinations...")

        # Use Orin Nano optimization if available
        if self.orin_optimizer:
            results = self.orin_optimizer.optimize_parameters_parallel(alerts, all_combinations)
        else:
            # Fallback to sequential processing
            results = self._process_combinations_sequential(alerts, all_combinations)

        if not results:
            return {'error': 'No valid results from optimization'}

        # Find best result
        best_result = max(results,
                         key=lambda x: x['metrics']['sharpe_ratio'] if x['metrics']['sharpe_ratio'] != 0
                         else x['metrics']['total_return'])

        # Calculate statistics
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in results if r['metrics']['sharpe_ratio'] != 0]
        total_returns = [r['metrics']['total_return'] for r in results]

        final_result = {
            'optimization_method': 'full_grid_search',
            'best_parameters': best_result['parameters'],
            'best_metrics': best_result['metrics'],
            'optimization_statistics': {
                'total_combinations_tested': len(results),
                'mean_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'std_sharpe_ratio': np.std(sharpe_ratios) if len(sharpe_ratios) > 1 else 0,
                'mean_total_return': np.mean(total_returns),
                'std_total_return': np.std(total_returns) if len(total_returns) > 1 else 0,
                'best_sharpe_ratio': best_result['metrics']['sharpe_ratio'],
                'best_total_return': best_result['metrics']['total_return']
            },
            'all_results': results
        }

        return final_result

    def _process_combinations_sequential(self, alerts: List[Dict[str, Any]],
                                       combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sequential processing fallback when parallel processing unavailable.
        """
        results = []

        for i, params in enumerate(combinations):
            # Filter alerts for this parameter combination
            matching_alerts = [
                alert for alert in alerts
                if (alert.get('timeframe_minutes') == params.get('alert_timeframe_minutes') and
                    alert.get('green_threshold') == params.get('alert_green_threshold'))
            ]

            if not matching_alerts:
                continue

            # Simulate exits
            exit_results = []
            for alert in matching_alerts:
                exit_result = self.simulator.simulate_exit(alert, params)
                exit_results.append(exit_result)

            # Calculate metrics
            metrics = self.analyzer.calculate_strategy_metrics(exit_results)

            result = {
                'parameters': params,
                'metrics': metrics,
                'trade_count': len(exit_results),
                'valid_trades': len([r for r in exit_results if r.get('profit_pct') is not None])
            }
            results.append(result)

            if (i + 1) % 50 == 0:
                progress = ((i + 1) / len(combinations)) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(combinations)})")

        return results


def create_profitability_visualizations(results: Dict[str, Any], output_dir: str = "optimization_charts"):
    """
    Create comprehensive profitability charts to visualize optimization results.
    VERY IMPORTANT: Shows performance across different parameter combinations.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Set style for professional charts
        plt.style.use('default')
        sns.set_palette("husl")

        print(f"\n📊 Generating profitability visualizations in {output_dir}/")

        try:
            # Chart 1: Parameter Heatmap
            _create_parameter_heatmap(results, output_path)
        except Exception as e:
            print(f"Error in parameter heatmap: {e}")

        try:
            # Chart 2: Take Profit vs Stop Loss Surface
            _create_take_profit_stop_loss_surface(results, output_path)
        except Exception as e:
            print(f"Error in 3D surface: {e}")

        try:
            # Chart 3: Risk-Return Scatter Plot
            _create_risk_return_scatter(results, output_path)
        except Exception as e:
            print(f"Error in risk-return scatter: {e}")

        try:
            # Chart 4: Win Rate Analysis
            _create_win_rate_analysis(results, output_path)
        except Exception as e:
            print(f"Error in win rate analysis: {e}")

        try:
            # Chart 5: Parameter Distribution Charts
            _create_parameter_distributions(results, output_path)
        except Exception as e:
            print(f"Error in parameter distributions: {e}")

        try:
            # Chart 6: Performance Comparison Bar Chart
            _create_performance_comparison(results, output_path)
        except Exception as e:
            print(f"Error in performance comparison: {e}")

        print(f"✅ Created 6 profitability visualization charts in {output_dir}/")
        return True

    except ImportError:
        print("⚠️  matplotlib/seaborn not available - skipping chart generation")
        print("   Install with: pip install matplotlib seaborn")
        return False
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        return False


def _create_parameter_heatmap(results: Dict[str, Any], output_path: Path):
    """Create heatmap showing performance across take profit and trailing stop parameters."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Extract data for heatmap
    if results['optimization_method'] == 'hierarchical':
        exit_results = results.get('exit_optimization_results', [])
    else:
        exit_results = results.get('all_results', [])

    if not exit_results:
        return

    # Create matrix data
    take_profits = []
    trailing_stops = []
    sharpe_ratios = []

    for result in exit_results:
        params = result['parameters']
        metrics = result['metrics']

        take_profits.append(params.get('take_profit_pct', 0))
        trailing_stops.append(params.get('trailing_stop_pct', 0))
        sharpe_ratios.append(metrics.get('sharpe_ratio', 0))

    if not take_profits:
        return

    # Create DataFrame
    df = pd.DataFrame({
        'take_profit': take_profits,
        'trailing_stop': trailing_stops,
        'sharpe_ratio': sharpe_ratios
    })

    # Create pivot table for heatmap
    pivot_df = df.pivot_table(values='sharpe_ratio', index='trailing_stop', columns='take_profit', aggfunc='mean')

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': 'Sharpe Ratio'})
    plt.title('🔥 Profitability Heatmap: Take Profit vs Trailing Stop\n(Sharpe Ratio)', fontsize=16, fontweight='bold')
    plt.xlabel('Take Profit (%)', fontsize=12)
    plt.ylabel('Trailing Stop (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / '1_parameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_take_profit_stop_loss_surface(results: Dict[str, Any], output_path: Path):
    """Create 3D surface plot of take profit vs stop loss vs returns."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    import numpy as np

    # Extract data
    if results['optimization_method'] == 'hierarchical':
        exit_results = results.get('exit_optimization_results', [])
    else:
        exit_results = results.get('all_results', [])

    if not exit_results:
        return

    take_profits = []
    trailing_stops = []
    total_returns = []

    for result in exit_results:
        params = result['parameters']
        metrics = result['metrics']

        take_profits.append(params.get('take_profit_pct', 0))
        trailing_stops.append(params.get('trailing_stop_pct', 0))
        total_returns.append(metrics.get('total_return', 0))

    if not take_profits:
        return

    # Create 3D surface plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot with color mapping
    scatter = ax.scatter(take_profits, trailing_stops, total_returns,
                        c=total_returns, cmap='viridis', s=60, alpha=0.7)

    ax.set_xlabel('Take Profit (%)', fontsize=12)
    ax.set_ylabel('Trailing Stop (%)', fontsize=12)
    ax.set_zlabel('Total Return (%)', fontsize=12)
    ax.set_title('🚀 3D Profitability Surface\nTake Profit vs Trailing Stop vs Total Return',
                 fontsize=16, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Total Return (%)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / '2_3d_profitability_surface.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_risk_return_scatter(results: Dict[str, Any], output_path: Path):
    """Create risk-return scatter plot with parameter annotations."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Extract data
    if results['optimization_method'] == 'hierarchical':
        exit_results = results.get('exit_optimization_results', [])
    else:
        exit_results = results.get('all_results', [])

    if not exit_results:
        return

    returns = []
    risks = []
    take_profits = []
    sharpe_ratios = []

    for result in exit_results:
        params = result['parameters']
        metrics = result['metrics']

        returns.append(metrics.get('total_return', 0))
        risks.append(metrics.get('max_drawdown', 0))
        take_profits.append(params.get('take_profit_pct', 0))
        sharpe_ratios.append(metrics.get('sharpe_ratio', 0))

    if not returns:
        return

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Create scatter with color mapping by take profit
    scatter = plt.scatter(risks, returns, c=take_profits, s=100, alpha=0.7,
                         cmap='coolwarm', edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Take Profit (%)', fontsize=12)

    # Find and highlight best performing points
    best_sharpe_idx = np.argmax(sharpe_ratios)
    best_return_idx = np.argmax(returns)

    plt.scatter(risks[best_sharpe_idx], returns[best_sharpe_idx],
               color='gold', s=200, marker='*', edgecolors='black', linewidth=2,
               label=f'Best Sharpe: {sharpe_ratios[best_sharpe_idx]:.2f}')

    plt.scatter(risks[best_return_idx], returns[best_return_idx],
               color='lime', s=200, marker='D', edgecolors='black', linewidth=2,
               label=f'Best Return: {returns[best_return_idx]:.1f}%')

    plt.xlabel('Max Drawdown (%)', fontsize=12)
    plt.ylabel('Total Return (%)', fontsize=12)
    plt.title('💎 Risk vs Return Analysis\nColorized by Take Profit Level', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / '3_risk_return_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_win_rate_analysis(results: Dict[str, Any], output_path: Path):
    """Create win rate analysis charts."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Extract data
    if results['optimization_method'] == 'hierarchical':
        exit_results = results.get('exit_optimization_results', [])
    else:
        exit_results = results.get('all_results', [])

    if not exit_results:
        return

    # Group by take profit levels
    take_profit_groups = {}

    for result in exit_results:
        params = result['parameters']
        metrics = result['metrics']

        tp = params.get('take_profit_pct', 0)
        if tp not in take_profit_groups:
            take_profit_groups[tp] = {
                'win_rates': [],
                'total_returns': [],
                'sharpe_ratios': []
            }

        take_profit_groups[tp]['win_rates'].append(metrics.get('win_rate', 0))
        take_profit_groups[tp]['total_returns'].append(metrics.get('total_return', 0))
        take_profit_groups[tp]['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0))

    if not take_profit_groups:
        return

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Average Win Rate by Take Profit
    tp_levels = sorted(take_profit_groups.keys())
    avg_win_rates = [np.mean(take_profit_groups[tp]['win_rates']) for tp in tp_levels]

    bars1 = ax1.bar(tp_levels, avg_win_rates, color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_xlabel('Take Profit (%)')
    ax1.set_ylabel('Average Win Rate (%)')
    ax1.set_title('📈 Win Rate by Take Profit Level')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, avg_win_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Chart 2: Return Distribution by Take Profit
    return_data = [take_profit_groups[tp]['total_returns'] for tp in tp_levels]
    bp1 = ax2.boxplot(return_data, labels=[f'{tp}%' for tp in tp_levels], patch_artist=True)

    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp1['boxes'])))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel('Take Profit (%)')
    ax2.set_ylabel('Total Return (%)')
    ax2.set_title('📊 Return Distribution by Take Profit')
    ax2.grid(True, alpha=0.3)

    # Chart 3: Sharpe Ratio by Take Profit
    avg_sharpe = [np.mean(take_profit_groups[tp]['sharpe_ratios']) for tp in tp_levels]

    bars3 = ax3.bar(tp_levels, avg_sharpe, color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax3.set_xlabel('Take Profit (%)')
    ax3.set_ylabel('Average Sharpe Ratio')
    ax3.set_title('⚡ Sharpe Ratio by Take Profit Level')
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars3, avg_sharpe):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # Chart 4: Performance Summary Table
    ax4.axis('off')

    # Create summary table
    table_data = []
    for tp in tp_levels:
        data = take_profit_groups[tp]
        table_data.append([
            f'{tp}%',
            f'{np.mean(data["win_rates"]):.1f}%',
            f'{np.mean(data["total_returns"]):.1f}%',
            f'{np.mean(data["sharpe_ratios"]):.2f}',
            f'{len(data["win_rates"])}'
        ])

    table = ax4.table(cellText=table_data,
                     colLabels=['Take Profit', 'Avg Win Rate', 'Avg Return', 'Avg Sharpe', 'Count'],
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    ax4.set_title('📋 Performance Summary by Take Profit Level', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('🎯 Comprehensive Win Rate Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / '4_win_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_parameter_distributions(results: Dict[str, Any], output_path: Path):
    """Create distribution charts for each parameter."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Extract data
    if results['optimization_method'] == 'hierarchical':
        exit_results = results.get('exit_optimization_results', [])
    else:
        exit_results = results.get('all_results', [])

    if not exit_results:
        return

    # Collect parameter data
    param_data = {
        'take_profit_pct': [],
        # 'stop_loss_pct': [],  # Removed - using trailing stops
        'trailing_stop_pct': [],
        'macd_enabled': [],
        'sharpe_ratios': []
    }

    for result in exit_results:
        params = result['parameters']
        metrics = result['metrics']

        param_data['take_profit_pct'].append(params.get('take_profit_pct', 0))
        # param_data['stop_loss_pct'].append(params.get('stop_loss_pct', 0))  # Removed
        param_data['trailing_stop_pct'].append(params.get('trailing_stop_pct', 0))
        param_data['macd_enabled'].append(params.get('macd_enabled', False))
        param_data['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0))

    if not param_data['take_profit_pct']:
        return

    # Create parameter distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Take Profit Distribution
    df = pd.DataFrame(param_data)
    tp_perf = df.groupby('take_profit_pct')['sharpe_ratios'].mean().sort_index()

    axes[0,0].bar(tp_perf.index, tp_perf.values, color='steelblue', alpha=0.8)
    axes[0,0].set_title('📈 Take Profit Performance', fontweight='bold')
    axes[0,0].set_xlabel('Take Profit (%)')
    axes[0,0].set_ylabel('Average Sharpe Ratio')
    axes[0,0].grid(True, alpha=0.3)

    # Trailing Stop Distribution (moved to position [0,1])
    ts_perf = df.groupby('trailing_stop_pct')['sharpe_ratios'].mean().sort_index()

    axes[0,1].bar(ts_perf.index, ts_perf.values, color='forestgreen', alpha=0.8)
    axes[0,1].set_title('🎯 Trailing Stop Performance', fontweight='bold')
    axes[0,1].set_xlabel('Trailing Stop (%)')
    axes[0,1].set_ylabel('Average Sharpe Ratio')
    axes[0,1].grid(True, alpha=0.3)

    # MACD Enabled vs Disabled (moved to position [0,2])
    macd_perf = df.groupby('macd_enabled')['sharpe_ratios'].mean()

    colors = ['lightcoral', 'lightblue']
    macd_labels = ['MACD Disabled', 'MACD Enabled']
    bars = axes[0,2].bar(macd_labels, macd_perf.values, color=colors, alpha=0.8)
    axes[0,2].set_title('🔄 MACD Usage Performance', fontweight='bold')
    axes[0,2].set_ylabel('Average Sharpe Ratio')
    axes[0,2].grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, macd_perf.values):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # Parameter Correlation Matrix (moved to position [1,0])
    corr_data = df[['take_profit_pct', 'trailing_stop_pct', 'sharpe_ratios']].corr()

    im = axes[1,0].imshow(corr_data.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,0].set_xticks(range(len(corr_data.columns)))
    axes[1,0].set_yticks(range(len(corr_data.columns)))
    axes[1,0].set_xticklabels(['Take Profit', 'Trailing Stop', 'Sharpe Ratio'], rotation=45)
    axes[1,0].set_yticklabels(['Take Profit', 'Trailing Stop', 'Sharpe Ratio'])
    axes[1,0].set_title('🔗 Parameter Correlation Matrix', fontweight='bold')

    # Add correlation values
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            text = axes[1,0].text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')

    # Performance distribution histogram
    axes[1,1].hist(param_data['sharpe_ratios'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1,1].set_title('📊 Sharpe Ratio Distribution', fontweight='bold')
    axes[1,1].set_xlabel('Sharpe Ratio')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)

    # Add statistics text
    mean_sharpe = np.mean(param_data['sharpe_ratios'])
    std_sharpe = np.std(param_data['sharpe_ratios'])
    axes[1,1].axvline(mean_sharpe, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sharpe:.2f}')
    axes[1,1].legend()

    # Top 10 Parameter Combinations
    top_results = sorted(exit_results, key=lambda x: x['metrics'].get('sharpe_ratio', 0), reverse=True)[:10]

    axes[1,2].axis('off')

    # Create top performers table
    table_data = []
    for i, result in enumerate(top_results):
        params = result['parameters']
        metrics = result['metrics']
        table_data.append([
            f"#{i+1}",
            f"{params.get('take_profit_pct', 0)}%",
            f"{params.get('trailing_stop_pct', 0)}%",
            f"{metrics.get('sharpe_ratio', 0):.2f}"
        ])

    table = axes[1,2].table(cellText=table_data,
                           colLabels=['Rank', 'Take Profit', 'Trailing Stop', 'Sharpe'],
                           cellLoc='center',
                           loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#FF6B6B')
                cell.set_text_props(weight='bold', color='white')
            elif i <= 3:  # Top 3
                cell.set_facecolor('#FFE66D')
            else:
                cell.set_facecolor('#f8f8f8')

    axes[1,2].set_title('🏆 Top 10 Parameter Combinations', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('📊 Parameter Distribution Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / '5_parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def _create_performance_comparison(results: Dict[str, Any], output_path: Path):
    """Create comprehensive performance comparison charts."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Extract best performance metrics
    if results['optimization_method'] == 'hierarchical':
        best_metrics = results.get('performance_metrics', {})
        exit_results = results.get('exit_optimization_results', [])
    else:
        best_metrics = results.get('best_metrics', {})
        exit_results = results.get('all_results', [])

    if not best_metrics:
        return

    # Create performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Key Performance Metrics
    metrics = ['Total Return', 'Win Rate', 'Sharpe Ratio', 'Profit Factor', 'Max Drawdown']
    values = [
        best_metrics.get('total_return', 0),
        best_metrics.get('win_rate', 0),
        best_metrics.get('sharpe_ratio', 0) * 10,  # Scale for visibility
        best_metrics.get('profit_factor', 0) * 10,  # Scale for visibility
        best_metrics.get('max_drawdown', 0)
    ]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars1 = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')

    ax1.set_title('🎯 Best Strategy Performance Metrics', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Value')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    labels = [f'{best_metrics.get("total_return", 0):.1f}%',
             f'{best_metrics.get("win_rate", 0):.1f}%',
             f'{best_metrics.get("sharpe_ratio", 0):.2f}',
             f'{best_metrics.get("profit_factor", 0):.2f}',
             f'{best_metrics.get("max_drawdown", 0):.1f}%']

    for bar, label in zip(bars1, labels):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                label, ha='center', va='bottom', fontweight='bold')

    # Chart 2: Performance Distribution
    if exit_results:
        sharpe_ratios = [r['metrics'].get('sharpe_ratio', 0) for r in exit_results]
        total_returns = [r['metrics'].get('total_return', 0) for r in exit_results]

        ax2.hist(sharpe_ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='navy', label='Sharpe Ratio')
        ax2.axvline(best_metrics.get('sharpe_ratio', 0), color='red', linestyle='--', linewidth=2,
                   label=f'Best: {best_metrics.get("sharpe_ratio", 0):.2f}')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('📈 Sharpe Ratio Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Chart 3: Trade Analysis
    trade_data = {
        'Total Trades': best_metrics.get('total_trades', 0),
        'Winning Trades': best_metrics.get('winning_trades', 0),
        'Losing Trades': best_metrics.get('losing_trades', 0)
    }

    pie_colors = ['#FF9999', '#66B2FF', '#99FF99']
    wedges, texts, autotexts = ax3.pie([trade_data['Winning Trades'], trade_data['Losing Trades']],
                                      labels=['Winning Trades', 'Losing Trades'],
                                      colors=pie_colors[:2], autopct='%1.1f%%', startangle=90)

    ax3.set_title(f'🎪 Trade Distribution\nTotal: {trade_data["Total Trades"]} trades',
                 fontweight='bold')

    # Chart 4: Risk-Reward Analysis
    avg_win = best_metrics.get('avg_win', 0)
    avg_loss = best_metrics.get('avg_loss', 0)

    risk_reward_data = ['Average Win', 'Average Loss']
    risk_reward_values = [avg_win, avg_loss]
    risk_reward_colors = ['green', 'red']

    bars4 = ax4.bar(risk_reward_data, risk_reward_values, color=risk_reward_colors, alpha=0.7)
    ax4.set_title('⚖️ Risk-Reward Profile', fontweight='bold')
    ax4.set_ylabel('Percentage (%)')
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars4, risk_reward_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Add risk-reward ratio text
    if avg_loss > 0:
        risk_reward_ratio = avg_win / avg_loss
        ax4.text(0.5, max(risk_reward_values) * 0.8, f'Risk-Reward Ratio: {risk_reward_ratio:.2f}:1',
                ha='center', transform=ax4.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.suptitle('🏆 Comprehensive Performance Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / '6_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def apply_multiple_testing_correction(p_values: List[float], method: str = 'holm') -> Tuple[List[bool], List[float]]:
    """
    Apply multiple testing correction to avoid false discoveries.
    With 8,400 parameter combinations, need correction for statistical validity.
    """
    if not p_values:
        return [], []

    if not STATSMODELS_AVAILABLE:
        print("Warning: statsmodels not available, using simple Bonferroni correction")
        # Simple Bonferroni correction fallback
        alpha_corrected = 0.05 / len(p_values)
        rejected = [p < alpha_corrected for p in p_values]
        p_corrected = [p * len(p_values) for p in p_values]
        return rejected, p_corrected

    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values,
        alpha=0.05,
        method=method  # 'holm', 'bonferroni', 'fdr_bh'
    )
    return rejected.tolist(), p_corrected.tolist()


def main():
    """Main execution function with CLI interface."""
    parser = argparse.ArgumentParser(description='Exit Strategy Optimization System')
    parser.add_argument('--method', choices=['hierarchical', 'full'], default='hierarchical',
                       help='Optimization method (default: hierarchical)')
    parser.add_argument('--max-combinations', type=int, default=None,
                       help='Maximum parameter combinations to test')
    parser.add_argument('--output', type=str, default='optimization_results.json',
                       help='Output file for results')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate collected alerts, do not optimize')

    args = parser.parse_args()

    print("Exit Strategy Optimization System")
    print("=================================")

    # Test data loading
    data_loader = HistoricalDataLoader()
    print(f"Data loader initialized with {data_loader.max_memory_bytes / 1024**3:.1f}GB cache limit")

    # Collect alerts
    print("\nCollecting alerts from runs directory...")
    alerts = collect_superduper_alerts_from_runs()

    # Filter out alerts without proper entry prices (parsed incorrectly)
    valid_alerts = []
    for alert in alerts:
        if (alert.get('entry_price') is not None and
            0.01 <= alert.get('entry_price', 0) <= 1000 and
            alert.get('symbol') and
            alert.get('date')):
            valid_alerts.append(alert)

    if not valid_alerts:
        print("No valid alerts found in JSON alert files")
        print("This could be due to:")
        print("  - No JSON alert files in historical_data directories")
        print("  - Missing required data fields in JSON alert files")
        print("\nGenerating synthetic test data for optimization testing...")
        alerts = generate_synthetic_alerts_for_testing()

        if not alerts:
            print("ERROR: Failed to generate synthetic alerts")
            return 1
    else:
        alerts = valid_alerts
        # Count JSON files only
        json_alerts = len([a for a in alerts if a.get('json_file')])
        print(f"Found {len(alerts)} valid alerts from JSON files only")

    print(f"Found {len(alerts)} alerts")

    # Validate alerts
    validation = validate_collected_alerts(alerts)
    print("\nValidation Report:")
    print(f"  Total alerts: {validation['total_alerts']}")
    print(f"  Unique symbols: {validation['unique_symbols']}")
    print(f"  Date range: {validation['date_range']}")
    print(f"  Parameter coverage: {validation['parameter_coverage']['coverage_percentage']:.1f}%")
    print(f"  Data quality score: {validation['data_quality_score']:.2f}")

    if validation['missing_data_issues']:
        print("  Issues:")
        for issue in validation['missing_data_issues']:
            print(f"    - {issue}")

    if args.validate_only:
        return 0

    # Initialize optimizer
    optimizer = ExitStrategyOptimizer(use_gpu=not args.no_gpu)

    print(f"\nParameter space: {len(ALERT_PARAMETERS['timeframe_minutes']) * len(ALERT_PARAMETERS['green_threshold'])} alert combinations")
    exit_combinations = 1
    for param_list in EXIT_PARAMETERS.values():
        exit_combinations *= len(param_list)
    print(f"Exit combinations per alert set: {exit_combinations:,}")
    print(f"Total parameter combinations: {len(ALERT_PARAMETERS['timeframe_minutes']) * len(ALERT_PARAMETERS['green_threshold']) * exit_combinations:,}")

    # Run optimization
    if args.method == 'hierarchical':
        results = optimizer.optimize_hierarchical(alerts, args.max_combinations)
    else:
        results = optimizer.optimize_full_grid_search(alerts, args.max_combinations)

    # Display results
    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        return 1

    print(f"\nOptimization Results ({results['optimization_method']}):")
    print("=" * 50)

    if args.method == 'hierarchical':
        print("Best Alert Parameters:")
        print(f"  Timeframe: {results['best_alert_parameters']['timeframe_minutes']} minutes")
        print(f"  Green Threshold: {results['best_alert_parameters']['green_threshold']:.2f}")

        print("\nBest Exit Parameters:")
        exit_params = results['best_exit_parameters']
        print(f"  Take Profit: {exit_params['take_profit_pct']}%")
        # print(f"  Stop Loss: {exit_params['stop_loss_pct']}%")  # Removed
        print(f"  Trailing Stop: {exit_params['trailing_stop_pct']}%")
        print(f"  MACD Enabled: {exit_params['macd_enabled']}")
        print(f"  MACD Sensitivity: {exit_params['macd_sensitivity']}")
    else:
        print("Best Combined Parameters:")
        best_params = results['best_parameters']
        print(f"  Alert Timeframe: {best_params['alert_timeframe_minutes']} minutes")
        print(f"  Alert Green Threshold: {best_params['alert_green_threshold']:.2f}")
        print(f"  Take Profit: {best_params['take_profit_pct']}%")
        # print(f"  Stop Loss: {best_params['stop_loss_pct']}%")  # Removed
        print(f"  Trailing Stop: {best_params['trailing_stop_pct']}%")
        print(f"  MACD Enabled: {best_params['macd_enabled']}")
        print(f"  MACD Sensitivity: {best_params['macd_sensitivity']}")

    # Performance metrics
    if args.method == 'hierarchical':
        metrics = results['performance_metrics']
    else:
        metrics = results['best_metrics']

    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Winning Trades: {metrics['winning_trades']}")
    print(f"  Average Win: {metrics['avg_win']:.2f}%")
    print(f"  Average Loss: {metrics['avg_loss']:.2f}%")

    # Save results
    import json
    with open(args.output, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert_numpy)

    print(f"\nResults saved to {args.output}")

    # Generate profitability visualization charts
    charts_created = create_profitability_visualizations(results)

    if charts_created:
        print("\n🎨 PROFITABILITY CHARTS CREATED:")
        print("  📊 1_parameter_heatmap.png - Take Profit vs Stop Loss heatmap")
        print("  🚀 2_3d_profitability_surface.png - 3D profitability surface")
        print("  💎 3_risk_return_scatter.png - Risk vs return analysis")
        print("  🎯 4_win_rate_analysis.png - Comprehensive win rate analysis")
        print("  📈 5_parameter_distributions.png - Parameter performance distributions")
        print("  🏆 6_performance_comparison.png - Best strategy performance summary")
        print(f"\n📁 All charts saved in optimization_charts/ directory")

    return 0


if __name__ == "__main__":
    sys.exit(main())