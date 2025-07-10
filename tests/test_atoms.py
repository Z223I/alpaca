"""
Unit tests for atoms in the alert analysis system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atoms.analysis.filter_trading_hours import filter_trading_hours
from atoms.analysis.filter_alert_hours import filter_alert_hours
from atoms.analysis.align_timestamps import align_alerts_to_market_data
from atoms.analysis.validate_data import validate_market_data, validate_alert_data
from atoms.metrics.success_rate import calculate_success_rate, calculate_success_rate_by_group
from atoms.metrics.calculate_returns import calculate_simple_return, calculate_trade_returns
from atoms.simulation.trade_executor import simulate_trade


class TestFilterTradingHours:
    """Test trading hours filtering functionality."""
    
    def test_filter_trading_hours_basic(self):
        """Test basic trading hours filtering."""
        # Create test data with timestamps across trading and non-trading hours
        timestamps = pd.date_range(
            start='2025-01-15 08:00:00',
            end='2025-01-15 18:00:00',
            freq='h',
            tz='US/Eastern'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': ['AAPL'] * len(timestamps),
            'price': [150.0] * len(timestamps)
        })
        
        filtered = filter_trading_hours(df)
        
        # Should only include 9:30 AM to 4:00 PM
        assert len(filtered) > 0
        assert all(filtered['timestamp'].dt.time >= pd.Timestamp('09:30').time())
        assert all(filtered['timestamp'].dt.time <= pd.Timestamp('16:00').time())
    
    def test_filter_trading_hours_weekends(self):
        """Test filtering of weekend data."""
        # Create weekend data
        timestamps = pd.date_range(
            start='2025-01-18 10:00:00',  # Saturday
            end='2025-01-19 15:00:00',    # Sunday
            freq='h',
            tz='US/Eastern'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': ['AAPL'] * len(timestamps),
            'price': [150.0] * len(timestamps)
        })
        
        filtered = filter_trading_hours(df)
        
        # Should be empty (no weekend trading)
        assert len(filtered) == 0
    
    def test_filter_trading_hours_empty_df(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        filtered = filter_trading_hours(empty_df)
        assert len(filtered) == 0


class TestFilterAlertHours:
    """Test alert hours filtering functionality."""
    
    def test_filter_alert_hours_basic(self):
        """Test basic alert hours filtering."""
        timestamps = pd.date_range(
            start='2025-01-15 08:00:00',
            end='2025-01-15 18:00:00',
            freq='h',
            tz='US/Eastern'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': ['AAPL'] * len(timestamps),
            'priority': ['High'] * len(timestamps)
        })
        
        filtered = filter_alert_hours(df)
        
        # Should only include 9:30 AM to 3:30 PM
        assert len(filtered) > 0
        assert all(filtered['timestamp'].dt.time >= pd.Timestamp('09:30').time())
        assert all(filtered['timestamp'].dt.time <= pd.Timestamp('15:30').time())


class TestValidateData:
    """Test data validation functionality."""
    
    def test_validate_market_data_valid(self):
        """Test validation of valid market data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15', periods=5, freq='h'),
            'symbol': ['AAPL'] * 5,
            'open': [150.0, 151.0, 152.0, 153.0, 154.0],
            'high': [151.0, 152.0, 153.0, 154.0, 155.0],
            'low': [149.0, 150.0, 151.0, 152.0, 153.0],
            'close': [150.5, 151.5, 152.5, 153.5, 154.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = validate_market_data(df)
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_market_data_invalid_ohlc(self):
        """Test validation with invalid OHLC relationships."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15', periods=2, freq='h'),
            'symbol': ['AAPL'] * 2,
            'open': [150.0, 151.0],
            'high': [149.0, 150.0],  # High < Open (invalid)
            'low': [148.0, 149.0],
            'close': [149.5, 150.5],
            'volume': [1000, 1100]
        })
        
        result = validate_market_data(df)
        # OHLC validation creates warnings, not errors
        assert len(result['warnings']) > 0
    
    def test_validate_alert_data_valid(self):
        """Test validation of valid alert data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15', periods=3, freq='h'),
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'priority': ['High', 'Medium', 'High'],
            'current_price': [150.0, 2500.0, 350.0],
            'breakout_type': ['bullish', 'bearish', 'bullish'],
            'confidence_score': [0.8, 0.7, 0.9],
            'recommended_stop_loss': [142.5, 2375.0, 332.5],
            'recommended_take_profit': [165.0, 2750.0, 385.0]
        })
        
        result = validate_alert_data(df)
        # With all required columns, should be valid
        assert result['valid'] is True
        assert len(result['errors']) == 0


class TestSuccessRate:
    """Test success rate calculation functionality."""
    
    def test_calculate_success_rate_basic(self):
        """Test basic success rate calculation."""
        df = pd.DataFrame({
            'status': ['SUCCESS', 'SUCCESS', 'LOSS', 'SUCCESS', 'LOSS']
        })
        
        rate = calculate_success_rate(df)
        assert rate == 60.0  # 3/5 = 60%
    
    def test_calculate_success_rate_empty(self):
        """Test success rate with empty DataFrame."""
        empty_df = pd.DataFrame()
        rate = calculate_success_rate(empty_df)
        assert rate == 0.0
    
    def test_calculate_success_rate_by_group(self):
        """Test success rate calculation by group."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'MSFT'],
            'status': ['SUCCESS', 'LOSS', 'SUCCESS', 'SUCCESS', 'LOSS']
        })
        
        result = calculate_success_rate_by_group(df, 'symbol')
        
        assert len(result) == 3
        assert result[result['symbol'] == 'AAPL']['success_rate'].iloc[0] == 50.0
        assert result[result['symbol'] == 'GOOGL']['success_rate'].iloc[0] == 100.0
        assert result[result['symbol'] == 'MSFT']['success_rate'].iloc[0] == 0.0


class TestCalculateReturns:
    """Test return calculation functionality."""
    
    def test_calculate_simple_return_basic(self):
        """Test basic simple return calculation."""
        entry_price = 100.0
        exit_price = 110.0
        
        return_pct = calculate_simple_return(entry_price, exit_price)
        assert return_pct == 10.0  # 10% gain
    
    def test_calculate_simple_return_loss(self):
        """Test simple return calculation for loss."""
        entry_price = 100.0
        exit_price = 90.0
        
        return_pct = calculate_simple_return(entry_price, exit_price)
        assert return_pct == -10.0  # 10% loss
    
    def test_calculate_trade_returns(self):
        """Test trade returns calculation on DataFrame."""
        df = pd.DataFrame({
            'entry_price': [100.0, 200.0, 50.0],
            'exit_price': [110.0, 180.0, 55.0]
        })
        
        returns = calculate_trade_returns(df)
        
        expected = [10.0, -10.0, 10.0]  # 10%, -10%, 10%
        assert all(abs(returns - expected) < 0.01)
    
    def test_calculate_trade_returns_empty(self):
        """Test trade returns with empty DataFrame."""
        empty_df = pd.DataFrame()
        returns = calculate_trade_returns(empty_df)
        assert len(returns) == 0


class TestTradeExecutor:
    """Test trade execution simulation functionality."""
    
    def test_simulate_trade_basic(self):
        """Test basic trade simulation."""
        alert = {
            'symbol': 'AAPL',
            'timestamp': '2025-01-15 10:00:00',
            'current_price': 150.0,
            'breakout_type': 'bullish_breakout',
            'recommended_stop_loss': 142.5,
            'recommended_take_profit': 165.0
        }
        
        # Create market data that hits take profit
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15 10:05:00', periods=5, freq='5min'),
            'symbol': ['AAPL'] * 5,
            'open': [150.5, 155.0, 160.0, 165.0, 164.0],
            'high': [155.0, 160.0, 165.0, 167.0, 166.0],
            'low': [150.0, 154.0, 159.0, 164.0, 163.0],
            'close': [154.5, 159.5, 164.5, 166.5, 165.5]
        })
        
        result = simulate_trade(alert, market_data)
        
        assert result['status'] == 'SUCCESS'
        assert result['exit_reason'] == 'TAKE_PROFIT'
        assert result['return_pct'] == 10.0  # (165-150)/150 * 100
    
    def test_simulate_trade_stop_loss(self):
        """Test trade simulation hitting stop loss."""
        alert = {
            'symbol': 'AAPL',
            'timestamp': '2025-01-15 10:00:00',
            'current_price': 150.0,
            'breakout_type': 'bullish_breakout',
            'recommended_stop_loss': 142.5,
            'recommended_take_profit': 165.0
        }
        
        # Create market data that hits stop loss
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15 10:05:00', periods=3, freq='5min'),
            'symbol': ['AAPL'] * 3,
            'open': [149.0, 145.0, 140.0],
            'high': [149.5, 145.5, 140.5],
            'low': [148.0, 142.0, 139.0],  # Hits stop loss
            'close': [148.5, 142.5, 139.5]
        })
        
        result = simulate_trade(alert, market_data)
        
        assert result['status'] == 'STOPPED_OUT'
        assert result['exit_reason'] == 'STOP_LOSS'
        assert result['return_pct'] == -5.0  # (142.5-150)/150 * 100
    
    def test_simulate_trade_no_data(self):
        """Test trade simulation with no market data."""
        alert = {
            'symbol': 'AAPL',
            'timestamp': '2025-01-15 10:00:00',
            'current_price': 150.0,
            'breakout_type': 'bullish_breakout',
            'recommended_stop_loss': 142.5,
            'recommended_take_profit': 165.0
        }
        
        empty_data = pd.DataFrame()
        
        result = simulate_trade(alert, empty_data)
        
        assert result['status'] == 'FAILED'
        assert result['exit_reason'] == 'NO_DATA'
        assert result['return_pct'] == 0.0


class TestAlignTimestamps:
    """Test timestamp alignment functionality."""
    
    def test_align_alerts_with_market_data(self):
        """Test aligning alerts with market data."""
        alerts = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15 10:00:00', periods=2, freq='h'),
            'symbol': ['AAPL', 'GOOGL']
        })
        
        market_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-15 09:30:00', periods=10, freq='30min'),
            'symbol': ['AAPL'] * 5 + ['GOOGL'] * 5,
            'close': [150.0] * 10
        })
        
        result = align_alerts_to_market_data(alerts, market_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(alerts)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])