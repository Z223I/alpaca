"""
Comprehensive tests for EMA calculation utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from atoms.utils.calculate_ema import (
    calculate_ema,
    calculate_ema_manual
)


class TestEMACalculation:
    """Test suite for EMA calculation functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample candlestick data for testing."""
        return pd.DataFrame({
            'open': [100, 102, 101, 103, 104, 102, 105, 107, 106, 108, 110, 109, 111, 112],
            'high': [102, 104, 103, 105, 106, 104, 107, 109, 108, 110, 112, 111, 113, 114],
            'low': [99, 101, 100, 102, 103, 101, 104, 106, 105, 107, 109, 108, 110, 111],
            'close': [101, 103, 102, 104, 105, 103, 106, 108, 107, 109, 111, 110, 112, 113],
            'volume': [1000, 1200, 800, 1500, 1100, 900, 1300, 1600, 1000, 1400, 1800, 1200, 1500, 1700]
        })
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal data for edge case testing."""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
        })
    
    def test_calculate_ema_basic_9_period(self, sample_data):
        """Test basic EMA calculation with 9-period using close prices."""
        success, ema = calculate_ema(sample_data, 'close', 9)
        
        assert success is True
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)
        
        # Check that EMA starts calculating from the 9th period
        # Values before period 9 should be non-zero (pandas EMA implementation)
        assert all(ema >= 0)
        
        # EMA should be close to actual close prices
        assert abs(ema.iloc[-1] - sample_data['close'].iloc[-1]) < 10
    
    def test_calculate_ema_manual_9_period(self, sample_data):
        """Test manual EMA calculation with 9-period using close prices."""
        success, ema = calculate_ema_manual(sample_data, 'close', 9)
        
        assert success is True
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)
        
        # Check that first 8 values are zero (manual implementation)
        assert all(ema.iloc[:8] == 0.0)
        
        # 9th value should be SMA of first 9 values
        expected_sma = sample_data['close'].iloc[:9].mean()
        assert abs(ema.iloc[8] - expected_sma) < 0.01
    
    def test_calculate_ema_vs_manual_comparison(self, sample_data):
        """Test that pandas and manual EMA calculations are consistent after initial period."""
        success_pandas, ema_pandas = calculate_ema(sample_data, 'close', 9)
        success_manual, ema_manual = calculate_ema_manual(sample_data, 'close', 9)
        
        assert success_pandas is True
        assert success_manual is True
        
        # Both should have the same length
        assert len(ema_pandas) == len(ema_manual)
        
        # Values from period 9 onward should be very close
        # (pandas uses slightly different initialization method)
        for i in range(9, len(sample_data)):
            # Allow for small differences due to initialization methods
            assert abs(ema_pandas.iloc[i] - ema_manual.iloc[i]) < 1.0
    
    def test_calculate_ema_different_price_columns(self, sample_data):
        """Test EMA calculation using different price columns."""
        success_open, ema_open = calculate_ema(sample_data, 'open', 9)
        success_high, ema_high = calculate_ema(sample_data, 'high', 9)
        success_low, ema_low = calculate_ema(sample_data, 'low', 9)
        success_close, ema_close = calculate_ema(sample_data, 'close', 9)
        
        assert all([success_open, success_high, success_low, success_close])
        
        # All should have same length
        assert len(ema_open) == len(ema_high) == len(ema_low) == len(ema_close)
        
        # Values should be different (based on different price inputs)
        assert not ema_open.equals(ema_high)
        assert not ema_high.equals(ema_low)
        assert not ema_low.equals(ema_close)
    
    def test_calculate_ema_different_periods(self, sample_data):
        """Test EMA calculation with different periods."""
        success_5, ema_5 = calculate_ema(sample_data, 'close', 5)
        success_9, ema_9 = calculate_ema(sample_data, 'close', 9)
        success_21, ema_21 = calculate_ema(sample_data, 'close', 21)
        
        assert success_5 is True
        assert success_9 is True
        assert success_21 is False  # Not enough data for 21-period
        
        # Shorter periods should be more reactive (higher final values in uptrend)
        assert ema_5.iloc[-1] > ema_9.iloc[-1]
    
    def test_calculate_ema_insufficient_data(self):
        """Test EMA calculation with insufficient data."""
        short_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],  # Only 5 periods
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        success, ema = calculate_ema(short_data, 'close', 9)
        
        assert success is False
        assert isinstance(ema, pd.Series)
        assert len(ema) == 5
        assert all(ema == 0.0)
    
    def test_calculate_ema_manual_insufficient_data(self):
        """Test manual EMA calculation with insufficient data."""
        short_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],  # Only 5 periods
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        success, ema = calculate_ema_manual(short_data, 'close', 9)
        
        assert success is False
        assert isinstance(ema, pd.Series)
        assert len(ema) == 5
        assert all(ema == 0.0)
    
    def test_calculate_ema_empty_dataframe(self):
        """Test EMA calculation with empty DataFrame."""
        empty_df = pd.DataFrame()
        success, ema = calculate_ema(empty_df, 'close', 9)
        
        assert success is False
        assert isinstance(ema, pd.Series)
        assert len(ema) == 0
    
    def test_calculate_ema_invalid_column(self, sample_data):
        """Test EMA calculation with invalid price column."""
        success, ema = calculate_ema(sample_data, 'invalid_column', 9)
        
        assert success is False
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)
        assert all(ema == 0.0)
    
    def test_calculate_ema_manual_invalid_column(self, sample_data):
        """Test manual EMA calculation with invalid price column."""
        success, ema = calculate_ema_manual(sample_data, 'invalid_column', 9)
        
        assert success is False
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)
        assert all(ema == 0.0)
    
    def test_calculate_ema_minimal_period(self, minimal_data):
        """Test EMA calculation with exactly enough data."""
        success, ema = calculate_ema(minimal_data, 'close', 9)
        
        assert success is True
        assert len(ema) == 9
        assert all(ema > 0)
    
    def test_calculate_ema_manual_minimal_period(self, minimal_data):
        """Test manual EMA calculation with exactly enough data."""
        success, ema = calculate_ema_manual(minimal_data, 'close', 9)
        
        assert success is True
        assert len(ema) == 9
        
        # First 8 values should be zero
        assert all(ema.iloc[:8] == 0.0)
        # 9th value should be SMA
        assert ema.iloc[8] > 0.0
    
    def test_calculate_ema_smoothing_factor(self, sample_data):
        """Test that EMA uses correct smoothing factor (alpha)."""
        period = 9
        success, ema = calculate_ema_manual(sample_data, 'close', period)
        
        assert success is True
        
        # Calculate expected alpha
        expected_alpha = 2 / (period + 1)  # 2 / 10 = 0.2
        
        # Manual calculation for 10th period
        sma = sample_data['close'].iloc[:9].mean()
        expected_ema_10 = expected_alpha * sample_data['close'].iloc[9] + (1 - expected_alpha) * sma
        
        # Compare with calculated value (10th period is index 9)
        assert abs(ema.iloc[9] - expected_ema_10) < 0.01
    
    def test_calculate_ema_trend_following(self, sample_data):
        """Test that EMA follows price trends appropriately."""
        success, ema = calculate_ema(sample_data, 'close', 9)
        
        assert success is True
        
        # In an uptrend, EMA should generally increase
        # Check last 5 periods for trend
        last_5_ema = ema.iloc[-5:].values
        last_5_close = sample_data['close'].iloc[-5:].values
        
        # EMA should lag behind price but follow the trend
        ema_trend = last_5_ema[-1] - last_5_ema[0]
        price_trend = last_5_close[-1] - last_5_close[0]
        
        # EMA and price should move in same direction
        assert (ema_trend > 0) == (price_trend > 0)
    
    def test_calculate_ema_with_nan_values(self, sample_data):
        """Test EMA calculation with NaN values in price data."""
        nan_data = sample_data.copy()
        nan_data.loc[5, 'close'] = np.nan
        
        success, ema = calculate_ema(nan_data, 'close', 9)
        
        # pandas ewm should handle NaN values
        assert success is True
        assert isinstance(ema, pd.Series)
    
    def test_calculate_ema_manual_alpha_calculation(self):
        """Test that manual EMA calculation uses correct alpha formula."""
        test_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        })
        
        period = 5
        success, ema = calculate_ema_manual(test_data, 'close', period)
        
        assert success is True
        
        # Verify alpha calculation
        expected_alpha = 2 / (period + 1)  # 2 / 6 = 0.333...
        
        # Manual calculation for 6th period (index 5)
        sma = test_data['close'].iloc[:5].mean()  # 102.0
        expected_6th = expected_alpha * test_data['close'].iloc[5] + (1 - expected_alpha) * sma
        
        assert abs(ema.iloc[5] - expected_6th) < 0.01
    
    def test_calculate_ema_index_preservation(self, sample_data):
        """Test that EMA preserves original DataFrame index."""
        # Set custom index
        sample_data.index = pd.date_range('2024-01-01', periods=len(sample_data), freq='H')
        
        success, ema = calculate_ema(sample_data, 'close', 9)
        
        assert success is True
        pd.testing.assert_index_equal(ema.index, sample_data.index)
    
    def test_calculate_ema_data_types(self, sample_data):
        """Test EMA calculation with different data types."""
        # Convert to different numeric types
        float_data = sample_data.astype(float)
        success_float, ema_float = calculate_ema(float_data, 'close', 9)
        
        int_data = sample_data.astype(int)
        success_int, ema_int = calculate_ema(int_data, 'close', 9)
        
        assert success_float is True
        assert success_int is True
        
        # Results should be very close
        np.testing.assert_array_almost_equal(ema_float.values, ema_int.values, decimal=6)
    
    def test_calculate_ema_period_edge_cases(self, sample_data):
        """Test EMA calculation with edge case periods."""
        # Period of 1 (should behave like price itself)
        success_1, ema_1 = calculate_ema(sample_data, 'close', 1)
        assert success_1 is True
        # With period 1, alpha = 2/2 = 1.0, so EMA should equal prices
        np.testing.assert_array_almost_equal(ema_1.values, sample_data['close'].values, decimal=6)
        
        # Period of 2
        success_2, ema_2 = calculate_ema(sample_data, 'close', 2)
        assert success_2 is True
        assert len(ema_2) == len(sample_data)
    
    def test_calculate_ema_manual_period_edge_cases(self, sample_data):
        """Test manual EMA calculation with edge case periods."""
        # Period of 1
        success_1, ema_1 = calculate_ema_manual(sample_data, 'close', 1)
        assert success_1 is True
        # First value should be zero, second should be first close price
        assert ema_1.iloc[0] == sample_data['close'].iloc[0]
        
        # Period of 2
        success_2, ema_2 = calculate_ema_manual(sample_data, 'close', 2)
        assert success_2 is True
        assert ema_2.iloc[0] == 0.0  # First value is zero
        assert ema_2.iloc[1] > 0.0   # Second value is SMA of first 2
    
    def test_calculate_ema_performance_large_dataset(self):
        """Test EMA calculation performance with large dataset."""
        # Create large dataset (1000 rows)
        large_data = pd.DataFrame({
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        import time
        start_time = time.time()
        success, ema = calculate_ema(large_data, 'close', 20)
        end_time = time.time()
        
        assert success is True
        assert len(ema) == 1000
        # Should complete in reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
    
    def test_calculate_ema_consistency_over_time(self):
        """Test that EMA calculation is consistent with repeated calls."""
        test_data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126]
        })
        
        # Calculate EMA multiple times
        success1, ema1 = calculate_ema(test_data, 'close', 9)
        success2, ema2 = calculate_ema(test_data, 'close', 9)
        success3, ema3 = calculate_ema(test_data, 'close', 9)
        
        assert all([success1, success2, success3])
        
        # Results should be identical
        pd.testing.assert_series_equal(ema1, ema2)
        pd.testing.assert_series_equal(ema2, ema3)
    
    def test_calculate_ema_manual_step_by_step_validation(self):
        """Test manual EMA calculation with step-by-step validation."""
        test_data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110, 112, 114, 116]
        })
        
        period = 5
        success, ema = calculate_ema_manual(test_data, 'close', period)
        
        assert success is True
        
        # Validate step by step
        alpha = 2 / (period + 1)  # 2/6 = 0.333...
        
        # First value (index 4) should be SMA of first 5
        expected_sma = test_data['close'].iloc[:5].mean()  # (100+102+104+106+108)/5 = 104
        assert abs(ema.iloc[4] - expected_sma) < 0.01
        
        # Second EMA value (index 5)
        expected_ema_5 = alpha * test_data['close'].iloc[5] + (1 - alpha) * expected_sma
        assert abs(ema.iloc[5] - expected_ema_5) < 0.01
        
        # Third EMA value (index 6)
        expected_ema_6 = alpha * test_data['close'].iloc[6] + (1 - alpha) * expected_ema_5
        assert abs(ema.iloc[6] - expected_ema_6) < 0.01