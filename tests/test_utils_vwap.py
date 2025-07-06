"""
Comprehensive tests for VWAP calculation utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from atoms.utils.calculate_vwap import (
    calculate_vwap,
    calculate_vwap_typical,
    calculate_vwap_hlc,
    calculate_vwap_ohlc
)


class TestVWAPCalculation:
    """Test suite for VWAP calculation functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample candlestick data for testing."""
        return pd.DataFrame({
            'open': [100, 102, 101, 103, 104, 102, 105, 107, 106, 108],
            'high': [102, 104, 103, 105, 106, 104, 107, 109, 108, 110],
            'low': [99, 101, 100, 102, 103, 101, 104, 106, 105, 107],
            'close': [101, 103, 102, 104, 105, 103, 106, 108, 107, 109],
            'volume': [1000, 1200, 800, 1500, 1100, 900, 1300, 1600, 1000, 1400]
        })
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal data for edge case testing."""
        return pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [98],
            'close': [101],
            'volume': [1000]
        })
    
    def test_calculate_vwap_basic(self, sample_data):
        """Test basic VWAP calculation using close prices."""
        success, vwap = calculate_vwap(sample_data, 'close')
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        
        # Manual calculation for first few points
        expected_first = sample_data['close'].iloc[0]  # First point = close price
        assert abs(vwap.iloc[0] - expected_first) < 0.01
        
        # Second point calculation
        price_vol_sum = (sample_data['close'].iloc[0] * sample_data['volume'].iloc[0] + 
                        sample_data['close'].iloc[1] * sample_data['volume'].iloc[1])
        volume_sum = sample_data['volume'].iloc[0] + sample_data['volume'].iloc[1]
        expected_second = price_vol_sum / volume_sum
        assert abs(vwap.iloc[1] - expected_second) < 0.01
    
    def test_calculate_vwap_open_prices(self, sample_data):
        """Test VWAP calculation using open prices."""
        success, vwap = calculate_vwap(sample_data, 'open')
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        
        # First point should equal open price
        assert abs(vwap.iloc[0] - sample_data['open'].iloc[0]) < 0.01
    
    def test_calculate_vwap_high_prices(self, sample_data):
        """Test VWAP calculation using high prices."""
        success, vwap = calculate_vwap(sample_data, 'high')
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        
        # First point should equal high price
        assert abs(vwap.iloc[0] - sample_data['high'].iloc[0]) < 0.01
    
    def test_calculate_vwap_low_prices(self, sample_data):
        """Test VWAP calculation using low prices."""
        success, vwap = calculate_vwap(sample_data, 'low')
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        
        # First point should equal low price
        assert abs(vwap.iloc[0] - sample_data['low'].iloc[0]) < 0.01
    
    def test_calculate_vwap_typical(self, sample_data):
        """Test VWAP calculation using typical price (HLC/3)."""
        success, vwap = calculate_vwap_typical(sample_data)
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        
        # Manual calculation for first point
        typical_price = (sample_data['high'].iloc[0] + sample_data['low'].iloc[0] + 
                        sample_data['close'].iloc[0]) / 3
        assert abs(vwap.iloc[0] - typical_price) < 0.01
    
    def test_calculate_vwap_hlc(self, sample_data):
        """Test VWAP calculation using HLC method (should be same as typical)."""
        success, vwap = calculate_vwap_hlc(sample_data)
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        
        # Should be identical to typical price method
        success_typical, vwap_typical = calculate_vwap_typical(sample_data)
        assert success_typical is True
        pd.testing.assert_series_equal(vwap, vwap_typical)
    
    def test_calculate_vwap_ohlc(self, sample_data):
        """Test VWAP calculation using OHLC average price."""
        success, vwap = calculate_vwap_ohlc(sample_data)
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        
        # Manual calculation for first point
        ohlc_price = (sample_data['open'].iloc[0] + sample_data['high'].iloc[0] + 
                     sample_data['low'].iloc[0] + sample_data['close'].iloc[0]) / 4
        assert abs(vwap.iloc[0] - ohlc_price) < 0.01
    
    def test_calculate_vwap_minimal_data(self, minimal_data):
        """Test VWAP calculation with minimal data."""
        success, vwap = calculate_vwap(minimal_data, 'close')
        
        assert success is True
        assert len(vwap) == 1
        assert vwap.iloc[0] == minimal_data['close'].iloc[0]
    
    def test_calculate_vwap_empty_dataframe(self):
        """Test VWAP calculation with empty DataFrame."""
        empty_df = pd.DataFrame()
        success, vwap = calculate_vwap(empty_df, 'close')
        
        assert success is False
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == 0
    
    def test_calculate_vwap_missing_price_column(self, sample_data):
        """Test VWAP calculation with missing price column."""
        success, vwap = calculate_vwap(sample_data, 'invalid_column')
        
        assert success is False
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_data)
        assert all(v == 0.0 for v in vwap)
    
    def test_calculate_vwap_missing_volume_column(self, sample_data):
        """Test VWAP calculation with missing volume column."""
        df_no_volume = sample_data.drop('volume', axis=1)
        success, vwap = calculate_vwap(df_no_volume, 'close')
        
        assert success is False
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(df_no_volume)
        assert all(v == 0.0 for v in vwap)
    
    def test_calculate_vwap_zero_volume(self, sample_data):
        """Test VWAP calculation with zero volume values."""
        zero_volume_data = sample_data.copy()
        zero_volume_data['volume'] = 0
        
        success, vwap = calculate_vwap(zero_volume_data, 'close')
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        # Should return zeros due to division by zero protection
        assert all(v == 0.0 for v in vwap)
    
    def test_calculate_vwap_mixed_zero_volume(self, sample_data):
        """Test VWAP calculation with mixed zero and non-zero volume."""
        mixed_volume_data = sample_data.copy()
        mixed_volume_data.loc[0, 'volume'] = 0
        
        success, vwap = calculate_vwap(mixed_volume_data, 'close')
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        # First value should be 0 due to zero volume
        assert vwap.iloc[0] == 0.0
        # Later values should be calculated normally
        assert vwap.iloc[1] > 0.0
    
    def test_calculate_vwap_typical_missing_columns(self, sample_data):
        """Test typical VWAP calculation with missing required columns."""
        # Missing high column
        df_no_high = sample_data.drop('high', axis=1)
        success, vwap = calculate_vwap_typical(df_no_high)
        
        assert success is False
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(df_no_high)
        assert all(v == 0.0 for v in vwap)
    
    def test_calculate_vwap_ohlc_missing_columns(self, sample_data):
        """Test OHLC VWAP calculation with missing required columns."""
        # Missing open column
        df_no_open = sample_data.drop('open', axis=1)
        success, vwap = calculate_vwap_ohlc(df_no_open)
        
        assert success is False
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(df_no_open)
        assert all(v == 0.0 for v in vwap)
    
    def test_calculate_vwap_nan_values(self, sample_data):
        """Test VWAP calculation with NaN values."""
        nan_data = sample_data.copy()
        nan_data.loc[2, 'close'] = np.nan
        
        success, vwap = calculate_vwap(nan_data, 'close')
        
        assert success is True
        assert isinstance(vwap, pd.Series)
        # Should handle NaN values gracefully
        assert not np.isnan(vwap.iloc[0])
        assert not np.isnan(vwap.iloc[1])
    
    def test_calculate_vwap_cumulative_nature(self, sample_data):
        """Test that VWAP is cumulative (values should generally increase with price trend)."""
        success, vwap = calculate_vwap(sample_data, 'close')
        
        assert success is True
        
        # VWAP should be cumulative - each point incorporates all previous data
        # Manual verification of cumulative calculation
        cumulative_pv = (sample_data['close'] * sample_data['volume']).cumsum()
        cumulative_vol = sample_data['volume'].cumsum()
        expected_vwap = cumulative_pv / cumulative_vol
        
        pd.testing.assert_series_equal(vwap, expected_vwap, check_names=False)
    
    def test_calculate_vwap_large_volume_impact(self, sample_data):
        """Test that high volume periods have more impact on VWAP."""
        # Create data with one very high volume bar
        high_volume_data = sample_data.copy()
        high_volume_data.loc[5, 'volume'] = 10000  # Much higher than others
        high_volume_data.loc[5, 'close'] = 120  # High price
        
        success, vwap = calculate_vwap(high_volume_data, 'close')
        
        assert success is True
        
        # VWAP after high volume bar should be pulled toward that price
        # Compare with normal volume calculation
        success_normal, vwap_normal = calculate_vwap(sample_data, 'close')
        
        # VWAP should be different after the high volume bar
        assert vwap.iloc[6] != vwap_normal.iloc[6]
    
    def test_calculate_vwap_price_volume_consistency(self, sample_data):
        """Test consistency between different price methods."""
        # All methods should produce valid results
        success_close, vwap_close = calculate_vwap(sample_data, 'close')
        success_typical, vwap_typical = calculate_vwap_typical(sample_data)
        success_ohlc, vwap_ohlc = calculate_vwap_ohlc(sample_data)
        
        assert all([success_close, success_typical, success_ohlc])
        
        # All should have same length
        assert len(vwap_close) == len(vwap_typical) == len(vwap_ohlc)
        
        # Values should be different (using different price bases)
        assert not vwap_close.equals(vwap_typical)
        assert not vwap_close.equals(vwap_ohlc)
        assert not vwap_typical.equals(vwap_ohlc)
    
    def test_calculate_vwap_index_preservation(self, sample_data):
        """Test that VWAP preserves original DataFrame index."""
        # Set custom index
        sample_data.index = pd.date_range('2024-01-01', periods=len(sample_data), freq='H')
        
        success, vwap = calculate_vwap(sample_data, 'close')
        
        assert success is True
        pd.testing.assert_index_equal(vwap.index, sample_data.index)
    
    def test_calculate_vwap_data_types(self, sample_data):
        """Test VWAP calculation with different data types."""
        # Convert to different numeric types
        float_data = sample_data.astype(float)
        success_float, vwap_float = calculate_vwap(float_data, 'close')
        
        int_data = sample_data.astype(int)
        success_int, vwap_int = calculate_vwap(int_data, 'close')
        
        assert success_float is True
        assert success_int is True
        
        # Results should be very close (within floating point precision)
        np.testing.assert_array_almost_equal(vwap_float.values, vwap_int.values, decimal=10)
    
    def test_calculate_vwap_edge_case_single_volume(self):
        """Test VWAP with single volume unit."""
        single_volume_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1, 1, 1]
        })
        
        success, vwap = calculate_vwap(single_volume_data, 'close')
        
        assert success is True
        # With equal volume, VWAP should equal simple average
        expected_vwap = single_volume_data['close'].expanding().mean()
        pd.testing.assert_series_equal(vwap, expected_vwap, check_names=False)
    
    def test_calculate_vwap_performance_large_dataset(self):
        """Test VWAP calculation performance with large dataset."""
        # Create large dataset (1000 rows)
        large_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(100, 200, 1000),
            'low': np.random.uniform(100, 200, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        import time
        start_time = time.time()
        success, vwap = calculate_vwap(large_data, 'close')
        end_time = time.time()
        
        assert success is True
        assert len(vwap) == 1000
        # Should complete in reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0