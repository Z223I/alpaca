"""
Comprehensive tests for vector angle calculation utility.
"""

import pytest
import pandas as pd
import numpy as np
import math

from atoms.utils.calculate_vector_angle import calculate_vector_angle


class TestVectorAngleCalculation:
    """Test suite for vector angle calculation function."""
    
    @pytest.fixture
    def upward_trend_data(self):
        """Create data with clear upward trend."""
        return pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128],
            'high': [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129],
            'low': [99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127],
            'volume': [1000] * 15
        })
    
    @pytest.fixture
    def downward_trend_data(self):
        """Create data with clear downward trend."""
        return pd.DataFrame({
            'close': [128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100],
            'high': [129, 127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101],
            'low': [127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99],
            'volume': [1000] * 15
        })
    
    @pytest.fixture
    def flat_trend_data(self):
        """Create data with flat trend (no direction)."""
        return pd.DataFrame({
            'close': [100] * 15,
            'high': [101] * 15,
            'low': [99] * 15,
            'volume': [1000] * 15
        })
    
    @pytest.fixture
    def volatile_data(self):
        """Create data with high volatility but no clear trend."""
        return pd.DataFrame({
            'close': [100, 105, 98, 103, 96, 102, 99, 104, 97, 101, 100, 103, 98, 102, 100],
            'high': [106, 107, 104, 105, 103, 104, 105, 106, 103, 103, 102, 105, 104, 104, 102],
            'low': [95, 96, 92, 96, 90, 96, 93, 98, 91, 95, 94, 97, 92, 96, 94],
            'volume': [1000] * 15
        })
    
    def test_calculate_vector_angle_upward_trend(self, upward_trend_data):
        """Test vector angle calculation with clear upward trend."""
        angle = calculate_vector_angle(upward_trend_data, 'close', 15)
        
        # Should return positive angle for upward trend
        assert angle > 0
        assert isinstance(angle, float)
        
        # For linear upward trend with slope 2 per period, angle should be specific
        # slope = 2, angle = arctan(2) = ~63.43 degrees
        expected_angle = math.degrees(math.atan(2))
        assert abs(angle - expected_angle) < 0.01
    
    def test_calculate_vector_angle_downward_trend(self, downward_trend_data):
        """Test vector angle calculation with clear downward trend."""
        angle = calculate_vector_angle(downward_trend_data, 'close', 15)
        
        # Should return negative angle for downward trend
        assert angle < 0
        assert isinstance(angle, float)
        
        # For linear downward trend with slope -2 per period
        expected_angle = math.degrees(math.atan(-2))
        assert abs(angle - expected_angle) < 0.01
    
    def test_calculate_vector_angle_flat_trend(self, flat_trend_data):
        """Test vector angle calculation with flat trend."""
        angle = calculate_vector_angle(flat_trend_data, 'close', 15)
        
        # Should return ~0 degrees for flat trend
        assert abs(angle) < 0.01
        assert isinstance(angle, float)
    
    def test_calculate_vector_angle_different_price_columns(self, upward_trend_data):
        """Test vector angle calculation using different price columns."""
        angle_close = calculate_vector_angle(upward_trend_data, 'close', 15)
        angle_high = calculate_vector_angle(upward_trend_data, 'high', 15)
        angle_low = calculate_vector_angle(upward_trend_data, 'low', 15)
        
        # All should be positive (upward trend) but slightly different
        assert angle_close > 0
        assert angle_high > 0
        assert angle_low > 0
        
        # Should be very close since data has consistent trend
        assert abs(angle_close - angle_high) < 1.0
        assert abs(angle_close - angle_low) < 1.0
    
    def test_calculate_vector_angle_different_num_candles(self, upward_trend_data):
        """Test vector angle calculation with different number of candles."""
        angle_5 = calculate_vector_angle(upward_trend_data, 'close', 5)
        angle_10 = calculate_vector_angle(upward_trend_data, 'close', 10)
        angle_15 = calculate_vector_angle(upward_trend_data, 'close', 15)
        
        # All should be positive and very similar for linear trend
        assert angle_5 > 0
        assert angle_10 > 0
        assert angle_15 > 0
        
        # Should be very close for linear data
        assert abs(angle_5 - angle_10) < 0.1
        assert abs(angle_10 - angle_15) < 0.1
    
    def test_calculate_vector_angle_empty_dataframe(self):
        """Test vector angle calculation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            calculate_vector_angle(empty_df, 'close', 15)
    
    def test_calculate_vector_angle_invalid_column(self, upward_trend_data):
        """Test vector angle calculation with invalid price column."""
        with pytest.raises(ValueError, match="Column 'invalid_column' not found"):
            calculate_vector_angle(upward_trend_data, 'invalid_column', 15)
    
    def test_calculate_vector_angle_insufficient_data(self):
        """Test vector angle calculation with insufficient data."""
        short_data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108],  # Only 5 candles
            'volume': [1000] * 5
        })
        
        with pytest.raises(ValueError, match="DataFrame has only 5 rows, need at least 15"):
            calculate_vector_angle(short_data, 'close', 15)
    
    def test_calculate_vector_angle_nan_values(self):
        """Test vector angle calculation with NaN values."""
        nan_data = pd.DataFrame({
            'close': [100, 102, np.nan, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]
        })
        
        with pytest.raises(ValueError, match="Column 'close' contains non-numeric data"):
            calculate_vector_angle(nan_data, 'close', 15)
    
    def test_calculate_vector_angle_non_numeric_data(self):
        """Test vector angle calculation with non-numeric data."""
        text_data = pd.DataFrame({
            'close': ['not', 'a', 'number', 'data'] + [100] * 11
        })
        
        with pytest.raises(ValueError, match="Column 'close' contains non-numeric data"):
            calculate_vector_angle(text_data, 'close', 15)
    
    def test_calculate_vector_angle_mixed_data_types(self):
        """Test vector angle calculation with mixed numeric data types."""
        mixed_data = pd.DataFrame({
            'close': [100, 102.5, 104, 106.75, 108, 110.25, 112, 114.5, 116, 118.75, 120, 122.25, 124, 126.5, 128]
        })
        
        angle = calculate_vector_angle(mixed_data, 'close', 15)
        
        # Should handle mixed int/float data fine
        assert isinstance(angle, float)
        assert angle > 0  # Upward trend
    
    def test_calculate_vector_angle_extreme_values(self):
        """Test vector angle calculation with extreme values."""
        extreme_data = pd.DataFrame({
            'close': [0.01, 1000000, 0.001, 999999, 0.02] + [100] * 10
        })
        
        angle = calculate_vector_angle(extreme_data, 'close', 15)
        
        # Should handle extreme values without crashing
        assert isinstance(angle, float)
        assert not math.isnan(angle)
        assert not math.isinf(angle)
    
    def test_calculate_vector_angle_steep_upward(self):
        """Test vector angle calculation with very steep upward trend."""
        steep_data = pd.DataFrame({
            'close': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
        })
        
        angle = calculate_vector_angle(steep_data, 'close', 15)
        
        # Should return high positive angle
        assert angle > 45  # Steep upward trend
        assert angle < 90  # But not vertical
    
    def test_calculate_vector_angle_steep_downward(self):
        """Test vector angle calculation with very steep downward trend."""
        steep_data = pd.DataFrame({
            'close': [240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
        })
        
        angle = calculate_vector_angle(steep_data, 'close', 15)
        
        # Should return high negative angle
        assert angle < -45  # Steep downward trend
        assert angle > -90  # But not vertical
    
    def test_calculate_vector_angle_volatile_no_trend(self, volatile_data):
        """Test vector angle calculation with volatile data but no clear trend."""
        angle = calculate_vector_angle(volatile_data, 'close', 15)
        
        # Should return small angle (close to 0) for no clear trend
        assert abs(angle) < 10  # Less than 10 degrees
        assert isinstance(angle, float)
    
    def test_calculate_vector_angle_mathematical_accuracy(self):
        """Test mathematical accuracy of angle calculation."""
        # Create data with known slope
        # y = mx + b, where m = 1 (45 degree angle)
        linear_data = pd.DataFrame({
            'close': [100 + i for i in range(15)]  # slope = 1
        })
        
        angle = calculate_vector_angle(linear_data, 'close', 15)
        
        # Should be exactly 45 degrees (arctan(1) = 45°)
        expected_angle = 45.0
        assert abs(angle - expected_angle) < 0.01
    
    def test_calculate_vector_angle_zero_slope(self):
        """Test vector angle calculation with zero slope."""
        zero_slope_data = pd.DataFrame({
            'close': [100.0] * 15  # Perfect horizontal line
        })
        
        angle = calculate_vector_angle(zero_slope_data, 'close', 15)
        
        # Should be exactly 0 degrees
        assert abs(angle) < 0.001
    
    def test_calculate_vector_angle_small_movements(self):
        """Test vector angle calculation with very small price movements."""
        small_movement_data = pd.DataFrame({
            'close': [100.0 + 0.01 * i for i in range(15)]  # Very small increments
        })
        
        angle = calculate_vector_angle(small_movement_data, 'close', 15)
        
        # Should be very small positive angle
        assert angle > 0
        assert angle < 1.0  # Less than 1 degree
    
    def test_calculate_vector_angle_single_period(self):
        """Test vector angle calculation with minimum period (1)."""
        test_data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]
        })
        
        # Single period should raise an error due to linear algebra constraints
        with pytest.raises(Exception):  # Could be LinAlgError or ValueError
            calculate_vector_angle(test_data, 'close', 1)
    
    def test_calculate_vector_angle_two_periods(self):
        """Test vector angle calculation with two periods."""
        test_data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]
        })
        
        angle = calculate_vector_angle(test_data, 'close', 2)
        
        # With 2 candles (slope = 2), angle = arctan(2)
        expected_angle = math.degrees(math.atan(2))
        assert abs(angle - expected_angle) < 0.01
    
    def test_calculate_vector_angle_consistency(self):
        """Test that vector angle calculation is consistent across multiple calls."""
        test_data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]
        })
        
        angle1 = calculate_vector_angle(test_data, 'close', 15)
        angle2 = calculate_vector_angle(test_data, 'close', 15)
        angle3 = calculate_vector_angle(test_data, 'close', 15)
        
        # All calls should return identical results
        assert angle1 == angle2 == angle3
    
    def test_calculate_vector_angle_large_dataset_performance(self):
        """Test vector angle calculation performance with large dataset."""
        # Create large dataset (1000 rows)
        large_data = pd.DataFrame({
            'close': [100 + 0.1 * i for i in range(1000)]
        })
        
        import time
        start_time = time.time()
        angle = calculate_vector_angle(large_data, 'close', 50)
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 0.1
        assert isinstance(angle, float)
        assert angle > 0  # Upward trend
    
    def test_calculate_vector_angle_real_market_scenario(self):
        """Test vector angle calculation with realistic market data."""
        # Simulate realistic market price movement
        market_data = pd.DataFrame({
            'close': [150.00, 150.25, 150.10, 150.45, 150.30, 150.60, 150.75, 150.50, 150.80, 
                     150.95, 150.70, 151.00, 151.20, 150.90, 151.30]
        })
        
        angle = calculate_vector_angle(market_data, 'close', 15)
        
        # Should be small positive angle for gradual uptrend
        assert angle > 0
        assert angle < 20  # Realistic angle for stock movement
        assert isinstance(angle, float)
    
    def test_calculate_vector_angle_error_propagation(self):
        """Test that errors are properly propagated with detailed messages."""
        # Test error message content
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError) as exc_info:
            calculate_vector_angle(empty_df, 'close', 15)
        assert "DataFrame is empty" in str(exc_info.value)
        
        # Test insufficient data error message
        short_data = pd.DataFrame({'close': [100, 101, 102]})
        
        with pytest.raises(ValueError) as exc_info:
            calculate_vector_angle(short_data, 'close', 15)
        assert "DataFrame has only 3 rows, need at least 15" in str(exc_info.value)
        
        # Test missing column error message
        valid_data = pd.DataFrame({'price': [100] * 15})
        
        with pytest.raises(ValueError) as exc_info:
            calculate_vector_angle(valid_data, 'close', 15)
        assert "Column 'close' not found in DataFrame" in str(exc_info.value)
    
    def test_calculate_vector_angle_boundary_slopes(self):
        """Test vector angle calculation with boundary slope values."""
        # Test with slope approaching infinity (vertical line)
        steep_data = pd.DataFrame({
            'close': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
        })
        
        angle = calculate_vector_angle(steep_data, 'close', 15)
        
        # Should approach 90 degrees but not exceed it
        assert angle > 80
        assert angle < 90
        
        # Test with slope approaching negative infinity
        steep_down_data = pd.DataFrame({
            'close': [1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
        })
        
        angle_down = calculate_vector_angle(steep_down_data, 'close', 15)
        
        # Should approach -90 degrees but not exceed it
        assert angle_down < -80
        assert angle_down > -90