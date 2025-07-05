"""
Unit tests for ORB Calculator

Tests the Opening Range Breakout calculation logic.
"""

import pytest
import pandas as pd
from datetime import datetime, time
from unittest.mock import patch

from atoms.indicators.orb_calculator import ORBCalculator, ORBLevel


class TestORBLevel:
    """Test cases for ORB Level data class."""
    
    def test_orb_level_creation(self):
        """Test ORB level creation and calculations."""
        orb_level = ORBLevel(
            symbol="AAPL",
            orb_high=150.50,
            orb_low=149.00,
            orb_range=1.50,
            calculation_time=datetime.now(),
            sample_count=15
        )
        
        assert orb_level.symbol == "AAPL"
        assert orb_level.orb_high == 150.50
        assert orb_level.orb_low == 149.00
        assert orb_level.orb_range == 1.50
        assert orb_level.orb_midpoint == 149.75
        assert orb_level.sample_count == 15
    
    def test_orb_level_post_init(self):
        """Test ORB level post-initialization calculations."""
        orb_level = ORBLevel(
            symbol="AAPL",
            orb_high=150.00,
            orb_low=148.00,
            orb_range=0,  # Will be recalculated
            calculation_time=datetime.now(),
            sample_count=15
        )
        
        assert orb_level.orb_range == 2.00
        assert orb_level.orb_midpoint == 149.00
    
    def test_get_breakout_threshold(self):
        """Test breakout threshold calculation."""
        orb_level = ORBLevel(
            symbol="AAPL",
            orb_high=150.00,
            orb_low=148.00,
            orb_range=2.00,
            calculation_time=datetime.now(),
            sample_count=15
        )
        
        # Default threshold (0.2%)
        threshold = orb_level.get_breakout_threshold()
        expected = 150.00 * 1.002
        assert abs(threshold - expected) < 0.001
        
        # Custom threshold (0.5%)
        threshold = orb_level.get_breakout_threshold(0.005)
        expected = 150.00 * 1.005
        assert abs(threshold - expected) < 0.001
    
    def test_get_breakdown_threshold(self):
        """Test breakdown threshold calculation."""
        orb_level = ORBLevel(
            symbol="AAPL",
            orb_high=150.00,
            orb_low=148.00,
            orb_range=2.00,
            calculation_time=datetime.now(),
            sample_count=15
        )
        
        # Default threshold (0.2%)
        threshold = orb_level.get_breakdown_threshold()
        expected = 148.00 * 0.998
        assert abs(threshold - expected) < 0.001
        
        # Custom threshold (0.5%)
        threshold = orb_level.get_breakdown_threshold(0.005)
        expected = 148.00 * 0.995
        assert abs(threshold - expected) < 0.001


class TestORBCalculator:
    """Test cases for ORB Calculator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = ORBCalculator(orb_period_minutes=15)
        
        # Create test data
        timestamps = pd.date_range(
            start='2025-01-01 09:30:00',
            end='2025-01-01 16:00:00',
            freq='1min'
        )
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'high': [150.0 + i * 0.1 for i in range(len(timestamps))],
            'low': [149.0 + i * 0.1 for i in range(len(timestamps))],
            'close': [149.5 + i * 0.1 for i in range(len(timestamps))],
            'volume': [1000 + i * 10 for i in range(len(timestamps))]
        })
    
    def test_calculator_initialization(self):
        """Test ORB calculator initialization."""
        calculator = ORBCalculator(orb_period_minutes=30)
        
        assert calculator.orb_period_minutes == 30
        assert calculator.market_open == time(9, 30)
        assert len(calculator._orb_levels) == 0
    
    def test_calculate_orb_levels_success(self):
        """Test successful ORB level calculation."""
        orb_level = self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        assert orb_level is not None
        assert orb_level.symbol == "AAPL"
        assert orb_level.orb_high > orb_level.orb_low
        assert orb_level.sample_count > 0
        assert orb_level.orb_range > 0
    
    def test_calculate_orb_levels_empty_data(self):
        """Test ORB calculation with empty data."""
        empty_data = pd.DataFrame()
        orb_level = self.calculator.calculate_orb_levels("AAPL", empty_data)
        
        assert orb_level is None
    
    def test_calculate_orb_levels_no_timestamp(self):
        """Test ORB calculation without timestamp column."""
        data_no_timestamp = self.test_data.drop('timestamp', axis=1)
        orb_level = self.calculator.calculate_orb_levels("AAPL", data_no_timestamp)
        
        assert orb_level is None
    
    def test_calculate_orb_levels_insufficient_data(self):
        """Test ORB calculation with insufficient data."""
        # Create data with only 2 points
        small_data = self.test_data.head(2)
        orb_level = self.calculator.calculate_orb_levels("AAPL", small_data)
        
        assert orb_level is None
    
    def test_filter_opening_range(self):
        """Test opening range filtering."""
        orb_data = self.calculator._filter_opening_range(self.test_data)
        
        assert not orb_data.empty
        assert len(orb_data) <= 16  # 15 minutes + 1 for boundary
        
        # Check that all times are within opening range
        for _, row in orb_data.iterrows():
            time_val = row['time']
            assert time_val >= time(9, 30)
            assert time_val <= time(9, 45)
    
    def test_get_orb_level_cached(self):
        """Test getting cached ORB level."""
        # Calculate ORB level (should cache it)
        orb_level = self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        # Get from cache
        cached_level = self.calculator.get_orb_level("AAPL")
        
        assert cached_level is not None
        assert cached_level.symbol == orb_level.symbol
        assert cached_level.orb_high == orb_level.orb_high
        assert cached_level.orb_low == orb_level.orb_low
    
    def test_get_orb_level_not_cached(self):
        """Test getting ORB level that's not cached."""
        cached_level = self.calculator.get_orb_level("TSLA")
        
        assert cached_level is None
    
    def test_is_breakout_true(self):
        """Test breakout detection - positive case."""
        # Calculate ORB level
        orb_level = self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        # Test with price above breakout threshold
        breakout_price = orb_level.orb_high * 1.003  # 0.3% above
        
        is_breakout = self.calculator.is_breakout("AAPL", breakout_price)
        assert is_breakout == True
    
    def test_is_breakout_false(self):
        """Test breakout detection - negative case."""
        # Calculate ORB level
        orb_level = self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        # Test with price below breakout threshold
        no_breakout_price = orb_level.orb_high * 1.001  # 0.1% above (below threshold)
        
        is_breakout = self.calculator.is_breakout("AAPL", no_breakout_price)
        assert is_breakout == False
    
    def test_is_breakout_no_orb_level(self):
        """Test breakout detection without ORB level."""
        is_breakout = self.calculator.is_breakout("TSLA", 150.00)
        assert is_breakout is False
    
    def test_is_breakdown_true(self):
        """Test breakdown detection - positive case."""
        # Calculate ORB level
        orb_level = self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        # Test with price below breakdown threshold
        breakdown_price = orb_level.orb_low * 0.997  # 0.3% below
        
        is_breakdown = self.calculator.is_breakdown("AAPL", breakdown_price)
        assert is_breakdown == True
    
    def test_is_breakdown_false(self):
        """Test breakdown detection - negative case."""
        # Calculate ORB level
        orb_level = self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        # Test with price above breakdown threshold
        no_breakdown_price = orb_level.orb_low * 0.999  # 0.1% below (above threshold)
        
        is_breakdown = self.calculator.is_breakdown("AAPL", no_breakdown_price)
        assert is_breakdown == False
    
    def test_get_price_position(self):
        """Test price position calculation."""
        # Calculate ORB level
        orb_level = self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        # Test at ORB low (should be 0)
        position = self.calculator.get_price_position("AAPL", orb_level.orb_low)
        assert abs(position - 0.0) < 0.001
        
        # Test at ORB high (should be 1)
        position = self.calculator.get_price_position("AAPL", orb_level.orb_high)
        assert abs(position - 1.0) < 0.001
        
        # Test at midpoint (should be 0.5)
        position = self.calculator.get_price_position("AAPL", orb_level.orb_midpoint)
        assert abs(position - 0.5) < 0.001
    
    def test_get_price_position_no_orb_level(self):
        """Test price position without ORB level."""
        position = self.calculator.get_price_position("TSLA", 150.00)
        assert position is None
    
    def test_get_price_position_zero_range(self):
        """Test price position with zero ORB range."""
        # Create mock ORB level with zero range
        zero_range_level = ORBLevel(
            symbol="AAPL",
            orb_high=150.00,
            orb_low=150.00,
            orb_range=0.00,
            calculation_time=datetime.now(),
            sample_count=15
        )
        
        self.calculator._orb_levels["AAPL"] = zero_range_level
        
        position = self.calculator.get_price_position("AAPL", 150.00)
        assert position is None
    
    def test_clear_cache(self):
        """Test clearing ORB level cache."""
        # Calculate ORB level (should cache it)
        self.calculator.calculate_orb_levels("AAPL", self.test_data)
        
        assert len(self.calculator._orb_levels) == 1
        
        # Clear cache
        self.calculator.clear_cache()
        
        assert len(self.calculator._orb_levels) == 0
    
    def test_get_cached_symbols(self):
        """Test getting cached symbols."""
        # Initially empty
        symbols = self.calculator.get_cached_symbols()
        assert symbols == []
        
        # Calculate ORB levels for multiple symbols
        self.calculator.calculate_orb_levels("AAPL", self.test_data)
        self.calculator.calculate_orb_levels("TSLA", self.test_data)
        
        symbols = self.calculator.get_cached_symbols()
        assert len(symbols) == 2
        assert "AAPL" in symbols
        assert "TSLA" in symbols