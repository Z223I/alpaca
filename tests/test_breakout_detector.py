"""
Unit tests for ORB Breakout Detector

Tests the breakout detection algorithm and signal generation.
"""

import pytest
import pandas as pd
from datetime import datetime, time
from unittest.mock import Mock, patch

from atoms.alerts.breakout_detector import (
    BreakoutDetector, BreakoutSignal, BreakoutType
)
from atoms.indicators.orb_calculator import ORBCalculator, ORBLevel


class TestBreakoutSignal:
    """Test cases for BreakoutSignal data class."""
    
    def test_breakout_signal_creation_bullish(self):
        """Test bullish breakout signal creation."""
        signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.0,  # Will be calculated
            volume_ratio=2.0,
            timestamp=datetime.now()
        )
        
        assert signal.symbol == "AAPL"
        assert signal.breakout_type == BreakoutType.BULLISH_BREAKOUT
        assert signal.current_price == 151.00
        assert signal.orb_high == 150.00
        assert signal.orb_low == 148.00
        assert abs(signal.breakout_percentage - 0.67) < 0.01  # (151-150)/150 * 100
        assert signal.volume_ratio == 2.0
    
    def test_breakout_signal_creation_bearish(self):
        """Test bearish breakdown signal creation."""
        signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            current_price=147.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.0,  # Will be calculated
            volume_ratio=1.8,
            timestamp=datetime.now()
        )
        
        assert signal.breakout_type == BreakoutType.BEARISH_BREAKDOWN
        assert abs(signal.breakout_percentage - 0.68) < 0.01  # (148-147)/148 * 100
    
    def test_breakout_signal_no_breakout(self):
        """Test no breakout signal."""
        signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.NO_BREAKOUT,
            current_price=149.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.0,
            volume_ratio=1.0,
            timestamp=datetime.now()
        )
        
        assert signal.breakout_type == BreakoutType.NO_BREAKOUT
        assert signal.breakout_percentage == 0.0


class TestBreakoutDetector:
    """Test cases for BreakoutDetector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orb_calculator = Mock(spec=ORBCalculator)
        self.detector = BreakoutDetector(self.orb_calculator)
        
        # Create mock ORB level
        self.mock_orb_level = ORBLevel(
            symbol="AAPL",
            orb_high=150.00,
            orb_low=148.00,
            orb_range=2.00,
            calculation_time=datetime.now(),
            sample_count=15
        )
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = BreakoutDetector()
        assert detector.orb_calculator is not None
        assert len(detector.recent_signals) == 0
    
    def test_detect_bullish_breakout_success(self):
        """Test successful bullish breakout detection."""
        # Setup mock
        self.orb_calculator.get_orb_level.return_value = self.mock_orb_level
        
        # Test breakout above threshold
        with patch.object(self.detector, '_is_bullish_breakout', return_value=True):
            signal = self.detector.detect_breakout("AAPL", 151.00, 2.0)
        
        assert signal is not None
        assert signal.breakout_type == BreakoutType.BULLISH_BREAKOUT
        assert signal.current_price == 151.00
        assert signal.volume_ratio == 2.0
    
    def test_detect_bearish_breakdown_success(self):
        """Test successful bearish breakdown detection."""
        # Setup mock
        self.orb_calculator.get_orb_level.return_value = self.mock_orb_level
        
        # Test breakdown below threshold
        with patch.object(self.detector, '_is_bearish_breakdown', return_value=True):
            signal = self.detector.detect_breakout("AAPL", 147.00, 1.8)
        
        assert signal is not None
        assert signal.breakout_type == BreakoutType.BEARISH_BREAKDOWN
        assert signal.current_price == 147.00
    
    def test_detect_breakout_no_orb_level(self):
        """Test breakout detection without ORB level."""
        self.orb_calculator.get_orb_level.return_value = None
        
        signal = self.detector.detect_breakout("AAPL", 151.00, 2.0)
        assert signal is None
    
    def test_detect_breakout_no_breakout(self):
        """Test no breakout detection."""
        self.orb_calculator.get_orb_level.return_value = self.mock_orb_level
        
        # Price within range, no breakout
        with patch.object(self.detector, '_is_bullish_breakout', return_value=False), \
             patch.object(self.detector, '_is_bearish_breakdown', return_value=False):
            signal = self.detector.detect_breakout("AAPL", 149.00, 1.0)
        
        assert signal is None
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_bullish_breakout_success(self, mock_config):
        """Test bullish breakout detection logic."""
        mock_config.breakout_threshold = 0.002
        mock_config.volume_multiplier = 1.5
        
        # Price above threshold with sufficient volume
        result = self.detector._is_bullish_breakout(150.31, self.mock_orb_level, 2.0)
        assert result is True
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_bullish_breakout_price_too_low(self, mock_config):
        """Test bullish breakout with price below threshold."""
        mock_config.breakout_threshold = 0.002
        mock_config.volume_multiplier = 1.5
        
        # Price below threshold
        result = self.detector._is_bullish_breakout(150.20, self.mock_orb_level, 2.0)
        assert result is False
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_bullish_breakout_volume_too_low(self, mock_config):
        """Test bullish breakout with insufficient volume."""
        mock_config.breakout_threshold = 0.002
        mock_config.volume_multiplier = 1.5
        
        # Volume below threshold
        result = self.detector._is_bullish_breakout(150.31, self.mock_orb_level, 1.0)
        assert result is False
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_bearish_breakdown_success(self, mock_config):
        """Test bearish breakdown detection logic."""
        mock_config.breakout_threshold = 0.002
        mock_config.volume_multiplier = 1.5
        
        # Price below threshold with sufficient volume
        result = self.detector._is_bearish_breakdown(147.70, self.mock_orb_level, 2.0)
        assert result is True
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_bearish_breakdown_price_too_high(self, mock_config):
        """Test bearish breakdown with price above threshold."""
        mock_config.breakout_threshold = 0.002
        mock_config.volume_multiplier = 1.5
        
        # Price above threshold
        result = self.detector._is_bearish_breakdown(147.80, self.mock_orb_level, 2.0)
        assert result is False
    
    def test_calculate_technical_indicators_success(self):
        """Test technical indicators calculation."""
        # Create test data
        data = pd.DataFrame({
            'high': [150.0, 150.5, 151.0, 150.8, 151.2] * 3,
            'low': [149.0, 149.5, 150.0, 149.8, 150.2] * 3,
            'close': [149.5, 150.0, 150.5, 150.0, 150.8] * 3,
            'volume': [1000, 1100, 1200, 1050, 1300] * 3
        })
        
        indicators = self.detector.calculate_technical_indicators(data)
        
        assert 'ema_9' in indicators
        assert 'vwap' in indicators
        assert 'ema_deviation' in indicators
        assert 'vwap_deviation' in indicators
        assert all(isinstance(v, float) for v in indicators.values())
    
    def test_calculate_technical_indicators_insufficient_data(self):
        """Test technical indicators with insufficient data."""
        # Create minimal data (less than 9 periods)
        data = pd.DataFrame({
            'high': [150.0, 150.5],
            'low': [149.0, 149.5],
            'close': [149.5, 150.0],
            'volume': [1000, 1100]
        })
        
        indicators = self.detector.calculate_technical_indicators(data)
        assert indicators == {}
    
    def test_recent_signal_management(self):
        """Test recent signal storage and retrieval."""
        signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            timestamp=datetime.now()
        )
        
        # Test update and retrieval
        self.detector.update_recent_signal(signal)
        retrieved = self.detector.get_recent_signal("AAPL")
        
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.breakout_type == BreakoutType.BULLISH_BREAKOUT
    
    def test_get_recent_signal_not_found(self):
        """Test getting signal for non-existent symbol."""
        signal = self.detector.get_recent_signal("TSLA")
        assert signal is None
    
    def test_clear_recent_signals(self):
        """Test clearing recent signals."""
        signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            timestamp=datetime.now()
        )
        
        self.detector.update_recent_signal(signal)
        assert len(self.detector.recent_signals) == 1
        
        self.detector.clear_recent_signals()
        assert len(self.detector.recent_signals) == 0
    
    def test_get_all_recent_signals_sorted(self):
        """Test getting all recent signals sorted by timestamp."""
        # Create signals with different timestamps
        signal1 = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            timestamp=datetime(2025, 1, 1, 10, 0, 0)
        )
        
        signal2 = BreakoutSignal(
            symbol="TSLA",
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            current_price=147.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.68,
            volume_ratio=1.8,
            timestamp=datetime(2025, 1, 1, 11, 0, 0)
        )
        
        self.detector.update_recent_signal(signal1)
        self.detector.update_recent_signal(signal2)
        
        signals = self.detector.get_all_recent_signals()
        assert len(signals) == 2
        assert signals[0].timestamp > signals[1].timestamp  # Most recent first
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_within_alert_window_true(self, mock_config):
        """Test alert window check - within window."""
        mock_config.alert_window_start = "09:30"
        mock_config.alert_window_end = "15:30"
        
        # Test time within window
        test_time = datetime(2025, 1, 1, 12, 0, 0)
        result = self.detector.is_within_alert_window(test_time)
        assert result is True
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_within_alert_window_false(self, mock_config):
        """Test alert window check - outside window."""
        mock_config.alert_window_start = "09:30"
        mock_config.alert_window_end = "15:30"
        
        # Test time outside window
        test_time = datetime(2025, 1, 1, 16, 0, 0)
        result = self.detector.is_within_alert_window(test_time)
        assert result is False
    
    @patch('atoms.alerts.breakout_detector.config')
    def test_is_within_alert_window_boundary(self, mock_config):
        """Test alert window check - at boundaries."""
        mock_config.alert_window_start = "09:30"
        mock_config.alert_window_end = "15:30"
        
        # Test start boundary
        start_time = datetime(2025, 1, 1, 9, 30, 0)
        assert self.detector.is_within_alert_window(start_time) is True
        
        # Test end boundary
        end_time = datetime(2025, 1, 1, 15, 30, 0)
        assert self.detector.is_within_alert_window(end_time) is True