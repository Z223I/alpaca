"""
Unit tests for ORB Alert Formatter

Tests the alert formatting, prioritization, and output generation.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch

from atoms.alerts.alert_formatter import (
    AlertFormatter, ORBAlert, AlertPriority
)
from atoms.alerts.breakout_detector import BreakoutSignal, BreakoutType
from atoms.alerts.confidence_scorer import ConfidenceComponents


class TestORBAlert:
    """Test cases for ORBAlert data class."""
    
    def test_orb_alert_creation_bullish(self):
        """Test ORB alert creation for bullish breakout."""
        alert = ORBAlert(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 1, 10, 30, 0),
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            orb_range=0.0,  # Will be calculated
            orb_midpoint=0.0,  # Will be calculated
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            confidence_score=0.85,
            priority=AlertPriority.HIGH,
            confidence_level="VERY_HIGH",
            recommended_stop_loss=0.0,  # Will be calculated
            recommended_take_profit=0.0,  # Will be calculated
            alert_message=""  # Will be generated
        )
        
        assert alert.symbol == "AAPL"
        assert alert.orb_range == 2.00  # 150 - 148
        assert alert.orb_midpoint == 149.00  # (150 + 148) / 2
        assert alert.recommended_stop_loss > 0
        assert alert.recommended_take_profit > 0
        assert len(alert.alert_message) > 0
        assert "↑" in alert.alert_message  # Bullish indicator
    
    def test_orb_alert_creation_bearish(self):
        """Test ORB alert creation for bearish breakdown."""
        alert = ORBAlert(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 1, 10, 30, 0),
            current_price=147.00,
            orb_high=150.00,
            orb_low=148.00,
            orb_range=0.0,
            orb_midpoint=0.0,
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            breakout_percentage=0.68,
            volume_ratio=1.8,
            confidence_score=0.75,
            priority=AlertPriority.MEDIUM,
            confidence_level="HIGH",
            recommended_stop_loss=0.0,
            recommended_take_profit=0.0,
            alert_message=""
        )
        
        assert alert.breakout_type == BreakoutType.BEARISH_BREAKDOWN
        assert alert.recommended_stop_loss > alert.current_price  # Stop above current for short
        assert alert.recommended_take_profit < alert.current_price  # Target below current for short
        assert "↓" in alert.alert_message  # Bearish indicator
    
    def test_orb_alert_risk_calculation_bullish(self):
        """Test risk level calculation for bullish breakout."""
        alert = ORBAlert(
            symbol="AAPL",
            timestamp=datetime.now(),
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            orb_range=2.0,
            orb_midpoint=149.0,
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            confidence_score=0.85,
            priority=AlertPriority.HIGH,
            confidence_level="VERY_HIGH",
            recommended_stop_loss=0.0,
            recommended_take_profit=0.0,
            alert_message=""
        )
        
        # Stop loss should be 7.5% below current price (config.stop_loss_percent)
        expected_stop = 151.00 * (1 - 7.5/100)  # 151.00 * 0.925
        assert abs(alert.recommended_stop_loss - expected_stop) < 0.01
        
        # Take profit should be 4% above current price (config.take_profit_percent)  
        expected_target = 151.00 * (1 + 4.0/100)  # 151.00 * 1.04
        assert abs(alert.recommended_take_profit - expected_target) < 0.01


class TestAlertFormatter:
    """Test cases for AlertFormatter."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.formatter = AlertFormatter()
        
        # Create test breakout signal
        self.test_signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            timestamp=datetime(2025, 1, 1, 10, 30, 0)
        )
        
        # Create test confidence
        self.test_confidence = ConfidenceComponents(
            pc1_score=0.8,
            pc2_score=0.7,
            pc3_score=0.6,
            total_score=0.78
        )
    
    
    def test_formatter_initialization(self):
        """Test formatter initialization."""
        formatter = AlertFormatter()
        assert len(formatter.alert_history) == 0
        assert formatter.daily_alert_count == 0
    
    def test_create_alert_success(self):
        """Test successful alert creation."""
        alert = self.formatter.create_alert(self.test_signal, self.test_confidence)
        
        assert isinstance(alert, ORBAlert)
        assert alert.symbol == "AAPL"
        assert alert.current_price == 151.00
        assert abs(alert.confidence_score - self.test_confidence.total_score) < 0.01
        assert len(self.formatter.alert_history) == 1
        assert self.formatter.daily_alert_count == 1
    
    def test_calculate_priority_high(self):
        """Test high priority calculation."""
        priority = self.formatter._calculate_priority(0.85, 2.0)
        assert priority == AlertPriority.HIGH
    
    def test_calculate_priority_medium_confidence(self):
        """Test medium priority from good confidence."""
        priority = self.formatter._calculate_priority(0.75, 1.0)
        assert priority == AlertPriority.MEDIUM
    
    def test_calculate_priority_medium_volume(self):
        """Test medium priority from high volume."""
        priority = self.formatter._calculate_priority(0.50, 2.5)
        assert priority == AlertPriority.MEDIUM
    
    def test_calculate_priority_low(self):
        """Test low priority calculation."""
        priority = self.formatter._calculate_priority(0.65, 1.0)
        assert priority == AlertPriority.LOW
    
    def test_calculate_priority_very_low(self):
        """Test very low priority calculation."""
        priority = self.formatter._calculate_priority(0.45, 1.0)
        assert priority == AlertPriority.VERY_LOW
    
    def test_get_confidence_level_mapping(self):
        """Test confidence level mapping."""
        assert self.formatter._get_confidence_level(0.90) == "VERY_HIGH"
        assert self.formatter._get_confidence_level(0.75) == "HIGH"
        assert self.formatter._get_confidence_level(0.65) == "MEDIUM"
        assert self.formatter._get_confidence_level(0.55) == "LOW"
        assert self.formatter._get_confidence_level(0.40) == "VERY_LOW"
    
    def test_format_console_output(self):
        """Test console output formatting."""
        alert = self.formatter.create_alert(self.test_signal, self.test_confidence)
        console_output = self.formatter.format_console_output(alert)
        
        assert "AAPL" in console_output
        assert "151.00" in console_output
        assert "↑" in console_output  # Bullish indicator
        assert "Volume:" in console_output
        assert "Confidence:" in console_output
        assert "Priority:" in console_output
    
    def test_format_json_output(self):
        """Test JSON output formatting."""
        alert = self.formatter.create_alert(self.test_signal, self.test_confidence)
        json_output = self.formatter.format_json_output(alert)
        
        # Parse JSON to verify structure
        data = json.loads(json_output)
        
        assert data['symbol'] == "AAPL"
        assert data['current_price'] == 151.00
        assert data['breakout_type'] == "bullish_breakout"
        assert data['priority'] == "MEDIUM"  # Based on test conditions
        assert 'timestamp' in data
        assert 'recommended_stop_loss' in data
        assert 'recommended_take_profit' in data
    
    
    def test_get_alerts_by_priority(self):
        """Test filtering alerts by priority."""
        # Create multiple alerts with different priorities
        # Create a high priority alert (high confidence + high volume)
        high_conf = ConfidenceComponents(0.9, 0.85, 0.8, 0.88)
        low_conf = ConfidenceComponents(0.4, 0.3, 0.2, 0.35)
        
        high_vol_signal = BreakoutSignal(
            symbol="AAPL", breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.0, orb_high=150.0, orb_low=148.0,
            breakout_percentage=1.0, volume_ratio=2.5, timestamp=datetime.now()
        )
        
        low_vol_signal = BreakoutSignal(
            symbol="TSLA", breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=150.1, orb_high=150.0, orb_low=148.0,
            breakout_percentage=0.1, volume_ratio=0.8, timestamp=datetime.now()
        )
        
        alert1 = self.formatter.create_alert(high_vol_signal, high_conf)
        alert2 = self.formatter.create_alert(low_vol_signal, low_conf)
        
        # Get alerts by different priorities
        high_priority_alerts = self.formatter.get_alerts_by_priority(AlertPriority.HIGH)
        medium_priority_alerts = self.formatter.get_alerts_by_priority(AlertPriority.MEDIUM)
        low_priority_alerts = self.formatter.get_alerts_by_priority(AlertPriority.LOW)
        very_low_priority_alerts = self.formatter.get_alerts_by_priority(AlertPriority.VERY_LOW)
        
        # Should have created alerts with different priorities
        total_alerts = len(high_priority_alerts) + len(medium_priority_alerts) + len(low_priority_alerts) + len(very_low_priority_alerts)
        assert total_alerts == 2
        
        # Verify priority filtering works
        if high_priority_alerts:
            assert all(alert.priority == AlertPriority.HIGH for alert in high_priority_alerts)
        if very_low_priority_alerts:
            assert all(alert.priority == AlertPriority.VERY_LOW for alert in very_low_priority_alerts)
    
    def test_get_alerts_by_symbol(self):
        """Test filtering alerts by symbol."""
        # Create alerts for different symbols
        tsla_signal = BreakoutSignal(
            symbol="TSLA", breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.0, orb_high=150.0, orb_low=148.0,
            breakout_percentage=0.67, volume_ratio=2.0, timestamp=datetime.now()
        )
        
        alert1 = self.formatter.create_alert(self.test_signal, self.test_confidence)
        alert2 = self.formatter.create_alert(tsla_signal, self.test_confidence)
        
        aapl_alerts = self.formatter.get_alerts_by_symbol("AAPL")
        tsla_alerts = self.formatter.get_alerts_by_symbol("TSLA")
        
        assert len(aapl_alerts) == 1
        assert len(tsla_alerts) == 1
        assert aapl_alerts[0].symbol == "AAPL"
        assert tsla_alerts[0].symbol == "TSLA"
    
    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        # Create multiple alerts
        for i in range(5):
            signal = BreakoutSignal(
                symbol=f"SYM{i}", breakout_type=BreakoutType.BULLISH_BREAKOUT,
                current_price=151.0, orb_high=150.0, orb_low=148.0,
                breakout_percentage=0.67, volume_ratio=2.0,
                timestamp=datetime(2025, 1, 1, 10, 30, i)
            )
            self.formatter.create_alert(signal, self.test_confidence)
        
        recent_alerts = self.formatter.get_recent_alerts(3)
        
        assert len(recent_alerts) == 3
        # Should be sorted by timestamp (most recent first)
        for i in range(len(recent_alerts) - 1):
            assert recent_alerts[i].timestamp >= recent_alerts[i + 1].timestamp
    
    def test_get_daily_summary(self):
        """Test daily summary generation."""
        # Create alerts with different priorities
        high_conf = ConfidenceComponents(0.9, 0.8, 0.7, 0.88)
        med_conf = ConfidenceComponents(0.7, 0.6, 0.5, 0.68)
        
        high_vol_signal = BreakoutSignal(
            symbol="AAPL", breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.0, orb_high=150.0, orb_low=148.0,
            breakout_percentage=0.67, volume_ratio=3.0, timestamp=datetime.now()
        )
        
        med_vol_signal = BreakoutSignal(
            symbol="TSLA", breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.0, orb_high=150.0, orb_low=148.0,
            breakout_percentage=0.67, volume_ratio=1.8, timestamp=datetime.now()
        )
        
        alert1 = self.formatter.create_alert(high_vol_signal, high_conf)
        alert2 = self.formatter.create_alert(med_vol_signal, med_conf)
        
        summary = self.formatter.get_daily_summary()
        
        assert 'date' in summary
        assert 'total_alerts' in summary
        assert 'priority_breakdown' in summary
        assert 'symbol_breakdown' in summary
        assert 'avg_confidence' in summary
        assert 'max_confidence' in summary
        
        assert summary['total_alerts'] == 2
        assert summary['avg_confidence'] > 0
        assert summary['max_confidence'] >= summary['avg_confidence']
    
    def test_clear_history(self):
        """Test clearing alert history."""
        alert = self.formatter.create_alert(self.test_signal, self.test_confidence)
        
        assert len(self.formatter.alert_history) == 1
        assert self.formatter.daily_alert_count == 1
        
        self.formatter.clear_history()
        
        assert len(self.formatter.alert_history) == 0
        assert self.formatter.daily_alert_count == 0
    
    def test_alert_history_limit(self):
        """Test that alert history is limited to 1000 entries."""
        # Create 1005 alerts
        for i in range(1005):
            signal = BreakoutSignal(
                symbol=f"SYM{i % 10}", breakout_type=BreakoutType.BULLISH_BREAKOUT,
                current_price=151.0, orb_high=150.0, orb_low=148.0,
                breakout_percentage=0.67, volume_ratio=2.0, timestamp=datetime.now()
            )
            self.formatter.create_alert(signal, self.test_confidence)
        
        assert len(self.formatter.alert_history) == 1000  # Should be limited to 1000
        assert self.formatter.daily_alert_count == 1005  # Count should still be accurate
    
