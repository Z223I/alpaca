"""
PyTest unit tests for ORB alerts using BMNR data from 2025-06-30.

This test suite uses real BMNR data to validate the ORB alert system's ability to:
- Detect bullish breakouts above ORB high
- Calculate confidence scores correctly
- Generate appropriate alert priorities
- Handle high-volume scenarios

Based on BMNR data analysis (15-minute ORB):
- ORB High: $23.80, ORB Low: $17.64, Range: $6.16
- Max breakout: $45.97 (93.19% above ORB high)
- High volume scenarios with 3x+ average volume
"""

import pytest
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from atoms.alerts.breakout_detector import BreakoutDetector, BreakoutSignal, BreakoutType
from atoms.alerts.confidence_scorer import ConfidenceScorer, ConfidenceComponents
from atoms.alerts.alert_formatter import AlertFormatter, ORBAlert, AlertPriority
from atoms.indicators.orb_calculator import ORBCalculator, ORBLevel
from atoms.websocket.alpaca_stream import MarketData
from molecules.orb_alert_engine import ORBAlertEngine


class TestBMNRORBAlerts:
    """Test ORB alerts using real BMNR data from 2025-06-30."""
    
    @pytest.fixture
    def bmnr_data(self):
        """Load BMNR stock data from JSON file."""
        test_data_dir = os.path.dirname(os.path.abspath(__file__))
        stock_data_file = os.path.join(test_data_dir, 'data', '20250630.json')
        with open(stock_data_file, 'r') as f:
            data = json.load(f)
        
        bmnr_bars = data.get('BMNR', [])
        df = pd.DataFrame(bmnr_bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    
    @pytest.fixture
    def bmnr_orb_levels(self, bmnr_data):
        """Calculate ORB levels from BMNR data."""
        orb_start = bmnr_data.timestamp.min()
        orb_end = orb_start + pd.Timedelta(minutes=15)
        orb_data = bmnr_data[bmnr_data.timestamp <= orb_end]
        
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        
        return ORBLevel(
            symbol='BMNR',
            orb_high=orb_high,
            orb_low=orb_low,
            orb_range=orb_high - orb_low,
            calculation_time=orb_start,
            sample_count=len(orb_data)
        )
    
    @pytest.fixture
    def breakout_detector(self):
        """Create breakout detector with mocked ORB calculator."""
        orb_calculator = Mock(spec=ORBCalculator)
        detector = BreakoutDetector(orb_calculator)
        return detector
    
    @pytest.fixture
    def confidence_scorer(self):
        """Create confidence scorer."""
        return ConfidenceScorer()
    
    @pytest.fixture
    def alert_formatter(self):
        """Create alert formatter."""
        return AlertFormatter()
    
    def test_bmnr_orb_level_calculation(self, bmnr_data, bmnr_orb_levels):
        """Test ORB level calculation matches expected values."""
        # Expected values from data analysis (15-minute ORB)
        expected_orb_high = 23.80
        expected_orb_low = 17.64
        expected_orb_range = 6.16
        
        assert abs(bmnr_orb_levels.orb_high - expected_orb_high) < 0.01
        assert abs(bmnr_orb_levels.orb_low - expected_orb_low) < 0.01
        assert abs(bmnr_orb_levels.orb_range - expected_orb_range) < 0.01
    
    def test_bmnr_major_bullish_breakout_detection(self, bmnr_data, bmnr_orb_levels, breakout_detector):
        """Test detection of major bullish breakout to $45.97."""
        # Mock the ORB calculator to return our calculated levels
        breakout_detector.orb_calculator.get_orb_level.return_value = bmnr_orb_levels
        
        # Test the maximum breakout point
        max_breakout_bar = bmnr_data.loc[bmnr_data['high'].idxmax()]
        current_price = max_breakout_bar['high']  # $45.97
        volume_ratio = 2.5  # High volume scenario
        
        # Mock alert window check
        with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
            breakout_signal = breakout_detector.detect_breakout(
                symbol='BMNR',
                current_price=current_price,
                volume_ratio=volume_ratio,
                timestamp=max_breakout_bar['timestamp']
            )
        
        assert breakout_signal is not None
        assert breakout_signal.breakout_type == BreakoutType.BULLISH_BREAKOUT
        assert breakout_signal.current_price == current_price
        assert breakout_signal.orb_high == bmnr_orb_levels.orb_high
        
        # Check breakout percentage (~93.19%)
        expected_breakout_pct = ((current_price - bmnr_orb_levels.orb_high) / bmnr_orb_levels.orb_high) * 100
        assert abs(breakout_signal.breakout_percentage - expected_breakout_pct) < 0.1
    
    def test_bmnr_high_volume_scenarios(self, bmnr_data, bmnr_orb_levels, breakout_detector):
        """Test high volume breakout scenarios that should generate HIGH priority alerts."""
        # Mock the ORB calculator
        breakout_detector.orb_calculator.get_orb_level.return_value = bmnr_orb_levels
        
        # Test high volume bars (>3x average volume = 520,575)
        avg_volume = bmnr_data['volume'].mean()
        high_volume_bars = bmnr_data[bmnr_data['volume'] > avg_volume * 3]
        
        assert len(high_volume_bars) >= 10  # Should have many high volume scenarios
        
        # Test specific high volume breakout
        test_bar = high_volume_bars.iloc[4]  # One of the high volume bars
        volume_ratio = test_bar['volume'] / avg_volume
        
        # Mock alert window check
        with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
            breakout_signal = breakout_detector.detect_breakout(
                symbol='BMNR',
                current_price=test_bar['high'],
                volume_ratio=volume_ratio,
                timestamp=test_bar['timestamp']
            )
        
        if breakout_signal:  # If it's a breakout
            assert breakout_signal.volume_ratio >= 3.0
            assert breakout_signal.breakout_type == BreakoutType.BULLISH_BREAKOUT
    
    def test_bmnr_confidence_scoring_high_confidence(self, bmnr_orb_levels, confidence_scorer):
        """Test confidence scoring for high-confidence BMNR scenarios."""
        # Create a high-confidence breakout signal
        high_confidence_signal = BreakoutSignal(
            symbol='BMNR',
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=45.97,  # Max breakout price
            orb_high=bmnr_orb_levels.orb_high,
            orb_low=bmnr_orb_levels.orb_low,
            breakout_percentage=79.22,  # Significant breakout
            volume_ratio=5.0,  # Very high volume
            timestamp=datetime(2025, 6, 30, 16, 37, 0)
        )
        
        # Mock technical indicators for high confidence
        technical_indicators = {
            'rsi': 65.0,  # Moderate RSI
            'macd_signal': 0.5,  # Positive MACD
            'volume_sma_ratio': 5.0,  # High volume vs SMA
            'price_vs_vwap': 1.05  # Price above VWAP
        }
        
        confidence = confidence_scorer.calculate_confidence_score(
            high_confidence_signal, technical_indicators
        )
        
        # Should generate high confidence scores
        assert confidence.total_score >= 0.80  # High confidence
        assert confidence.pc1_score >= 0.80  # Strong ORB momentum
        assert confidence.pc2_score >= 0.80  # Strong volume component
    
    def test_bmnr_alert_priority_high_priority(self, alert_formatter, bmnr_orb_levels, confidence_scorer):
        """Test that high-confidence BMNR scenarios generate HIGH priority alerts."""
        # Create high-confidence breakout signal
        signal = BreakoutSignal(
            symbol='BMNR',
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=45.97,
            orb_high=bmnr_orb_levels.orb_high,
            orb_low=bmnr_orb_levels.orb_low,
            breakout_percentage=79.22,
            volume_ratio=5.0,
            timestamp=datetime(2025, 6, 30, 16, 37, 0)
        )
        
        # High confidence components (calculated to exceed 0.85 threshold)
        confidence = ConfidenceComponents(
            pc1_score=0.95,  # Very strong ORB momentum
            pc2_score=0.90,  # Strong volume dynamics
            pc3_score=0.85,  # Good technical alignment
            total_score=0.0  # Will be calculated in __post_init__
        )
        
        alert = alert_formatter.create_alert(signal, confidence)
        
        # Should generate HIGH priority alert
        assert alert.priority == AlertPriority.HIGH
        assert alert.confidence_level == "VERY_HIGH"
        assert alert.symbol == 'BMNR'
        assert alert.current_price == 45.97
        assert alert.breakout_type == BreakoutType.BULLISH_BREAKOUT
    
    def test_bmnr_multiple_breakout_scenarios(self, bmnr_data, bmnr_orb_levels, breakout_detector):
        """Test multiple breakout scenarios throughout the BMNR trading day."""
        # Mock the ORB calculator
        breakout_detector.orb_calculator.get_orb_level.return_value = bmnr_orb_levels
        
        # Find all potential breakouts
        orb_start = bmnr_data.timestamp.min()
        orb_end = orb_start + pd.Timedelta(minutes=15)
        post_orb_data = bmnr_data[bmnr_data.timestamp > orb_end]
        
        bullish_breakouts = post_orb_data[post_orb_data['high'] > bmnr_orb_levels.orb_high]
        
        # Should have many breakout opportunities
        assert len(bullish_breakouts) >= 100  # Expected from analysis
        
        # Test various breakout scenarios
        test_count = 0
        for idx, bar in bullish_breakouts.head(10).iterrows():
            volume_ratio = 2.0  # Assume reasonable volume
            
            with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
                breakout_signal = breakout_detector.detect_breakout(
                    symbol='BMNR',
                    current_price=bar['high'],
                    volume_ratio=volume_ratio,
                    timestamp=bar['timestamp']
                )
            
            if breakout_signal:
                test_count += 1
                assert breakout_signal.breakout_type == BreakoutType.BULLISH_BREAKOUT
                assert breakout_signal.current_price > bmnr_orb_levels.orb_high
                assert breakout_signal.breakout_percentage > 0
        
        # Should have generated several breakout signals
        assert test_count >= 5
    
    def test_bmnr_risk_management_calculations(self, alert_formatter, bmnr_orb_levels):
        """Test risk management calculations for BMNR alerts."""
        # Create a typical BMNR breakout scenario
        signal = BreakoutSignal(
            symbol='BMNR',
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=30.00,  # Moderate breakout
            orb_high=bmnr_orb_levels.orb_high,
            orb_low=bmnr_orb_levels.orb_low,
            breakout_percentage=16.99,  # 17% breakout
            volume_ratio=2.5,
            timestamp=datetime(2025, 6, 30, 11, 30, 0)
        )
        
        confidence = ConfidenceComponents(
            pc1_score=0.75,
            pc2_score=0.70,
            pc3_score=0.65,
            total_score=0.72
        )
        
        alert = alert_formatter.create_alert(signal, confidence)
        
        # Verify risk management calculations
        assert alert.recommended_stop_loss > 0
        assert alert.recommended_take_profit > alert.current_price
        assert alert.recommended_stop_loss < alert.current_price
        
        # Stop loss should be 7.5% below current price for bullish breakout
        expected_stop_loss = 30.00 * (1 - 7.5/100)  # 30.00 * 0.925
        assert abs(alert.recommended_stop_loss - expected_stop_loss) < 0.1
        
        # Take profit should be 4% above current price for bullish breakout
        expected_take_profit = 30.00 * (1 + 4.0/100)  # 30.00 * 1.04
        assert abs(alert.recommended_take_profit - expected_take_profit) < 0.1
    
    def test_bmnr_alert_message_formatting(self, alert_formatter, bmnr_orb_levels):
        """Test alert message formatting for BMNR alerts."""
        signal = BreakoutSignal(
            symbol='BMNR',
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=35.50,
            orb_high=bmnr_orb_levels.orb_high,
            orb_low=bmnr_orb_levels.orb_low,
            breakout_percentage=38.37,
            volume_ratio=3.2,
            timestamp=datetime(2025, 6, 30, 14, 15, 30)
        )
        
        confidence = ConfidenceComponents(
            pc1_score=0.82,
            pc2_score=0.78,
            pc3_score=0.75,
            total_score=0.80
        )
        
        alert = alert_formatter.create_alert(signal, confidence)
        
        # Verify alert message contains key information
        assert 'BMNR' in alert.alert_message
        assert '35.50' in alert.alert_message
        assert '↑' in alert.alert_message  # Bullish indicator
        assert '49.' in alert.alert_message  # Breakout percentage (approximately)
        assert '3.2x' in alert.alert_message  # Volume ratio
        assert 'MEDIUM' in alert.alert_message  # Priority
        assert 'Stop:' in alert.alert_message
        assert 'Target:' in alert.alert_message
    
    def test_bmnr_very_high_confidence_scenarios(self, alert_formatter, bmnr_orb_levels):
        """Test scenarios that should generate VERY HIGH confidence alerts."""
        # Maximum breakout scenario
        signal = BreakoutSignal(
            symbol='BMNR',
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=45.97,  # Maximum price from data
            orb_high=bmnr_orb_levels.orb_high,
            orb_low=bmnr_orb_levels.orb_low,
            breakout_percentage=79.22,
            volume_ratio=8.0,  # Very high volume
            timestamp=datetime(2025, 6, 30, 16, 37, 0)
        )
        
        # Very high confidence components (calculated for >0.90)
        confidence = ConfidenceComponents(
            pc1_score=0.98,  # Exceptional ORB momentum
            pc2_score=0.95,  # Exceptional volume
            pc3_score=0.90,  # Strong technical alignment
            total_score=0.0  # Will be calculated in __post_init__
        )
        
        alert = alert_formatter.create_alert(signal, confidence)
        
        # Should generate VERY HIGH confidence, HIGH priority alert
        assert alert.priority == AlertPriority.HIGH
        assert alert.confidence_level == "VERY_HIGH"
        assert alert.confidence_score >= 0.90
        assert alert.breakout_percentage > 75.0
        assert alert.volume_ratio >= 8.0
    
    def test_bmnr_end_of_day_scenarios(self, bmnr_data, bmnr_orb_levels, breakout_detector):
        """Test breakout detection during end-of-day scenarios."""
        # Mock the ORB calculator
        breakout_detector.orb_calculator.get_orb_level.return_value = bmnr_orb_levels
        
        # Test end-of-day data (after 4:00 PM)
        eod_data = bmnr_data[bmnr_data['timestamp'].dt.hour >= 16]
        
        if not eod_data.empty:
            # Test a late-day breakout
            test_bar = eod_data.iloc[-10]  # Near end of day
            volume_ratio = 2.0
            
            with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
                breakout_signal = breakout_detector.detect_breakout(
                    symbol='BMNR',
                    current_price=test_bar['high'],
                    volume_ratio=volume_ratio,
                    timestamp=test_bar['timestamp']
                )
            
            if breakout_signal and test_bar['high'] > bmnr_orb_levels.orb_high:
                assert breakout_signal.breakout_type == BreakoutType.BULLISH_BREAKOUT
                assert breakout_signal.timestamp.hour >= 16
    
    def test_bmnr_comprehensive_alert_generation(self, bmnr_data, bmnr_orb_levels):
        """Test comprehensive alert generation using real BMNR data flow."""
        # This test simulates the complete alert generation process
        alert_formatter = AlertFormatter()
        confidence_scorer = ConfidenceScorer()
        
        # Test multiple scenarios from the data
        avg_volume = bmnr_data['volume'].mean()
        
        # Find bars that break above ORB high
        breakout_bars = bmnr_data[bmnr_data['high'] > bmnr_orb_levels.orb_high]
        
        generated_alerts = []
        
        for idx, bar in breakout_bars.head(5).iterrows():
            # Create breakout signal
            signal = BreakoutSignal(
                symbol='BMNR',
                breakout_type=BreakoutType.BULLISH_BREAKOUT,
                current_price=bar['high'],
                orb_high=bmnr_orb_levels.orb_high,
                orb_low=bmnr_orb_levels.orb_low,
                breakout_percentage=0.0,  # Will be calculated
                volume_ratio=bar['volume'] / avg_volume,
                timestamp=bar['timestamp']
            )
            
            # Mock technical indicators
            technical_indicators = {
                'rsi': 60.0,
                'macd_signal': 0.3,
                'volume_sma_ratio': signal.volume_ratio,
                'price_vs_vwap': 1.02
            }
            
            # Calculate confidence
            confidence = confidence_scorer.calculate_confidence_score(
                signal, technical_indicators
            )
            
            # Generate alert
            alert = alert_formatter.create_alert(signal, confidence)
            generated_alerts.append(alert)
        
        # Verify we generated alerts
        assert len(generated_alerts) >= 3
        
        # Verify at least one HIGH priority alert
        high_priority_alerts = [a for a in generated_alerts if a.priority == AlertPriority.HIGH]
        assert len(high_priority_alerts) >= 1
        
        # Verify alert quality
        for alert in generated_alerts:
            assert alert.symbol == 'BMNR'
            assert alert.breakout_type == BreakoutType.BULLISH_BREAKOUT
            assert alert.current_price > bmnr_orb_levels.orb_high
            assert alert.confidence_score > 0.0
            assert alert.alert_message != ""
            assert alert.recommended_stop_loss > 0
            assert alert.recommended_take_profit > alert.current_price
