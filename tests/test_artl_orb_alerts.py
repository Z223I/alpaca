"""
PyTest unit tests for ORB alerts using ARTL data from 2025-06-30.

This test suite uses real ARTL data to validate the ORB alert system's ability to:
- Detect bearish breakdowns below ORB low
- Calculate confidence scores for bearish scenarios
- Generate appropriate alert priorities for short signals
- Handle high-volume bearish scenarios

Based on ARTL data analysis (15-minute ORB):
- ORB High: $25.10, ORB Low: $19.04, Range: $6.06
- Massive bearish day: -34.5% decline from $19.27 to $12.62
- Bearish breakdowns, max breakdown: $11.99 (37.02% below ORB low)
- High-volume bearish scenarios with significant selling pressure
"""

import pytest
import pandas as pd
import json
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from atoms.alerts.breakout_detector import BreakoutDetector, BreakoutSignal, BreakoutType
from atoms.alerts.confidence_scorer import ConfidenceScorer, ConfidenceComponents
from atoms.alerts.alert_formatter import AlertFormatter, AlertPriority
from atoms.indicators.orb_calculator import ORBCalculator, ORBLevel


class TestARTLORBAlerts:
    """Test ORB alerts using real ARTL bearish data from 2025-06-30."""
    
    @pytest.fixture
    def artl_data(self):
        """Load ARTL stock data from JSON file."""
        test_data_dir = os.path.dirname(os.path.abspath(__file__))
        stock_data_file = os.path.join(test_data_dir, 'data', '20250630.json')
        with open(stock_data_file, 'r') as f:
            data = json.load(f)
        
        artl_bars = data.get('ARTL', [])
        df = pd.DataFrame(artl_bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    
    @pytest.fixture
    def artl_orb_levels(self, artl_data):
        """Calculate ORB levels from ARTL data."""
        orb_start = artl_data.timestamp.min()
        orb_end = orb_start + pd.Timedelta(minutes=15)
        orb_data = artl_data[artl_data.timestamp <= orb_end]
        
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        
        return ORBLevel(
            symbol='ARTL',
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
    def alert_formatter(self, tmp_path):
        """Create alert formatter with temporary output directory."""
        return AlertFormatter(str(tmp_path))
    
    def test_artl_orb_level_calculation(self, artl_data, artl_orb_levels):
        """Test ORB level calculation matches expected values."""
        # Expected values from data analysis (15-minute ORB)
        expected_orb_high = 25.10
        expected_orb_low = 19.04
        expected_orb_range = 6.06
        
        assert abs(artl_orb_levels.orb_high - expected_orb_high) < 0.01
        assert abs(artl_orb_levels.orb_low - expected_orb_low) < 0.01
        assert abs(artl_orb_levels.orb_range - expected_orb_range) < 0.01
    
    def test_artl_massive_bearish_breakdown_detection(self, artl_data, artl_orb_levels, breakout_detector):
        """Test detection of massive bearish breakdown to $11.99."""
        # Mock the ORB calculator to return our calculated levels
        breakout_detector.orb_calculator.get_orb_level.return_value = artl_orb_levels
        
        # Test the maximum breakdown point
        max_breakdown_bar = artl_data.loc[artl_data['low'].idxmin()]
        current_price = max_breakdown_bar['low']  # $11.99
        volume_ratio = 2.0  # Reasonable volume scenario
        
        # Mock alert window check
        with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
            breakout_signal = breakout_detector.detect_breakout(
                symbol='ARTL',
                current_price=current_price,
                volume_ratio=volume_ratio,
                timestamp=max_breakdown_bar['timestamp']
            )
        
        assert breakout_signal is not None
        assert breakout_signal.breakout_type == BreakoutType.BEARISH_BREAKDOWN
        assert breakout_signal.current_price == current_price
        assert breakout_signal.orb_low == artl_orb_levels.orb_low
        
        # Check breakdown percentage (~37.02%)
        expected_breakdown_pct = ((artl_orb_levels.orb_low - current_price) / artl_orb_levels.orb_low) * 100
        assert abs(breakout_signal.breakout_percentage - expected_breakdown_pct) < 0.1
        assert breakout_signal.breakout_percentage > 35.0  # Significant breakdown
    
    def test_artl_high_volume_bearish_scenarios(self, artl_data, artl_orb_levels, breakout_detector):
        """Test high volume bearish breakdown scenarios that should generate HIGH priority alerts."""
        # Mock the ORB calculator
        breakout_detector.orb_calculator.get_orb_level.return_value = artl_orb_levels
        
        # Test high volume bearish bars
        avg_volume = artl_data['volume'].mean()
        high_volume_bars = artl_data[artl_data['volume'] > avg_volume * 2]
        bearish_bars = high_volume_bars[high_volume_bars['low'] < artl_orb_levels.orb_low]
        
        assert len(bearish_bars) >= 20  # Should have many high volume bearish scenarios
        
        # Test specific high volume bearish breakdown
        test_bar = bearish_bars.iloc[0]  # First high volume bearish bar
        volume_ratio = test_bar['volume'] / avg_volume
        
        # Mock alert window check
        with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
            breakout_signal = breakout_detector.detect_breakout(
                symbol='ARTL',
                current_price=test_bar['low'],
                volume_ratio=volume_ratio,
                timestamp=test_bar['timestamp']
            )
        
        assert breakout_signal is not None
        assert breakout_signal.breakout_type == BreakoutType.BEARISH_BREAKDOWN
        assert breakout_signal.volume_ratio >= 2.0
        assert breakout_signal.current_price < artl_orb_levels.orb_low
    
    def test_artl_early_bearish_breakdown_detection(self, artl_data, artl_orb_levels, breakout_detector):
        """Test early bearish breakdown detection in the morning session."""
        # Mock the ORB calculator
        breakout_detector.orb_calculator.get_orb_level.return_value = artl_orb_levels
        
        # Find early bearish breakdowns (11:xx hour)
        early_bearish = artl_data[
            (artl_data['timestamp'].dt.hour == 11) & 
            (artl_data['low'] < artl_orb_levels.orb_low)
        ]
        
        assert len(early_bearish) >= 30  # Should have many early breakdowns
        
        # Test one of the early breakdown scenarios
        test_bar = early_bearish.iloc[10]  # Mid-morning breakdown
        volume_ratio = 3.0  # High volume scenario
        
        with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
            breakout_signal = breakout_detector.detect_breakout(
                symbol='ARTL',
                current_price=test_bar['low'],
                volume_ratio=volume_ratio,
                timestamp=test_bar['timestamp']
            )
        
        assert breakout_signal is not None
        assert breakout_signal.breakout_type == BreakoutType.BEARISH_BREAKDOWN
        assert breakout_signal.timestamp.hour == 11  # Morning breakdown
    
    def test_artl_confidence_scoring_bearish_breakdown(self, artl_orb_levels, confidence_scorer):
        """Test confidence scoring for high-confidence ARTL bearish scenarios."""
        # Create a high-confidence bearish breakdown signal
        high_confidence_signal = BreakoutSignal(
            symbol='ARTL',
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            current_price=12.50,  # Significant breakdown
            orb_high=artl_orb_levels.orb_high,
            orb_low=artl_orb_levels.orb_low,
            breakout_percentage=32.85,  # Major breakdown
            volume_ratio=7.0,  # Very high volume
            timestamp=datetime(2025, 6, 30, 11, 15, 0)
        )
        
        # Mock technical indicators for high confidence bearish scenario
        technical_indicators = {
            'rsi': 25.0,  # Oversold RSI
            'macd_signal': -0.8,  # Negative MACD
            'volume_sma_ratio': 7.0,  # High volume vs SMA
            'price_vs_vwap': 0.85  # Price below VWAP
        }
        
        confidence = confidence_scorer.calculate_confidence_score(
            high_confidence_signal, technical_indicators
        )
        
        # Should generate high confidence scores for bearish breakdown
        assert confidence.total_score >= 0.75  # High confidence
        assert confidence.pc1_score >= 0.75  # Strong ORB momentum (bearish)
        assert confidence.pc2_score >= 0.80  # Strong volume component
    
    def test_artl_bearish_alert_priority_high_priority(self, alert_formatter, artl_orb_levels):
        """Test that high-confidence ARTL bearish scenarios generate HIGH priority alerts."""
        # Create high-confidence bearish breakdown signal
        signal = BreakoutSignal(
            symbol='ARTL',
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            current_price=11.99,  # Maximum breakdown
            orb_high=artl_orb_levels.orb_high,
            orb_low=artl_orb_levels.orb_low,
            breakout_percentage=37.02,  # Massive breakdown
            volume_ratio=6.0,  # High volume
            timestamp=datetime(2025, 6, 30, 16, 4, 0)
        )
        
        # High confidence components for bearish scenario
        confidence = ConfidenceComponents(
            pc1_score=0.95,  # Very strong bearish ORB momentum
            pc2_score=0.90,  # Strong volume dynamics
            pc3_score=0.85,  # Good technical alignment (bearish)
            total_score=0.0  # Will be calculated in __post_init__
        )
        
        alert = alert_formatter.create_alert(signal, confidence)
        
        # Should generate HIGH priority alert for bearish breakdown
        assert alert.priority == AlertPriority.HIGH
        assert alert.confidence_level == "VERY_HIGH"
        assert alert.symbol == 'ARTL'
        assert alert.current_price == 11.99
        assert alert.breakout_type == BreakoutType.BEARISH_BREAKDOWN
        assert alert.breakout_percentage > 37.0
    
    def test_artl_multiple_bearish_breakdown_scenarios(self, artl_data, artl_orb_levels, breakout_detector):
        """Test multiple bearish breakdown scenarios throughout the ARTL trading day."""
        # Mock the ORB calculator
        breakout_detector.orb_calculator.get_orb_level.return_value = artl_orb_levels
        
        # Find all potential bearish breakdowns
        orb_start = artl_data.timestamp.min()
        orb_end = orb_start + pd.Timedelta(minutes=15)
        post_orb_data = artl_data[artl_data.timestamp > orb_end]
        
        bearish_breakdowns = post_orb_data[post_orb_data['low'] < artl_orb_levels.orb_low]
        
        # Should have many breakdown opportunities
        assert len(bearish_breakdowns) >= 250  # Expected from analysis
        
        # Test various breakdown scenarios
        test_count = 0
        for _, bar in bearish_breakdowns.head(15).iterrows():
            volume_ratio = 2.5  # Assume reasonable volume
            
            with patch.object(breakout_detector, 'is_within_alert_window', return_value=True):
                breakout_signal = breakout_detector.detect_breakout(
                    symbol='ARTL',
                    current_price=bar['low'],
                    volume_ratio=volume_ratio,
                    timestamp=bar['timestamp']
                )
            
            if breakout_signal:
                test_count += 1
                assert breakout_signal.breakout_type == BreakoutType.BEARISH_BREAKDOWN
                assert breakout_signal.current_price < artl_orb_levels.orb_low
                assert breakout_signal.breakout_percentage > 0
        
        # Should have generated many bearish breakdown signals
        assert test_count >= 10
    
    def test_artl_bearish_risk_management_calculations(self, alert_formatter, artl_orb_levels):
        """Test risk management calculations for ARTL bearish alerts."""
        # Create a typical ARTL bearish breakdown scenario
        signal = BreakoutSignal(
            symbol='ARTL',
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            current_price=15.00,  # Moderate breakdown
            orb_high=artl_orb_levels.orb_high,
            orb_low=artl_orb_levels.orb_low,
            breakout_percentage=19.34,  # 19% breakdown
            volume_ratio=4.0,
            timestamp=datetime(2025, 6, 30, 12, 30, 0)
        )
        
        confidence = ConfidenceComponents(
            pc1_score=0.85,
            pc2_score=0.80,
            pc3_score=0.75,
            total_score=0.0  # Will be calculated
        )
        
        alert = alert_formatter.create_alert(signal, confidence)
        
        # Verify risk management calculations for bearish breakdown
        assert alert.recommended_stop_loss > 0
        assert alert.recommended_take_profit < alert.current_price  # Take profit below for short
        assert alert.recommended_stop_loss > alert.current_price   # Stop loss above for short
        
        # Stop loss should be 7.5% above current price for bearish breakdown
        expected_stop_loss = 15.00 * (1 + 7.5/100)  # 15.00 * 1.075
        assert abs(alert.recommended_stop_loss - expected_stop_loss) < 0.1
        
        # Take profit should be 4% below current price for bearish breakdown
        expected_take_profit = 15.00 * (1 - 4.0/100)  # 15.00 * 0.96
        assert abs(alert.recommended_take_profit - expected_take_profit) < 0.1
    
    def test_artl_bearish_alert_message_formatting(self, alert_formatter, artl_orb_levels):
        """Test alert message formatting for ARTL bearish alerts."""
        signal = BreakoutSignal(
            symbol='ARTL',
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            current_price=14.50,
            orb_high=artl_orb_levels.orb_high,
            orb_low=artl_orb_levels.orb_low,
            breakout_percentage=22.03,
            volume_ratio=5.2,
            timestamp=datetime(2025, 6, 30, 13, 45, 30)
        )
        
        confidence = ConfidenceComponents(
            pc1_score=0.88,
            pc2_score=0.82,
            pc3_score=0.78,
            total_score=0.0  # Will be calculated
        )
        
        alert = alert_formatter.create_alert(signal, confidence)
        
        # Verify alert message contains key information for bearish breakdown
        assert 'ARTL' in alert.alert_message
        assert '14.50' in alert.alert_message
        assert '↓' in alert.alert_message  # Bearish indicator
        assert '23.' in alert.alert_message  # Breakdown percentage (approximately)
        assert '5.2x' in alert.alert_message  # Volume ratio
        assert 'MEDIUM' in alert.alert_message  # Priority (confidence 0.82 < 0.85 threshold)
        assert 'Stop:' in alert.alert_message
        assert 'Target:' in alert.alert_message
        assert 'ORB Low' in alert.alert_message  # Indicates breakdown vs ORB Low
    
    def test_artl_progressive_bearish_breakdown_severity(self, artl_orb_levels):
        """Test progressive breakdown severity throughout the day."""
        # Test different severity levels of bearish breakdowns
        scenarios = [
            {'price': 18.00, 'time': '11:30', 'expected_breakdown': 5.5},   # Early mild breakdown
            {'price': 15.00, 'time': '12:30', 'expected_breakdown': 21.2},  # Moderate breakdown  
            {'price': 13.00, 'time': '14:30', 'expected_breakdown': 31.7},  # Severe breakdown
            {'price': 11.99, 'time': '16:04', 'expected_breakdown': 37.0},  # Maximum breakdown
        ]
        
        for scenario in scenarios:
            signal = BreakoutSignal(
                symbol='ARTL',
                breakout_type=BreakoutType.BEARISH_BREAKDOWN,
                current_price=scenario['price'],
                orb_high=artl_orb_levels.orb_high,
                orb_low=artl_orb_levels.orb_low,
                breakout_percentage=0.0,  # Will be calculated
                volume_ratio=3.0,
                timestamp=datetime(2025, 6, 30, int(scenario['time'][:2]), int(scenario['time'][3:]), 0)
            )
            
            # Calculate expected breakdown percentage
            expected_pct = ((artl_orb_levels.orb_low - scenario['price']) / artl_orb_levels.orb_low) * 100
            assert abs(signal.breakout_percentage - expected_pct) < 0.5
            assert signal.breakout_percentage >= scenario['expected_breakdown'] - 1.0
    
    def test_artl_intraday_bearish_momentum(self, artl_data, artl_orb_levels):
        """Test intraday bearish momentum analysis."""
        # Analyze price decline progression throughout the day
        hourly_lows = {}
        
        for hour in range(10, 17):  # Market hours
            hour_data = artl_data[artl_data['timestamp'].dt.hour == hour]
            if not hour_data.empty:
                hourly_lows[hour] = hour_data['low'].min()
        
        # Verify progressive bearish trend
        previous_low = None
        declining_hours = 0
        
        for hour in sorted(hourly_lows.keys()):
            current_low = hourly_lows[hour]
            if previous_low is not None and current_low < previous_low:
                declining_hours += 1
            previous_low = current_low
        
        # Should show consistent bearish momentum throughout the day
        assert declining_hours >= 4  # Most hours should show lower lows
        
        # Final hour should be significantly below ORB low
        final_low = hourly_lows[16] if 16 in hourly_lows else hourly_lows[max(hourly_lows.keys())]
        breakdown_pct = ((artl_orb_levels.orb_low - final_low) / artl_orb_levels.orb_low) * 100
        assert breakdown_pct >= 30.0  # Massive breakdown by end of day
    
    def test_artl_volume_spike_on_breakdown(self, artl_data, artl_orb_levels):
        """Test volume spike analysis during major breakdowns."""
        avg_volume = artl_data['volume'].mean()
        
        # Find the most severe breakdowns
        severe_breakdowns = artl_data[artl_data['low'] < artl_orb_levels.orb_low * 0.80]  # 20%+ below ORB low
        
        assert len(severe_breakdowns) >= 50  # Many severe breakdown bars
        
        # Analyze volume during severe breakdowns
        high_volume_severe = severe_breakdowns[severe_breakdowns['volume'] > avg_volume * 2]
        
        # Should have some volume spikes during major breakdowns
        assert len(high_volume_severe) >= 4
        
        # Check maximum volume during breakdown
        max_volume_breakdown = severe_breakdowns.loc[severe_breakdowns['volume'].idxmax()]
        volume_ratio = max_volume_breakdown['volume'] / avg_volume
        assert volume_ratio >= 3.0  # Reasonable volume spike
    
    def test_artl_comprehensive_bearish_alert_generation(self, artl_data, artl_orb_levels):
        """Test comprehensive bearish alert generation using real ARTL data flow."""
        # This test simulates the complete bearish alert generation process
        alert_formatter = AlertFormatter()
        confidence_scorer = ConfidenceScorer()
        
        # Test multiple bearish scenarios from the data
        avg_volume = artl_data['volume'].mean()
        bearish_breakdown_bars = artl_data[artl_data['low'] < artl_orb_levels.orb_low]
        
        generated_alerts = []
        
        # Test various severity levels
        test_scenarios = [
            bearish_breakdown_bars[bearish_breakdown_bars['low'] > artl_orb_levels.orb_low * 0.95],  # Mild breakdowns
            bearish_breakdown_bars[bearish_breakdown_bars['low'] < artl_orb_levels.orb_low * 0.80],  # Severe breakdowns
            bearish_breakdown_bars[bearish_breakdown_bars['volume'] > avg_volume * 3],              # High volume breakdowns
        ]
        
        for scenario_bars in test_scenarios:
            if not scenario_bars.empty:
                # Test first bar from each scenario
                bar = scenario_bars.iloc[0]
                
                # Create breakout signal
                signal = BreakoutSignal(
                    symbol='ARTL',
                    breakout_type=BreakoutType.BEARISH_BREAKDOWN,
                    current_price=bar['low'],
                    orb_high=artl_orb_levels.orb_high,
                    orb_low=artl_orb_levels.orb_low,
                    breakout_percentage=0.0,  # Will be calculated
                    volume_ratio=bar['volume'] / avg_volume,
                    timestamp=bar['timestamp']
                )
                
                # Mock technical indicators for bearish scenario
                technical_indicators = {
                    'rsi': 30.0,  # Oversold
                    'macd_signal': -0.5,  # Negative MACD
                    'volume_sma_ratio': signal.volume_ratio,
                    'price_vs_vwap': 0.90  # Price below VWAP
                }
                
                # Calculate confidence
                confidence = confidence_scorer.calculate_confidence_score(
                    signal, technical_indicators
                )
                
                # Generate alert
                alert = alert_formatter.create_alert(signal, confidence)
                generated_alerts.append(alert)
        
        # Verify we generated bearish alerts
        assert len(generated_alerts) >= 2
        
        # Verify at least one HIGH priority bearish alert
        high_priority_alerts = [a for a in generated_alerts if a.priority == AlertPriority.HIGH]
        assert len(high_priority_alerts) >= 1
        
        # Verify alert quality for bearish breakdowns
        for alert in generated_alerts:
            assert alert.symbol == 'ARTL'
            assert alert.breakout_type == BreakoutType.BEARISH_BREAKDOWN
            assert alert.current_price < artl_orb_levels.orb_low
            assert alert.confidence_score > 0.0
            assert alert.alert_message != ""
            assert alert.recommended_stop_loss > alert.current_price  # Stop above for short
            assert alert.recommended_take_profit < alert.current_price  # Target below for short
            assert alert.breakout_percentage > 0.0  # Should have meaningful breakdown percentage