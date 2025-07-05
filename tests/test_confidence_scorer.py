"""
Unit tests for ORB Confidence Scorer

Tests the PCA-based confidence scoring system.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch

from atoms.alerts.confidence_scorer import (
    ConfidenceScorer, ConfidenceComponents
)
from atoms.alerts.breakout_detector import BreakoutSignal, BreakoutType


class TestConfidenceComponents:
    """Test cases for ConfidenceComponents data class."""
    
    def test_confidence_components_creation(self):
        """Test confidence components creation and calculation."""
        components = ConfidenceComponents(
            pc1_score=0.8,
            pc2_score=0.6,
            pc3_score=0.4,
            total_score=0.0  # Will be calculated
        )
        
        assert components.pc1_score == 0.8
        assert components.pc2_score == 0.6
        assert components.pc3_score == 0.4
        # Total should be weighted sum based on config
        assert components.total_score > 0
    
    @patch('atoms.alerts.confidence_scorer.config')
    def test_confidence_components_weighted_calculation(self, mock_config):
        """Test weighted calculation of total score."""
        mock_config.pc1_weight = 0.8
        mock_config.pc2_weight = 0.1
        mock_config.pc3_weight = 0.1
        
        components = ConfidenceComponents(
            pc1_score=1.0,
            pc2_score=0.5,
            pc3_score=0.3,
            total_score=0.0
        )
        
        expected_total = 0.8 * 1.0 + 0.1 * 0.5 + 0.1 * 0.3
        assert abs(components.total_score - expected_total) < 0.001


class TestConfidenceScorer:
    """Test cases for ConfidenceScorer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scorer = ConfidenceScorer()
        
        # Create test breakout signal
        self.test_signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            timestamp=datetime.now()
        )
        
        # Create test technical indicators
        self.test_indicators = {
            'ema_9': 150.5,
            'vwap': 149.8,
            'ema_deviation': 0.003,
            'vwap_deviation': 0.008
        }
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = ConfidenceScorer()
        assert len(scorer.score_history) == 0
    
    def test_calculate_confidence_score_success(self):
        """Test successful confidence score calculation."""
        components = self.scorer.calculate_confidence_score(
            self.test_signal, self.test_indicators
        )
        
        assert isinstance(components, ConfidenceComponents)
        assert 0.0 <= components.pc1_score <= 1.0
        assert 0.0 <= components.pc2_score <= 1.0
        assert 0.0 <= components.pc3_score <= 1.0
        assert 0.0 <= components.total_score <= 1.0
    
    def test_calculate_confidence_score_stores_history(self):
        """Test that confidence scores are stored in history."""
        assert len(self.scorer.score_history) == 0
        
        self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        assert "AAPL" in self.scorer.score_history
        assert len(self.scorer.score_history["AAPL"]) == 1
    
    def test_calculate_pc1_score_bullish_breakout(self):
        """Test PC1 score calculation for bullish breakout."""
        score = self.scorer._calculate_pc1_score(self.test_signal)
        
        assert 0.0 <= score <= 1.0
        # Should be positive for bullish breakout
        assert score > 0.0
    
    def test_calculate_pc1_score_bearish_breakdown(self):
        """Test PC1 score calculation for bearish breakdown."""
        bearish_signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BEARISH_BREAKDOWN,
            current_price=147.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.68,
            volume_ratio=2.0,
            timestamp=datetime.now()
        )
        
        score = self.scorer._calculate_pc1_score(bearish_signal)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.0
    
    def test_calculate_pc1_score_no_breakout(self):
        """Test PC1 score calculation for no breakout."""
        no_breakout_signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.NO_BREAKOUT,
            current_price=149.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.0,
            volume_ratio=1.0,
            timestamp=datetime.now()
        )
        
        score = self.scorer._calculate_pc1_score(no_breakout_signal)
        assert score == 0.0
    
    def test_calculate_pc2_score_high_volume(self):
        """Test PC2 score calculation with high volume."""
        score = self.scorer._calculate_pc2_score(self.test_signal)
        
        assert 0.0 <= score <= 1.0
        # With 2.0x volume ratio, should get positive score
        assert score > 0.0
    
    def test_calculate_pc2_score_low_volume(self):
        """Test PC2 score calculation with low volume."""
        low_volume_signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=0.5,  # Low volume
            timestamp=datetime.now()
        )
        
        score = self.scorer._calculate_pc2_score(low_volume_signal)
        assert score == 0.0
    
    def test_calculate_pc2_score_exceptional_volume(self):
        """Test PC2 score calculation with exceptional volume."""
        high_volume_signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=4.0,  # Very high volume
            timestamp=datetime.now()
        )
        
        score = self.scorer._calculate_pc2_score(high_volume_signal)
        
        # Should get bonus for exceptional volume
        assert score > 0.5
    
    def test_calculate_pc3_score_with_indicators(self):
        """Test PC3 score calculation with technical indicators."""
        score = self.scorer._calculate_pc3_score(self.test_indicators)
        
        assert 0.0 <= score <= 1.0
        # With small deviations, should get positive but modest score
        assert score > 0.0
    
    def test_calculate_pc3_score_no_indicators(self):
        """Test PC3 score calculation without indicators."""
        score = self.scorer._calculate_pc3_score({})
        assert score == 0.0
    
    def test_calculate_pc3_score_high_deviation(self):
        """Test PC3 score calculation with high technical deviations."""
        high_deviation_indicators = {
            'ema_9': 150.5,
            'vwap': 149.8,
            'ema_deviation': 0.08,  # 8% deviation
            'vwap_deviation': 0.06  # 6% deviation
        }
        
        score = self.scorer._calculate_pc3_score(high_deviation_indicators)
        
        # High deviations should yield high score (capped at 1.0)
        assert score == 1.0
    
    def test_get_confidence_level_very_high(self):
        """Test confidence level classification - very high."""
        level = self.scorer.get_confidence_level(0.90)
        assert level == "VERY_HIGH"
    
    def test_get_confidence_level_high(self):
        """Test confidence level classification - high."""
        level = self.scorer.get_confidence_level(0.75)
        assert level == "HIGH"
    
    def test_get_confidence_level_medium(self):
        """Test confidence level classification - medium."""
        level = self.scorer.get_confidence_level(0.65)
        assert level == "MEDIUM"
    
    def test_get_confidence_level_low(self):
        """Test confidence level classification - low."""
        level = self.scorer.get_confidence_level(0.55)
        assert level == "LOW"
    
    def test_get_confidence_level_very_low(self):
        """Test confidence level classification - very low."""
        level = self.scorer.get_confidence_level(0.40)
        assert level == "VERY_LOW"
    
    def test_should_generate_alert_true(self):
        """Test alert generation decision - should generate."""
        components = ConfidenceComponents(
            pc1_score=0.8,
            pc2_score=0.7,
            pc3_score=0.6,
            total_score=0.75
        )
        
        # Mock config to have lower threshold
        with patch('atoms.alerts.confidence_scorer.config') as mock_config:
            mock_config.min_confidence_score = 0.70
            result = self.scorer.should_generate_alert(components)
            assert result is True
    
    def test_should_generate_alert_false(self):
        """Test alert generation decision - should not generate."""
        components = ConfidenceComponents(
            pc1_score=0.6,
            pc2_score=0.5,
            pc3_score=0.4,
            total_score=0.55
        )
        
        # Mock config to have higher threshold
        with patch('atoms.alerts.confidence_scorer.config') as mock_config:
            mock_config.min_confidence_score = 0.70
            result = self.scorer.should_generate_alert(components)
            assert result is False
    
    def test_get_score_history_existing_symbol(self):
        """Test getting score history for existing symbol."""
        # Generate some scores
        self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        history = self.scorer.get_score_history("AAPL")
        assert len(history) == 2
        assert all(isinstance(comp, ConfidenceComponents) for comp in history)
    
    def test_get_score_history_nonexistent_symbol(self):
        """Test getting score history for nonexistent symbol."""
        history = self.scorer.get_score_history("TSLA")
        assert history == []
    
    def test_get_average_score_with_history(self):
        """Test getting average score with history."""
        # Generate multiple scores
        for _ in range(3):
            self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        avg_score = self.scorer.get_average_score("AAPL")
        assert avg_score is not None
        assert 0.0 <= avg_score <= 1.0
    
    def test_get_average_score_no_history(self):
        """Test getting average score without history."""
        avg_score = self.scorer.get_average_score("TSLA")
        assert avg_score is None
    
    def test_get_average_score_custom_lookback(self):
        """Test getting average score with custom lookback period."""
        # Generate 5 scores
        for _ in range(5):
            self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        # Get average of last 3
        avg_score = self.scorer.get_average_score("AAPL", lookback_periods=3)
        assert avg_score is not None
    
    def test_get_score_statistics_with_data(self):
        """Test getting score statistics with data."""
        # Generate multiple scores
        for _ in range(5):
            self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        stats = self.scorer.get_score_statistics("AAPL")
        
        assert 'total_mean' in stats
        assert 'total_std' in stats
        assert 'total_max' in stats
        assert 'total_min' in stats
        assert 'pc1_mean' in stats
        assert 'pc2_mean' in stats
        assert 'pc3_mean' in stats
        assert 'score_count' in stats
        assert stats['score_count'] == 5
    
    def test_get_score_statistics_no_data(self):
        """Test getting score statistics without data."""
        stats = self.scorer.get_score_statistics("TSLA")
        assert stats == {}
    
    def test_clear_history_specific_symbol(self):
        """Test clearing history for specific symbol."""
        # Generate scores for multiple symbols
        self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        tsla_signal = BreakoutSignal(
            symbol="TSLA",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=151.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.67,
            volume_ratio=2.0,
            timestamp=datetime.now()
        )
        self.scorer.calculate_confidence_score(tsla_signal, self.test_indicators)
        
        # Clear only AAPL
        self.scorer.clear_history("AAPL")
        
        assert "AAPL" not in self.scorer.score_history
        assert "TSLA" in self.scorer.score_history
    
    def test_clear_history_all_symbols(self):
        """Test clearing history for all symbols."""
        # Generate scores
        self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        # Clear all
        self.scorer.clear_history()
        
        assert len(self.scorer.score_history) == 0
    
    def test_get_top_symbols_by_confidence(self):
        """Test getting top symbols by confidence score."""
        # Generate scores for multiple symbols with different confidence levels
        high_confidence_signal = BreakoutSignal(
            symbol="AAPL",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=152.00,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=1.33,
            volume_ratio=3.0,
            timestamp=datetime.now()
        )
        
        low_confidence_signal = BreakoutSignal(
            symbol="TSLA",
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            current_price=150.10,
            orb_high=150.00,
            orb_low=148.00,
            breakout_percentage=0.07,
            volume_ratio=1.2,
            timestamp=datetime.now()
        )
        
        # Generate scores
        self.scorer.calculate_confidence_score(high_confidence_signal, self.test_indicators)
        self.scorer.calculate_confidence_score(low_confidence_signal, self.test_indicators)
        
        top_symbols = self.scorer.get_top_symbols_by_confidence(limit=10)
        
        assert len(top_symbols) == 2
        assert top_symbols[0][0] == "AAPL"  # Should be first (higher confidence)
        assert top_symbols[1][0] == "TSLA"  # Should be second
        assert top_symbols[0][1] > top_symbols[1][1]  # AAPL score > TSLA score
    
    def test_score_history_limit(self):
        """Test that score history is limited to 100 entries."""
        # Generate 105 scores
        for _ in range(105):
            self.scorer.calculate_confidence_score(self.test_signal, self.test_indicators)
        
        history = self.scorer.get_score_history("AAPL")
        assert len(history) == 100  # Should be limited to 100