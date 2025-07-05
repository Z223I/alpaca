"""
PCA-Based Confidence Scoring System

This module implements confidence scoring based on PCA analysis showing:
- 82.31% variance explained by ORB patterns (PC1)
- 8.54% variance from volume dynamics (PC2)
- 3.78% variance from technical divergences (PC3)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from atoms.config.alert_config import config
from atoms.alerts.breakout_detector import BreakoutSignal, BreakoutType


@dataclass
class ConfidenceComponents:
    """Individual components of confidence score."""
    pc1_score: float  # ORB momentum component
    pc2_score: float  # Volume dynamics component
    pc3_score: float  # Technical divergence component
    total_score: float  # Weighted total
    
    def __post_init__(self):
        """Calculate weighted total score."""
        self.total_score = (
            config.pc1_weight * self.pc1_score +
            config.pc2_weight * self.pc2_score +
            config.pc3_weight * self.pc3_score
        )


class ConfidenceScorer:
    """PCA-based confidence scoring for breakout signals."""
    
    def __init__(self):
        """Initialize confidence scorer."""
        self.score_history: Dict[str, List[ConfidenceComponents]] = {}
        
    def calculate_confidence_score(self, signal: BreakoutSignal, 
                                 technical_indicators: Dict[str, float]) -> ConfidenceComponents:
        """
        Calculate comprehensive confidence score for a breakout signal.
        
        Args:
            signal: BreakoutSignal to score
            technical_indicators: Technical indicator values
            
        Returns:
            ConfidenceComponents with individual and total scores
        """
        # PC1: ORB momentum component (82.31% variance)
        pc1_score = self._calculate_pc1_score(signal)
        
        # PC2: Volume dynamics component (8.54% variance)
        pc2_score = self._calculate_pc2_score(signal)
        
        # PC3: Technical divergence component (3.78% variance)
        pc3_score = self._calculate_pc3_score(technical_indicators)
        
        components = ConfidenceComponents(
            pc1_score=pc1_score,
            pc2_score=pc2_score,
            pc3_score=pc3_score,
            total_score=0.0  # Will be calculated in __post_init__
        )
        
        # Store in history
        if signal.symbol not in self.score_history:
            self.score_history[signal.symbol] = []
        self.score_history[signal.symbol].append(components)
        
        # Keep only recent scores (last 100)
        if len(self.score_history[signal.symbol]) > 100:
            self.score_history[signal.symbol] = self.score_history[signal.symbol][-100:]
            
        return components
    
    def _calculate_pc1_score(self, signal: BreakoutSignal) -> float:
        """
        Calculate PC1 score: ORB momentum component.
        
        This represents the strength of the breakout relative to the ORB range.
        Higher breakout percentages and stronger momentum yield higher scores.
        
        Args:
            signal: BreakoutSignal to analyze
            
        Returns:
            PC1 score (0.0 to 1.0)
        """
        # For no breakout, return 0
        if signal.breakout_type == BreakoutType.NO_BREAKOUT:
            return 0.0
            
        # Base score from breakout percentage
        breakout_strength = min(abs(signal.breakout_percentage) / 2.0, 1.0)  # Cap at 2%
        
        # Adjust for breakout type
        if signal.breakout_type == BreakoutType.BULLISH_BREAKOUT:
            direction_multiplier = 1.0
        elif signal.breakout_type == BreakoutType.BEARISH_BREAKDOWN:
            direction_multiplier = 0.8  # Slightly lower weight for bearish
        else:
            direction_multiplier = 0.0
            
        # Calculate ORB range quality
        orb_range = signal.orb_high - signal.orb_low
        orb_range_quality = min(orb_range / signal.orb_low, 0.1) * 10  # 0-1 scale
        
        # Combine components
        pc1_score = (
            breakout_strength * 0.6 +
            direction_multiplier * 0.25 +
            orb_range_quality * 0.15
        )
        
        return max(0.0, min(1.0, pc1_score))
    
    def _calculate_pc2_score(self, signal: BreakoutSignal) -> float:
        """
        Calculate PC2 score: Volume dynamics component.
        
        This represents the strength of volume confirmation for the breakout.
        Higher volume ratios indicate stronger institutional interest.
        
        Args:
            signal: BreakoutSignal to analyze
            
        Returns:
            PC2 score (0.0 to 1.0)
        """
        # Volume ratio scoring
        volume_ratio = signal.volume_ratio
        
        # Logarithmic scaling for volume (diminishing returns)
        if volume_ratio >= 1.0:
            volume_score = min(np.log(volume_ratio) / np.log(5.0), 1.0)  # Cap at 5x
        else:
            volume_score = 0.0
            
        # Bonus for exceptionally high volume
        if volume_ratio >= 3.0:
            volume_score += 0.2
            
        # Penalty for low volume
        if volume_ratio < config.volume_multiplier:
            volume_score *= 0.5
            
        return max(0.0, min(1.0, volume_score))
    
    def _calculate_pc3_score(self, technical_indicators: Dict[str, float]) -> float:
        """
        Calculate PC3 score: Technical divergence component.
        
        This represents technical indicator alignment with the breakout.
        EMA and VWAP deviations indicate momentum strength.
        
        Args:
            technical_indicators: Dictionary of technical indicator values
            
        Returns:
            PC3 score (0.0 to 1.0)
        """
        if not technical_indicators:
            return 0.0
            
        ema_deviation = technical_indicators.get('ema_deviation', 0.0)
        vwap_deviation = technical_indicators.get('vwap_deviation', 0.0)
        
        # Score based on technical divergence
        # Higher deviations indicate stronger momentum
        ema_score = min(ema_deviation / 0.05, 1.0)  # 5% max deviation
        vwap_score = min(vwap_deviation / 0.05, 1.0)  # 5% max deviation
        
        # Combine EMA and VWAP scores
        pc3_score = (ema_score + vwap_score) / 2.0
        
        return max(0.0, min(1.0, pc3_score))
    
    def get_confidence_level(self, total_score: float) -> str:
        """
        Get confidence level description based on total score.
        
        Args:
            total_score: Total confidence score (0.0 to 1.0)
            
        Returns:
            Confidence level string
        """
        if total_score >= 0.85:
            return "VERY_HIGH"
        elif total_score >= 0.70:
            return "HIGH"
        elif total_score >= 0.60:
            return "MEDIUM"
        elif total_score >= 0.50:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def should_generate_alert(self, components: ConfidenceComponents) -> bool:
        """
        Determine if an alert should be generated based on confidence score.
        
        Args:
            components: ConfidenceComponents to evaluate
            
        Returns:
            True if alert should be generated
        """
        return components.total_score >= config.min_confidence_score
    
    def get_score_history(self, symbol: str) -> List[ConfidenceComponents]:
        """
        Get confidence score history for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of ConfidenceComponents
        """
        return self.score_history.get(symbol, [])
    
    def get_average_score(self, symbol: str, lookback_periods: int = 10) -> Optional[float]:
        """
        Get average confidence score for a symbol over recent periods.
        
        Args:
            symbol: Trading symbol
            lookback_periods: Number of periods to average
            
        Returns:
            Average confidence score or None
        """
        history = self.get_score_history(symbol)
        if not history:
            return None
            
        recent_scores = [comp.total_score for comp in history[-lookback_periods:]]
        return sum(recent_scores) / len(recent_scores)
    
    def get_score_statistics(self, symbol: str) -> Dict[str, float]:
        """
        Get comprehensive statistics for a symbol's confidence scores.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with score statistics
        """
        history = self.get_score_history(symbol)
        if not history:
            return {}
            
        total_scores = [comp.total_score for comp in history]
        pc1_scores = [comp.pc1_score for comp in history]
        pc2_scores = [comp.pc2_score for comp in history]
        pc3_scores = [comp.pc3_score for comp in history]
        
        return {
            'total_mean': np.mean(total_scores),
            'total_std': np.std(total_scores),
            'total_max': np.max(total_scores),
            'total_min': np.min(total_scores),
            'pc1_mean': np.mean(pc1_scores),
            'pc2_mean': np.mean(pc2_scores),
            'pc3_mean': np.mean(pc3_scores),
            'score_count': len(history)
        }
    
    def clear_history(self, symbol: str = None) -> None:
        """
        Clear confidence score history.
        
        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            self.score_history.pop(symbol, None)
        else:
            self.score_history.clear()
    
    def get_top_symbols_by_confidence(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get top symbols by average confidence score.
        
        Args:
            limit: Maximum number of symbols to return
            
        Returns:
            List of (symbol, average_score) tuples
        """
        symbol_scores = []
        
        for symbol in self.score_history:
            avg_score = self.get_average_score(symbol)
            if avg_score is not None:
                symbol_scores.append((symbol, avg_score))
                
        # Sort by average score (descending)
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        
        return symbol_scores[:limit]