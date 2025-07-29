"""
ORB Breakout Detection Algorithm

This module implements the core breakout detection logic based on PCA analysis
showing 82.31% variance explained by ORB patterns.
"""

import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from atoms.config.alert_config import config
from atoms.indicators.orb_calculator import ORBLevel, ORBCalculator
from atoms.utils.calculate_ema import calculate_ema


class BreakoutType(Enum):
    """Types of breakout patterns."""
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKDOWN = "bearish_breakdown"
    NO_BREAKOUT = "no_breakout"


@dataclass
class BreakoutSignal:
    """Breakout signal data."""
    symbol: str
    breakout_type: BreakoutType
    current_price: float
    orb_high: float
    orb_low: float
    breakout_percentage: float
    volume_ratio: float
    timestamp: datetime
    confidence_score: float = 0.0
    
    def __post_init__(self):
        """Calculate breakout percentage after initialization."""
        if self.breakout_type == BreakoutType.BULLISH_BREAKOUT:
            self.breakout_percentage = ((self.current_price - self.orb_high) / self.orb_high) * 100
        elif self.breakout_type == BreakoutType.BEARISH_BREAKDOWN:
            self.breakout_percentage = ((self.orb_low - self.current_price) / self.orb_low) * 100
        else:
            self.breakout_percentage = 0.0


class BreakoutDetector:
    """Core breakout detection algorithm based on PCA findings."""
    
    def __init__(self, orb_calculator: ORBCalculator = None):
        """
        Initialize breakout detector.
        
        Args:
            orb_calculator: ORB calculator instance
        """
        self.orb_calculator = orb_calculator or ORBCalculator()
        self.recent_signals: Dict[str, BreakoutSignal] = {}
        
    def detect_breakout(self, symbol: str, current_price: float, 
                       volume_ratio: float, timestamp: datetime = None) -> Optional[BreakoutSignal]:
        """
        Detect breakout based on current price and volume data.
        
        Args:
            symbol: Trading symbol
            current_price: Current stock price
            volume_ratio: Current volume vs average volume
            timestamp: Current timestamp
            
        Returns:
            BreakoutSignal if breakout detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Get ORB levels for the symbol
        orb_level = self.orb_calculator.get_orb_level(symbol)
        if orb_level is None:
            return None
            
        # Check for bullish breakout
        if self._is_bullish_breakout(current_price, orb_level, volume_ratio):
            return BreakoutSignal(
                symbol=symbol,
                breakout_type=BreakoutType.BULLISH_BREAKOUT,
                current_price=current_price,
                orb_high=orb_level.orb_high,
                orb_low=orb_level.orb_low,
                breakout_percentage=0.0,  # Will be calculated in __post_init__
                volume_ratio=volume_ratio,
                timestamp=timestamp
            )
            
        # Check for bearish breakdown
        elif self._is_bearish_breakdown(current_price, orb_level, volume_ratio):
            return BreakoutSignal(
                symbol=symbol,
                breakout_type=BreakoutType.BEARISH_BREAKDOWN,
                current_price=current_price,
                orb_high=orb_level.orb_high,
                orb_low=orb_level.orb_low,
                breakout_percentage=0.0,  # Will be calculated in __post_init__
                volume_ratio=volume_ratio,
                timestamp=timestamp
            )
            
        return None
    
    def _is_bullish_breakout(self, current_price: float, orb_level: ORBLevel, 
                           volume_ratio: float) -> bool:
        """
        Check if current conditions indicate a bullish breakout.
        
        Args:
            current_price: Current stock price
            orb_level: ORB level data
            volume_ratio: Volume ratio vs average
            
        Returns:
            True if bullish breakout detected
        """
        # Price must be above ORB high + threshold
        breakout_threshold = orb_level.get_breakout_threshold(config.breakout_threshold)
        if current_price < breakout_threshold:
            return False
            
        # Volume must meet minimum requirement
        if volume_ratio < config.volume_multiplier:
            return False
            
        return True
    
    def _is_bearish_breakdown(self, current_price: float, orb_level: ORBLevel, 
                            volume_ratio: float) -> bool:
        """
        Check if current conditions indicate a bearish breakdown.
        
        Args:
            current_price: Current stock price
            orb_level: ORB level data
            volume_ratio: Volume ratio vs average
            
        Returns:
            True if bearish breakdown detected
        """
        # Price must be below ORB low - threshold
        breakdown_threshold = orb_level.get_breakdown_threshold(config.breakout_threshold)
        if current_price > breakdown_threshold:
            return False
            
        # Volume must meet minimum requirement
        if volume_ratio < config.volume_multiplier:
            return False
            
        return True
    
    def calculate_technical_indicators(self, symbol_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical indicators for confidence scoring.
        
        Args:
            symbol_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of technical indicator values
        """
        indicators = {}
        
        if len(symbol_data) < 9:
            return indicators
            
        # Calculate EMA(9) using utility function
        ema9_success, ema9_values = calculate_ema(symbol_data, period=9)
        if ema9_success:
            ema_9 = ema9_values.iloc[-1]
            indicators['ema_9'] = ema_9
        else:
            indicators['ema_9'] = None
            return indicators
        
        # Calculate EMA(20) if we have enough data
        if len(symbol_data) >= 20:
            ema20_success, ema20_values = calculate_ema(symbol_data, period=20)
            if ema20_success:
                ema_20 = ema20_values.iloc[-1]
                indicators['ema_20'] = ema_20
                
                # Calculate EMA9 vs EMA20 relationship
                indicators['ema_9_above_20'] = ema_9 > ema_20
                indicators['ema_9_below_20'] = ema_9 < ema_20
                indicators['ema_divergence'] = (ema_9 - ema_20) / ema_20 if ema_20 > 0 else 0.0
            else:
                # EMA20 calculation failed
                indicators['ema_20'] = None
                indicators['ema_9_above_20'] = None
                indicators['ema_9_below_20'] = None
                indicators['ema_divergence'] = None
        else:
            # Not enough data for EMA20
            indicators['ema_20'] = None
            indicators['ema_9_above_20'] = None
            indicators['ema_9_below_20'] = None
            indicators['ema_divergence'] = None
        
        # Calculate VWAP
        typical_price = (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3
        vwap = (typical_price * symbol_data['volume']).sum() / symbol_data['volume'].sum()
        indicators['vwap'] = vwap
        
        # Calculate current price deviations
        current_price = symbol_data['close'].iloc[-1]
        indicators['ema_deviation'] = abs(current_price - ema_9) / current_price
        indicators['vwap_deviation'] = abs(current_price - vwap) / current_price
        
        # Add current candlestick OHLC data from the latest row
        latest_row = symbol_data.iloc[-1]
        indicators['open_price'] = float(latest_row['open']) if latest_row['open'] is not None else None
        indicators['high_price'] = float(latest_row['high'])
        indicators['low_price'] = float(latest_row['low'])
        indicators['close_price'] = float(latest_row['close'])
        indicators['volume'] = int(latest_row['volume'])
        
        return indicators
    
    def get_recent_signal(self, symbol: str) -> Optional[BreakoutSignal]:
        """
        Get most recent breakout signal for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Most recent BreakoutSignal or None
        """
        return self.recent_signals.get(symbol)
    
    def update_recent_signal(self, signal: BreakoutSignal) -> None:
        """
        Update the most recent signal for a symbol.
        
        Args:
            signal: BreakoutSignal to store
        """
        self.recent_signals[signal.symbol] = signal
    
    def clear_recent_signals(self) -> None:
        """Clear all recent signals."""
        self.recent_signals.clear()
    
    def get_all_recent_signals(self) -> List[BreakoutSignal]:
        """
        Get all recent signals sorted by timestamp.
        
        Returns:
            List of BreakoutSignal objects
        """
        signals = list(self.recent_signals.values())
        return sorted(signals, key=lambda s: s.timestamp, reverse=True)
    
    def is_within_alert_window(self, timestamp: datetime = None) -> bool:
        """
        Check if current time is within alert window.
        
        Args:
            timestamp: Timestamp to check (defaults to now)
            
        Returns:
            True if within alert window
        """
        import pytz
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Handle Eastern Time for alert window check
        if timestamp.tzinfo is None:
            # Timestamps are already timezone-naive Eastern Time (from websocket), use directly
            current_time = timestamp.time()
        else:
            # If timezone-aware, convert to Eastern Time
            et_tz = pytz.timezone('US/Eastern')
            timestamp_et = timestamp.astimezone(et_tz)
            current_time = timestamp_et.time()
        
        # Parse alert window times (in Eastern Time)
        start_time = time(*map(int, config.alert_window_start.split(':')))
        end_time = time(*map(int, config.alert_window_end.split(':')))
        
        return start_time <= current_time <= end_time