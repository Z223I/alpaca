"""
Alert Formatting and Prioritization System

This module handles alert formatting, prioritization, and output generation
for the ORB trading alerts system.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from atoms.alerts.breakout_detector import BreakoutSignal, BreakoutType
from atoms.alerts.confidence_scorer import ConfidenceComponents
from atoms.config.alert_config import config


class AlertPriority(Enum):
    """Alert priority levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


@dataclass
class ORBAlert:
    """Complete ORB alert with all metadata."""
    symbol: str
    timestamp: datetime
    current_price: float
    orb_high: float
    orb_low: float
    orb_range: float
    orb_midpoint: float
    breakout_type: BreakoutType
    breakout_percentage: float
    volume_ratio: float
    confidence_score: float
    priority: AlertPriority
    confidence_level: str
    recommended_stop_loss: float
    recommended_take_profit: float
    alert_message: str
    # EMA Technical Indicators
    ema_9: Optional[float] = None
    ema_20: Optional[float] = None
    ema_9_above_20: Optional[bool] = None
    ema_9_below_20: Optional[bool] = None
    ema_divergence: Optional[float] = None
    # Current candlestick OHLC data
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[int] = None

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.orb_range = self.orb_high - self.orb_low
        self.orb_midpoint = (self.orb_high + self.orb_low) / 2.0

        # Calculate stop loss and take profit
        self._calculate_risk_levels()

        # Generate alert message
        self.alert_message = self._generate_alert_message()

    def _calculate_risk_levels(self) -> None:
        """Calculate recommended stop loss and take profit levels using config percentages."""
        # Use configuration percentages
        stop_loss_percentage = config.stop_loss_percent / 100.0  # Convert to decimal
        take_profit_percentage = config.take_profit_percent / 100.0  # Convert to decimal

        if self.breakout_type == BreakoutType.BULLISH_BREAKOUT:
            # For bullish breakout, stop loss below current price
            self.recommended_stop_loss = self.current_price * (1 - stop_loss_percentage)
            # Take profit as percentage above current price
            self.recommended_take_profit = self.current_price * (1 + take_profit_percentage)
        elif self.breakout_type == BreakoutType.BEARISH_BREAKDOWN:
            # For bearish breakdown, stop loss above current price
            self.recommended_stop_loss = self.current_price * (1 + stop_loss_percentage)
            # Take profit as percentage below current price
            self.recommended_take_profit = self.current_price * (1 - take_profit_percentage)
        else:
            self.recommended_stop_loss = self.current_price
            self.recommended_take_profit = self.current_price

    def _generate_alert_message(self) -> str:
        """Generate formatted alert message."""
        time_str = self.timestamp.strftime("%H:%M:%S")

        if self.breakout_type == BreakoutType.BULLISH_BREAKOUT:
            direction = "↑"
            vs_level = "ORB High"
        elif self.breakout_type == BreakoutType.BEARISH_BREAKDOWN:
            direction = "↓"
            vs_level = "ORB Low"
        else:
            direction = "→"
            vs_level = "ORB Range"

        # Build EMA information string
        ema_info = ""
        if self.ema_9 is not None:
            ema_info += f" | EMA9: ${self.ema_9:.2f}"
        if self.ema_20 is not None:
            ema_info += f" | EMA20: ${self.ema_20:.2f}"
        if self.ema_9_above_20 is not None:
            ema_trend = "↑" if self.ema_9_above_20 else "↓"
            ema_info += f" | EMA9{ema_trend}EMA20"
        if self.ema_divergence is not None:
            ema_info += f" | Div: {self.ema_divergence*100:+.1f}%"

        message = (
            f"[{time_str}] ORB ALERT: {self.symbol} {direction} ${self.current_price:.2f} "
            f"({self.breakout_percentage:+.2f}% vs {vs_level})\n"
            f"Volume: {self.volume_ratio:.1f}x avg | Confidence: {self.confidence_score:.2f} | "
            f"Priority: {self.priority.value}{ema_info}\n"
            f"Stop: ${self.recommended_stop_loss:.2f} | Target: ${self.recommended_take_profit:.2f}"
        )

        return message


class AlertFormatter:
    """Alert formatting and output management."""

    def __init__(self):
        """Initialize alert formatter."""
        # Alert history
        self.alert_history: List[ORBAlert] = []
        self.daily_alert_count = 0

    def create_alert(self, signal: BreakoutSignal, 
                    confidence: ConfidenceComponents, 
                    technical_indicators: Optional[Dict[str, Any]] = None) -> ORBAlert:
        """
        Create a formatted alert from breakout signal and confidence data.

        Args:
            signal: BreakoutSignal to format
            confidence: ConfidenceComponents for the signal
            technical_indicators: Dictionary of technical indicators

        Returns:
            Formatted ORBAlert
        """
        # Determine priority based on confidence score and volume
        priority = self._calculate_priority(confidence.total_score, signal.volume_ratio)

        # Get confidence level description
        confidence_level = self._get_confidence_level(confidence.total_score)

        # Extract EMA indicators from technical_indicators if available
        ema_9 = None
        ema_20 = None
        ema_9_above_20 = None
        ema_9_below_20 = None
        ema_divergence = None

        # Extract candlestick OHLC data from technical_indicators if available
        open_price = None
        high_price = None
        low_price = None
        close_price = None
        volume = None

        if technical_indicators:
            ema_9 = technical_indicators.get('ema_9')
            ema_20 = technical_indicators.get('ema_20')
            ema_9_above_20 = technical_indicators.get('ema_9_above_20')
            ema_9_below_20 = technical_indicators.get('ema_9_below_20')
            ema_divergence = technical_indicators.get('ema_divergence')

            # Extract candlestick data
            open_price = technical_indicators.get('open_price')
            high_price = technical_indicators.get('high_price')
            low_price = technical_indicators.get('low_price')
            close_price = technical_indicators.get('close_price')
            volume = technical_indicators.get('volume')

        alert = ORBAlert(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            current_price=signal.current_price,
            orb_high=signal.orb_high,
            orb_low=signal.orb_low,
            orb_range=0.0,  # Will be calculated in __post_init__
            orb_midpoint=0.0,  # Will be calculated in __post_init__
            breakout_type=signal.breakout_type,
            breakout_percentage=signal.breakout_percentage,
            volume_ratio=signal.volume_ratio,
            confidence_score=confidence.total_score,
            priority=priority,
            confidence_level=confidence_level,
            recommended_stop_loss=0.0,  # Will be calculated in __post_init__
            recommended_take_profit=0.0,  # Will be calculated in __post_init__
            alert_message="",  # Will be generated in __post_init__
            # EMA Technical Indicators
            ema_9=ema_9,
            ema_20=ema_20,
            ema_9_above_20=ema_9_above_20,
            ema_9_below_20=ema_9_below_20,
            ema_divergence=ema_divergence,
            # Current candlestick OHLC data
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume
        )

        # Add to history
        self.alert_history.append(alert)
        self.daily_alert_count += 1

        # Limit history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        return alert

    def _calculate_priority(self, confidence_score: float, volume_ratio: float) -> AlertPriority:
        """
        Calculate alert priority based on confidence score and volume.

        Args:
            confidence_score: Confidence score (0.0 to 1.0)
            volume_ratio: Volume ratio vs average

        Returns:
            AlertPriority enum
        """
        # High priority: High confidence + high volume
        if confidence_score >= 0.85 and volume_ratio >= 2.0:
            return AlertPriority.HIGH

        # Medium priority: Good confidence or high volume
        elif confidence_score >= 0.70 or volume_ratio >= 2.0:
            return AlertPriority.MEDIUM

        # Low priority: Moderate confidence
        elif confidence_score >= 0.60:
            return AlertPriority.LOW

        # Very low priority: Low confidence
        else:
            return AlertPriority.VERY_LOW

    def _get_confidence_level(self, confidence_score: float) -> str:
        """
        Get confidence level description.

        Args:
            confidence_score: Confidence score (0.0 to 1.0)

        Returns:
            Confidence level string
        """
        if confidence_score >= 0.85:
            return "VERY_HIGH"
        elif confidence_score >= 0.70:
            return "HIGH"
        elif confidence_score >= 0.60:
            return "MEDIUM"
        elif confidence_score >= 0.50:
            return "LOW"
        else:
            return "VERY_LOW"

    def format_console_output(self, alert: ORBAlert) -> str:
        """
        Format alert for console output with color coding.

        Args:
            alert: ORBAlert to format

        Returns:
            Formatted console string with color codes
        """
        # ANSI color codes
        RED = "\033[31m"      # Red for bearish
        GREEN = "\033[32m"    # Green for bullish
        RESET = "\033[0m"     # Reset color

        # Choose color based on breakout type
        if alert.breakout_type == BreakoutType.BULLISH_BREAKOUT:
            color = GREEN
        elif alert.breakout_type == BreakoutType.BEARISH_BREAKDOWN:
            color = RED
        else:
            color = ""  # No color for other types

        # Apply color to the entire alert message
        if color:
            return f"{color}{alert.alert_message}{RESET}"
        else:
            return alert.alert_message

    def format_json_output(self, alert: ORBAlert) -> str:
        """
        Format alert as JSON string.

        Args:
            alert: ORBAlert to format

        Returns:
            JSON formatted string
        """
        alert_dict = asdict(alert)

        # Convert datetime to ISO format
        alert_dict['timestamp'] = alert.timestamp.isoformat()

        # Convert enums to strings
        alert_dict['breakout_type'] = alert.breakout_type.value
        alert_dict['priority'] = alert.priority.value

        # Ensure boolean values are properly serializable
        if 'ema_9_above_20' in alert_dict:
            alert_dict['ema_9_above_20'] = bool(alert_dict['ema_9_above_20']) if alert_dict['ema_9_above_20'] is not None else None
        if 'ema_9_below_20' in alert_dict:
            alert_dict['ema_9_below_20'] = bool(alert_dict['ema_9_below_20']) if alert_dict['ema_9_below_20'] is not None else None

        return json.dumps(alert_dict, indent=2)


    def get_alerts_by_priority(self, priority: AlertPriority) -> List[ORBAlert]:
        """
        Get alerts filtered by priority.

        Args:
            priority: AlertPriority to filter by

        Returns:
            List of ORBAlert objects
        """
        return [alert for alert in self.alert_history if alert.priority == priority]

    def get_alerts_by_symbol(self, symbol: str) -> List[ORBAlert]:
        """
        Get alerts filtered by symbol.

        Args:
            symbol: Symbol to filter by

        Returns:
            List of ORBAlert objects
        """
        return [alert for alert in self.alert_history if alert.symbol == symbol]

    def get_recent_alerts(self, limit: int = 10) -> List[ORBAlert]:
        """
        Get most recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of most recent ORBAlert objects
        """
        return sorted(self.alert_history, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Get daily alert summary statistics.

        Returns:
            Dictionary with daily summary data
        """
        today = datetime.now().date()
        today_alerts = [a for a in self.alert_history if a.timestamp.date() == today]

        priority_counts = {}
        for priority in AlertPriority:
            priority_counts[priority.value] = len([a for a in today_alerts if a.priority == priority])

        symbol_counts = {}
        for alert in today_alerts:
            symbol_counts[alert.symbol] = symbol_counts.get(alert.symbol, 0) + 1

        return {
            'date': today.isoformat(),
            'total_alerts': len(today_alerts),
            'priority_breakdown': priority_counts,
            'symbol_breakdown': symbol_counts,
            'avg_confidence': sum(a.confidence_score for a in today_alerts) / len(today_alerts) if today_alerts else 0,
            'max_confidence': max(a.confidence_score for a in today_alerts) if today_alerts else 0
        }

    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        self.daily_alert_count = 0