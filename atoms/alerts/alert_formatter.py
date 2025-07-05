"""
Alert Formatting and Prioritization System

This module handles alert formatting, prioritization, and output generation
for the ORB trading alerts system.
"""

import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .breakout_detector import BreakoutSignal, BreakoutType
from .confidence_scorer import ConfidenceComponents
from ..config.alert_config import config


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
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.orb_range = self.orb_high - self.orb_low
        self.orb_midpoint = (self.orb_high + self.orb_low) / 2.0
        
        # Calculate stop loss and take profit
        self._calculate_risk_levels()
        
        # Generate alert message
        self.alert_message = self._generate_alert_message()
    
    def _calculate_risk_levels(self) -> None:
        """Calculate recommended stop loss and take profit levels."""
        risk_percentage = 0.075  # 7.5% risk
        reward_ratio = 2.0  # 2:1 reward to risk ratio
        
        if self.breakout_type == BreakoutType.BULLISH_BREAKOUT:
            # For bullish breakout, stop loss below ORB low
            self.recommended_stop_loss = self.orb_low * (1 - risk_percentage)
            stop_distance = self.current_price - self.recommended_stop_loss
            self.recommended_take_profit = self.current_price + (stop_distance * reward_ratio)
        elif self.breakout_type == BreakoutType.BEARISH_BREAKDOWN:
            # For bearish breakdown, stop loss above ORB high
            self.recommended_stop_loss = self.orb_high * (1 + risk_percentage)
            stop_distance = self.recommended_stop_loss - self.current_price
            self.recommended_take_profit = self.current_price - (stop_distance * reward_ratio)
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
            
        message = (
            f"[{time_str}] ORB ALERT: {self.symbol} {direction} ${self.current_price:.2f} "
            f"({self.breakout_percentage:+.2f}% vs {vs_level})\n"
            f"Volume: {self.volume_ratio:.1f}x avg | Confidence: {self.confidence_score:.2f} | "
            f"Priority: {self.priority.value}\n"
            f"Stop: ${self.recommended_stop_loss:.2f} | Target: ${self.recommended_take_profit:.2f}"
        )
        
        return message


class AlertFormatter:
    """Alert formatting and output management."""
    
    def __init__(self, output_dir: str = "alerts"):
        """
        Initialize alert formatter.
        
        Args:
            output_dir: Directory for alert output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Alert history
        self.alert_history: List[ORBAlert] = []
        self.daily_alert_count = 0
        
    def create_alert(self, signal: BreakoutSignal, 
                    confidence: ConfidenceComponents) -> ORBAlert:
        """
        Create a formatted alert from breakout signal and confidence data.
        
        Args:
            signal: BreakoutSignal to format
            confidence: ConfidenceComponents for the signal
            
        Returns:
            Formatted ORBAlert
        """
        # Determine priority based on confidence score and volume
        priority = self._calculate_priority(confidence.total_score, signal.volume_ratio)
        
        # Get confidence level description
        confidence_level = self._get_confidence_level(confidence.total_score)
        
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
            alert_message=""  # Will be generated in __post_init__
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
        Format alert for console output.
        
        Args:
            alert: ORBAlert to format
            
        Returns:
            Formatted console string
        """
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
        
        return json.dumps(alert_dict, indent=2)
    
    def save_alert_to_file(self, alert: ORBAlert, file_format: str = "json") -> str:
        """
        Save alert to file.
        
        Args:
            alert: ORBAlert to save
            file_format: Format to save ("json", "csv", "txt")
            
        Returns:
            Path to saved file
        """
        timestamp_str = alert.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{alert.symbol}_{timestamp_str}.{file_format}"
        filepath = self.output_dir / filename
        
        if file_format == "json":
            with open(filepath, 'w') as f:
                f.write(self.format_json_output(alert))
        elif file_format == "csv":
            self._save_alert_csv(alert, filepath)
        elif file_format == "txt":
            with open(filepath, 'w') as f:
                f.write(self.format_console_output(alert))
        
        return str(filepath)
    
    def _save_alert_csv(self, alert: ORBAlert, filepath: Path) -> None:
        """Save alert to CSV format."""
        fieldnames = [
            'symbol', 'timestamp', 'current_price', 'orb_high', 'orb_low',
            'breakout_type', 'breakout_percentage', 'volume_ratio',
            'confidence_score', 'priority', 'recommended_stop_loss',
            'recommended_take_profit'
        ]
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            row = {
                'symbol': alert.symbol,
                'timestamp': alert.timestamp.isoformat(),
                'current_price': alert.current_price,
                'orb_high': alert.orb_high,
                'orb_low': alert.orb_low,
                'breakout_type': alert.breakout_type.value,
                'breakout_percentage': alert.breakout_percentage,
                'volume_ratio': alert.volume_ratio,
                'confidence_score': alert.confidence_score,
                'priority': alert.priority.value,
                'recommended_stop_loss': alert.recommended_stop_loss,
                'recommended_take_profit': alert.recommended_take_profit
            }
            writer.writerow(row)
    
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