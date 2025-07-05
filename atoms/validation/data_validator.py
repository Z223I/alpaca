"""
Data Quality Validation and Anomaly Detection

This module implements comprehensive data validation and anomaly detection
for market data to ensure reliable alert generation and system stability.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import statistics

from atoms.websocket.alpaca_stream import MarketData
from atoms.config.alert_config import config


class ValidationResult(Enum):
    """Result of data validation."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    ANOMALY = "anomaly"


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    STALE_DATA = "stale_data"
    DUPLICATE_DATA = "duplicate_data"
    MISSING_DATA = "missing_data"
    INVALID_RANGE = "invalid_range"
    STATISTICAL_OUTLIER = "statistical_outlier"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    symbol: str
    timestamp: datetime
    issue_type: AnomalyType
    severity: ValidationResult
    message: str
    current_value: Any
    expected_range: Optional[Tuple[float, float]] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationStats:
    """Statistics for data validation."""
    total_validations: int = 0
    valid_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    anomaly_count: int = 0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_validations == 0:
            return 0.0
        return (self.error_count + self.anomaly_count) / self.total_validations
    
    @property
    def quality_score(self) -> float:
        """Calculate overall data quality score (0-1)."""
        if self.total_validations == 0:
            return 1.0
        return self.valid_count / self.total_validations


class DataValidator:
    """
    Comprehensive data validation and anomaly detection system.
    
    Validates market data for:
    - Price and volume ranges
    - Statistical anomalies
    - Data freshness
    - Duplicate detection
    - Missing data detection
    """
    
    def __init__(self, lookback_periods: int = 100):
        """
        Initialize data validator.
        
        Args:
            lookback_periods: Number of periods to keep for statistical analysis
        """
        self.lookback_periods = lookback_periods
        self.historical_data: Dict[str, deque] = {}
        self.validation_issues: List[ValidationIssue] = []
        self.validation_stats = ValidationStats()
        self.last_seen_data: Dict[str, MarketData] = {}
        
        # Configuration
        self.price_change_threshold = 0.20  # 20% price change threshold
        self.volume_multiplier_threshold = 10.0  # 10x volume threshold
        self.stale_data_threshold = 300  # 5 minutes
        self.statistical_outlier_threshold = 3.0  # 3 standard deviations
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Valid ranges for market data
        self.valid_ranges = {
            'price': (0.01, 10000.0),  # $0.01 to $10,000
            'volume': (0, 1000000000),  # 0 to 1 billion shares
            'bid': (0.01, 10000.0),
            'ask': (0.01, 10000.0),
            'bid_size': (0, 1000000),
            'ask_size': (0, 1000000)
        }
    
    def validate_market_data(self, market_data: MarketData) -> List[ValidationIssue]:
        """
        Validate market data and detect anomalies.
        
        Args:
            market_data: Market data to validate
            
        Returns:
            List of validation issues found
        """
        issues = []
        symbol = market_data.symbol
        
        # Update validation stats
        self.validation_stats.total_validations += 1
        
        # Basic range validation
        issues.extend(self._validate_ranges(market_data))
        
        # Stale data detection
        issues.extend(self._detect_stale_data(market_data))
        
        # Duplicate data detection
        issues.extend(self._detect_duplicate_data(market_data))
        
        # Statistical anomaly detection
        issues.extend(self._detect_statistical_anomalies(market_data))
        
        # Price gap detection
        issues.extend(self._detect_price_gaps(market_data))
        
        # Volume spike detection
        issues.extend(self._detect_volume_spikes(market_data))
        
        # Update historical data
        self._update_historical_data(market_data)
        
        # Update last seen data
        self.last_seen_data[symbol] = market_data
        
        # Store issues
        self.validation_issues.extend(issues)
        
        # Update statistics
        if issues:
            has_error = any(issue.severity == ValidationResult.ERROR for issue in issues)
            has_anomaly = any(issue.severity == ValidationResult.ANOMALY for issue in issues)
            has_warning = any(issue.severity == ValidationResult.WARNING for issue in issues)
            
            if has_error:
                self.validation_stats.error_count += 1
            elif has_anomaly:
                self.validation_stats.anomaly_count += 1
            elif has_warning:
                self.validation_stats.warning_count += 1
            else:
                self.validation_stats.valid_count += 1
        else:
            self.validation_stats.valid_count += 1
        
        # Log issues
        for issue in issues:
            if issue.severity == ValidationResult.ERROR:
                self.logger.error(f"Data validation error for {symbol}: {issue.message}")
            elif issue.severity == ValidationResult.ANOMALY:
                self.logger.warning(f"Data anomaly detected for {symbol}: {issue.message}")
            elif issue.severity == ValidationResult.WARNING:
                self.logger.info(f"Data warning for {symbol}: {issue.message}")
        
        return issues
    
    def _validate_ranges(self, market_data: MarketData) -> List[ValidationIssue]:
        """Validate that data values are within expected ranges."""
        issues = []
        
        # Price validation
        if hasattr(market_data, 'price') and market_data.price is not None:
            price_range = self.valid_ranges['price']
            if not (price_range[0] <= market_data.price <= price_range[1]):
                issues.append(ValidationIssue(
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    issue_type=AnomalyType.INVALID_RANGE,
                    severity=ValidationResult.ERROR,
                    message=f"Price {market_data.price} outside valid range {price_range}",
                    current_value=market_data.price,
                    expected_range=price_range
                ))
        
        # Volume validation
        if hasattr(market_data, 'volume') and market_data.volume is not None:
            volume_range = self.valid_ranges['volume']
            if not (volume_range[0] <= market_data.volume <= volume_range[1]):
                issues.append(ValidationIssue(
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    issue_type=AnomalyType.INVALID_RANGE,
                    severity=ValidationResult.ERROR,
                    message=f"Volume {market_data.volume} outside valid range {volume_range}",
                    current_value=market_data.volume,
                    expected_range=volume_range
                ))
        
        return issues
    
    def _detect_stale_data(self, market_data: MarketData) -> List[ValidationIssue]:
        """Detect stale data based on timestamp."""
        issues = []
        
        current_time = datetime.now()
        data_age = (current_time - market_data.timestamp).total_seconds()
        
        if data_age > self.stale_data_threshold:
            severity = ValidationResult.ERROR if data_age > self.stale_data_threshold * 2 else ValidationResult.WARNING
            issues.append(ValidationIssue(
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                issue_type=AnomalyType.STALE_DATA,
                severity=severity,
                message=f"Data is {data_age:.1f} seconds old (threshold: {self.stale_data_threshold}s)",
                current_value=data_age,
                context={'threshold': self.stale_data_threshold}
            ))
        
        return issues
    
    def _detect_duplicate_data(self, market_data: MarketData) -> List[ValidationIssue]:
        """Detect duplicate data points."""
        issues = []
        symbol = market_data.symbol
        
        if symbol in self.last_seen_data:
            last_data = self.last_seen_data[symbol]
            
            # Check if this is exactly the same data point
            if (hasattr(market_data, 'price') and hasattr(last_data, 'price') and
                market_data.price == last_data.price and
                market_data.volume == last_data.volume and
                market_data.timestamp == last_data.timestamp):
                
                issues.append(ValidationIssue(
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    issue_type=AnomalyType.DUPLICATE_DATA,
                    severity=ValidationResult.WARNING,
                    message="Duplicate data point detected",
                    current_value=market_data.price,
                    context={'last_timestamp': last_data.timestamp}
                ))
        
        return issues
    
    def _detect_statistical_anomalies(self, market_data: MarketData) -> List[ValidationIssue]:
        """Detect statistical anomalies based on historical data."""
        issues = []
        symbol = market_data.symbol
        
        if symbol not in self.historical_data or len(self.historical_data[symbol]) < 10:
            return issues  # Need more data for statistical analysis
        
        historical_prices = [data.price for data in self.historical_data[symbol] if hasattr(data, 'price')]
        if len(historical_prices) < 10:
            return issues
        
        # Calculate Z-score for current price
        mean_price = statistics.mean(historical_prices)
        std_price = statistics.stdev(historical_prices)
        
        if std_price > 0:
            z_score = abs(market_data.price - mean_price) / std_price
            
            if z_score > self.statistical_outlier_threshold:
                severity = ValidationResult.ANOMALY if z_score > self.statistical_outlier_threshold * 1.5 else ValidationResult.WARNING
                issues.append(ValidationIssue(
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    issue_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    message=f"Price {market_data.price} is {z_score:.2f} standard deviations from mean {mean_price:.2f}",
                    current_value=market_data.price,
                    context={'z_score': z_score, 'mean': mean_price, 'std': std_price}
                ))
        
        return issues
    
    def _detect_price_gaps(self, market_data: MarketData) -> List[ValidationIssue]:
        """Detect significant price gaps."""
        issues = []
        symbol = market_data.symbol
        
        if symbol in self.last_seen_data:
            last_data = self.last_seen_data[symbol]
            
            if hasattr(market_data, 'price') and hasattr(last_data, 'price'):
                price_change = abs(market_data.price - last_data.price) / last_data.price
                
                if price_change > self.price_change_threshold:
                    severity = ValidationResult.ANOMALY if price_change > self.price_change_threshold * 2 else ValidationResult.WARNING
                    issues.append(ValidationIssue(
                        symbol=market_data.symbol,
                        timestamp=market_data.timestamp,
                        issue_type=AnomalyType.PRICE_GAP,
                        severity=severity,
                        message=f"Price gap of {price_change:.2%} detected (from {last_data.price} to {market_data.price})",
                        current_value=market_data.price,
                        context={'previous_price': last_data.price, 'change_percent': price_change}
                    ))
        
        return issues
    
    def _detect_volume_spikes(self, market_data: MarketData) -> List[ValidationIssue]:
        """Detect volume spikes."""
        issues = []
        symbol = market_data.symbol
        
        if symbol not in self.historical_data or len(self.historical_data[symbol]) < 10:
            return issues
        
        historical_volumes = [data.volume for data in self.historical_data[symbol] if hasattr(data, 'volume')]
        if len(historical_volumes) < 10:
            return issues
        
        avg_volume = statistics.mean(historical_volumes)
        
        if avg_volume > 0:
            volume_ratio = market_data.volume / avg_volume
            
            if volume_ratio > self.volume_multiplier_threshold:
                severity = ValidationResult.ANOMALY if volume_ratio > self.volume_multiplier_threshold * 2 else ValidationResult.WARNING
                issues.append(ValidationIssue(
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    issue_type=AnomalyType.VOLUME_SPIKE,
                    severity=severity,
                    message=f"Volume spike detected: {volume_ratio:.1f}x average volume",
                    current_value=market_data.volume,
                    context={'volume_ratio': volume_ratio, 'avg_volume': avg_volume}
                ))
        
        return issues
    
    def _update_historical_data(self, market_data: MarketData) -> None:
        """Update historical data for statistical analysis."""
        symbol = market_data.symbol
        
        if symbol not in self.historical_data:
            self.historical_data[symbol] = deque(maxlen=self.lookback_periods)
        
        self.historical_data[symbol].append(market_data)
    
    def get_validation_summary(self, symbol: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get validation summary for specified symbol or all symbols.
        
        Args:
            symbol: Symbol to analyze (None for all symbols)
            hours: Hours of history to include
            
        Returns:
            Dictionary with validation summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter issues by time and symbol
        relevant_issues = [
            issue for issue in self.validation_issues
            if issue.timestamp >= cutoff_time and (symbol is None or issue.symbol == symbol)
        ]
        
        # Count by type
        type_counts = {}
        severity_counts = {}
        
        for issue in relevant_issues:
            type_counts[issue.issue_type.value] = type_counts.get(issue.issue_type.value, 0) + 1
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1
        
        # Symbol-specific counts
        symbol_counts = {}
        for issue in relevant_issues:
            symbol_counts[issue.symbol] = symbol_counts.get(issue.symbol, 0) + 1
        
        return {
            'total_issues': len(relevant_issues),
            'issue_types': type_counts,
            'severity_counts': severity_counts,
            'symbol_counts': symbol_counts,
            'validation_stats': {
                'total_validations': self.validation_stats.total_validations,
                'error_rate': self.validation_stats.error_rate,
                'quality_score': self.validation_stats.quality_score
            },
            'time_window_hours': hours
        }
    
    def get_anomaly_report(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get detailed anomaly report.
        
        Args:
            symbol: Symbol to filter by (None for all)
            limit: Maximum number of anomalies to return
            
        Returns:
            List of anomaly dictionaries
        """
        # Filter and sort issues
        filtered_issues = [
            issue for issue in self.validation_issues
            if (symbol is None or issue.symbol == symbol) and
            issue.severity in [ValidationResult.ANOMALY, ValidationResult.ERROR]
        ]
        
        # Sort by timestamp (most recent first)
        filtered_issues.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Convert to dictionary format
        report = []
        for issue in filtered_issues[:limit]:
            report.append({
                'symbol': issue.symbol,
                'timestamp': issue.timestamp.isoformat(),
                'issue_type': issue.issue_type.value,
                'severity': issue.severity.value,
                'message': issue.message,
                'current_value': issue.current_value,
                'expected_range': issue.expected_range,
                'context': issue.context
            })
        
        return report
    
    def is_data_quality_acceptable(self, symbol: str = None, min_quality_score: float = 0.95) -> bool:
        """
        Check if data quality is acceptable for alert generation.
        
        Args:
            symbol: Symbol to check (None for overall)
            min_quality_score: Minimum acceptable quality score
            
        Returns:
            True if data quality is acceptable
        """
        if symbol is None:
            return self.validation_stats.quality_score >= min_quality_score
        
        # Calculate symbol-specific quality score
        symbol_issues = [issue for issue in self.validation_issues if issue.symbol == symbol]
        symbol_validations = len([issue for issue in symbol_issues]) + self.validation_stats.valid_count
        
        if symbol_validations == 0:
            return True  # No data yet, assume acceptable
        
        symbol_errors = len([issue for issue in symbol_issues if issue.severity in [ValidationResult.ERROR, ValidationResult.ANOMALY]])
        symbol_quality = 1.0 - (symbol_errors / symbol_validations)
        
        return symbol_quality >= min_quality_score
    
    def clear_validation_history(self, symbol: str = None) -> None:
        """
        Clear validation history.
        
        Args:
            symbol: Symbol to clear (None for all)
        """
        if symbol is None:
            self.validation_issues.clear()
            self.historical_data.clear()
            self.last_seen_data.clear()
            self.validation_stats = ValidationStats()
        else:
            self.validation_issues = [issue for issue in self.validation_issues if issue.symbol != symbol]
            self.historical_data.pop(symbol, None)
            self.last_seen_data.pop(symbol, None)
        
        self.logger.info(f"Cleared validation history for {symbol or 'all symbols'}")


# Global data validator instance
data_validator = DataValidator()