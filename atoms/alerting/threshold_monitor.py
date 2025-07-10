"""
Threshold monitoring atoms for performance-based alerting.

This atom provides comprehensive threshold monitoring capabilities
for performance metrics, system health, and trading outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
from dataclasses import dataclass, field
import logging


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class ThresholdType(Enum):
    """Types of threshold comparisons."""
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUAL_TO = "equal_to"
    NOT_EQUAL_TO = "not_equal_to"
    RANGE = "range"
    PERCENTAGE_CHANGE = "percentage_change"
    MOVING_AVERAGE = "moving_average"


@dataclass
class ThresholdRule:
    """Configuration for a threshold monitoring rule."""
    name: str
    metric_name: str
    threshold_type: ThresholdType
    threshold_value: Union[float, Tuple[float, float]]
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_minutes: int = 5
    consecutive_violations: int = 1
    window_minutes: int = 1
    
    # Optional callback function
    callback: Optional[Callable] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class ThresholdViolation:
    """Represents a threshold violation event."""
    rule_name: str
    metric_name: str
    current_value: float
    threshold_value: Union[float, Tuple[float, float]]
    threshold_type: ThresholdType
    severity: AlertSeverity
    description: str
    timestamp: datetime
    consecutive_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThresholdMonitor:
    """
    Monitor metrics against configured thresholds and generate alerts.
    
    This class provides flexible threshold monitoring with support for
    different comparison types, cooldown periods, and consecutive violations.
    """
    
    def __init__(self, 
                 enable_logging: bool = True,
                 default_cooldown_minutes: int = 5):
        """
        Initialize the threshold monitor.
        
        Args:
            enable_logging: Whether to enable internal logging
            default_cooldown_minutes: Default cooldown period for rules
        """
        self.rules: Dict[str, ThresholdRule] = {}
        self.violations_history: List[ThresholdViolation] = []
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.consecutive_violations: Dict[str, int] = {}
        
        self.default_cooldown_minutes = default_cooldown_minutes
        self.logger = logging.getLogger(__name__) if enable_logging else None
    
    def add_rule(self, rule: ThresholdRule) -> bool:
        """
        Add a new threshold monitoring rule.
        
        Args:
            rule: ThresholdRule configuration
            
        Returns:
            True if rule was added successfully
        """
        if rule.name in self.rules:
            if self.logger:
                self.logger.warning(f"Rule '{rule.name}' already exists, replacing")
        
        self.rules[rule.name] = rule
        if self.logger:
            self.logger.info(f"Added threshold rule: {rule.name}")
        
        return True
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a threshold monitoring rule.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed successfully
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            if rule_name in self.consecutive_violations:
                del self.consecutive_violations[rule_name]
            if self.logger:
                self.logger.info(f"Removed threshold rule: {rule_name}")
            return True
        return False
    
    def update_rule(self, rule_name: str, **kwargs) -> bool:
        """
        Update an existing threshold rule.
        
        Args:
            rule_name: Name of the rule to update
            **kwargs: Rule attributes to update
            
        Returns:
            True if rule was updated successfully
        """
        if rule_name not in self.rules:
            return False
        
        rule = self.rules[rule_name]
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        if self.logger:
            self.logger.info(f"Updated threshold rule: {rule_name}")
        
        return True
    
    def check_metric(self, 
                    metric_name: str, 
                    metric_value: float,
                    timestamp: Optional[datetime] = None) -> List[ThresholdViolation]:
        """
        Check a metric value against all applicable rules.
        
        Args:
            metric_name: Name of the metric
            metric_value: Current value of the metric
            timestamp: Timestamp for the metric (defaults to now)
            
        Returns:
            List of threshold violations
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store metric history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append((timestamp, metric_value))
        
        # Keep only recent history (last 24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        self.metric_history[metric_name] = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts >= cutoff_time
        ]
        
        violations = []
        
        # Check all rules that apply to this metric
        for rule_name, rule in self.rules.items():
            if not rule.enabled or rule.metric_name != metric_name:
                continue
            
            # Check cooldown period
            if rule.last_triggered:
                cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
                if timestamp - rule.last_triggered < cooldown_delta:
                    continue
            
            # Check threshold
            violation = self._check_threshold(rule, metric_value, timestamp)
            if violation:
                violations.append(violation)
        
        return violations
    
    def check_multiple_metrics(self, 
                             metrics: Dict[str, float],
                             timestamp: Optional[datetime] = None) -> List[ThresholdViolation]:
        """
        Check multiple metrics against their rules.
        
        Args:
            metrics: Dictionary of metric names and values
            timestamp: Timestamp for the metrics (defaults to now)
            
        Returns:
            List of all threshold violations
        """
        all_violations = []
        
        for metric_name, metric_value in metrics.items():
            violations = self.check_metric(metric_name, metric_value, timestamp)
            all_violations.extend(violations)
        
        return all_violations
    
    def get_rule_status(self, rule_name: str) -> Dict[str, Any]:
        """
        Get status information for a specific rule.
        
        Args:
            rule_name: Name of the rule
            
        Returns:
            Dictionary with rule status
        """
        if rule_name not in self.rules:
            return {'error': 'Rule not found'}
        
        rule = self.rules[rule_name]
        
        # Get recent violations for this rule
        recent_violations = [
            v for v in self.violations_history
            if v.rule_name == rule_name and 
            v.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        # Get recent metric values
        recent_metrics = []
        if rule.metric_name in self.metric_history:
            recent_metrics = self.metric_history[rule.metric_name][-10:]
        
        return {
            'rule_name': rule.name,
            'metric_name': rule.metric_name,
            'enabled': rule.enabled,
            'threshold_type': rule.threshold_type.value,
            'threshold_value': rule.threshold_value,
            'severity': rule.severity.value,
            'trigger_count': rule.trigger_count,
            'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
            'recent_violations': len(recent_violations),
            'consecutive_violations': self.consecutive_violations.get(rule_name, 0),
            'recent_metric_values': [(ts.isoformat(), val) for ts, val in recent_metrics]
        }
    
    def get_all_violations(self, 
                          hours_back: int = 24,
                          severity: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """
        Get all violations within the specified time period.
        
        Args:
            hours_back: Number of hours to look back
            severity: Filter by severity level (optional)
            
        Returns:
            List of violation dictionaries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        violations = [
            v for v in self.violations_history
            if v.timestamp >= cutoff_time and 
            (severity is None or v.severity == severity)
        ]
        
        return [
            {
                'rule_name': v.rule_name,
                'metric_name': v.metric_name,
                'current_value': v.current_value,
                'threshold_value': v.threshold_value,
                'threshold_type': v.threshold_type.value,
                'severity': v.severity.value,
                'description': v.description,
                'timestamp': v.timestamp.isoformat(),
                'consecutive_count': v.consecutive_count,
                'metadata': v.metadata
            }
            for v in violations
        ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all monitored metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        summary = {
            'total_rules': len(self.rules),
            'active_rules': len([r for r in self.rules.values() if r.enabled]),
            'metrics_monitored': len(self.metric_history),
            'total_violations_24h': len([
                v for v in self.violations_history
                if v.timestamp >= datetime.now() - timedelta(hours=24)
            ]),
            'violations_by_severity': {},
            'rules_by_metric': {}
        }
        
        # Count violations by severity
        for severity in AlertSeverity:
            count = len([
                v for v in self.violations_history
                if v.severity == severity and 
                v.timestamp >= datetime.now() - timedelta(hours=24)
            ])
            summary['violations_by_severity'][severity.value] = count
        
        # Count rules by metric
        for rule in self.rules.values():
            metric = rule.metric_name
            if metric not in summary['rules_by_metric']:
                summary['rules_by_metric'][metric] = 0
            summary['rules_by_metric'][metric] += 1
        
        return summary
    
    def _check_threshold(self, 
                        rule: ThresholdRule, 
                        metric_value: float,
                        timestamp: datetime) -> Optional[ThresholdViolation]:
        """
        Check if a metric value violates a threshold rule.
        
        Args:
            rule: ThresholdRule to check
            metric_value: Current metric value
            timestamp: Timestamp of the metric
            
        Returns:
            ThresholdViolation if threshold is violated, None otherwise
        """
        violated = False
        
        if rule.threshold_type == ThresholdType.GREATER_THAN:
            violated = metric_value > rule.threshold_value
        elif rule.threshold_type == ThresholdType.LESS_THAN:
            violated = metric_value < rule.threshold_value
        elif rule.threshold_type == ThresholdType.EQUAL_TO:
            violated = abs(metric_value - rule.threshold_value) < 0.001
        elif rule.threshold_type == ThresholdType.NOT_EQUAL_TO:
            violated = abs(metric_value - rule.threshold_value) >= 0.001
        elif rule.threshold_type == ThresholdType.RANGE:
            if isinstance(rule.threshold_value, tuple) and len(rule.threshold_value) == 2:
                min_val, max_val = rule.threshold_value
                violated = metric_value < min_val or metric_value > max_val
        elif rule.threshold_type == ThresholdType.PERCENTAGE_CHANGE:
            violated = self._check_percentage_change(rule, metric_value, timestamp)
        elif rule.threshold_type == ThresholdType.MOVING_AVERAGE:
            violated = self._check_moving_average(rule, metric_value, timestamp)
        
        if not violated:
            # Reset consecutive violations counter
            if rule.name in self.consecutive_violations:
                self.consecutive_violations[rule.name] = 0
            return None
        
        # Update consecutive violations counter
        self.consecutive_violations[rule.name] = self.consecutive_violations.get(rule.name, 0) + 1
        
        # Check if we need consecutive violations
        if self.consecutive_violations[rule.name] < rule.consecutive_violations:
            return None
        
        # Create violation
        violation = ThresholdViolation(
            rule_name=rule.name,
            metric_name=rule.metric_name,
            current_value=metric_value,
            threshold_value=rule.threshold_value,
            threshold_type=rule.threshold_type,
            severity=rule.severity,
            description=rule.description,
            timestamp=timestamp,
            consecutive_count=self.consecutive_violations[rule.name]
        )
        
        # Update rule tracking
        rule.last_triggered = timestamp
        rule.trigger_count += 1
        
        # Store violation
        self.violations_history.append(violation)
        
        # Execute callback if provided
        if rule.callback:
            try:
                rule.callback(violation)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error executing callback for rule {rule.name}: {e}")
        
        if self.logger:
            self.logger.warning(f"Threshold violation: {rule.name} - {violation.description}")
        
        return violation
    
    def _check_percentage_change(self, 
                               rule: ThresholdRule, 
                               metric_value: float,
                               timestamp: datetime) -> bool:
        """Check percentage change threshold."""
        if rule.metric_name not in self.metric_history:
            return False
        
        history = self.metric_history[rule.metric_name]
        if len(history) < 2:
            return False
        
        # Get previous value
        previous_value = history[-2][1]
        
        if previous_value == 0:
            return False
        
        percentage_change = ((metric_value - previous_value) / previous_value) * 100
        return abs(percentage_change) > rule.threshold_value
    
    def _check_moving_average(self, 
                            rule: ThresholdRule, 
                            metric_value: float,
                            timestamp: datetime) -> bool:
        """Check moving average threshold."""
        if rule.metric_name not in self.metric_history:
            return False
        
        history = self.metric_history[rule.metric_name]
        if len(history) < rule.window_minutes:
            return False
        
        # Calculate moving average
        recent_values = [val for _, val in history[-rule.window_minutes:]]
        moving_average = sum(recent_values) / len(recent_values)
        
        # Check if current value deviates from moving average
        deviation = abs(metric_value - moving_average)
        return deviation > rule.threshold_value
    
    def export_configuration(self, filepath: str) -> bool:
        """
        Export threshold rules configuration to JSON file.
        
        Args:
            filepath: Path to output file
            
        Returns:
            True if export successful
        """
        try:
            config_data = {
                'rules': {},
                'export_timestamp': datetime.now().isoformat()
            }
            
            for rule_name, rule in self.rules.items():
                config_data['rules'][rule_name] = {
                    'name': rule.name,
                    'metric_name': rule.metric_name,
                    'threshold_type': rule.threshold_type.value,
                    'threshold_value': rule.threshold_value,
                    'severity': rule.severity.value,
                    'description': rule.description,
                    'enabled': rule.enabled,
                    'cooldown_minutes': rule.cooldown_minutes,
                    'consecutive_violations': rule.consecutive_violations,
                    'window_minutes': rule.window_minutes,
                    'created_at': rule.created_at.isoformat(),
                    'trigger_count': rule.trigger_count
                }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_configuration(self, filepath: str) -> bool:
        """
        Import threshold rules configuration from JSON file.
        
        Args:
            filepath: Path to input file
            
        Returns:
            True if import successful
        """
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            for rule_name, rule_config in config_data.get('rules', {}).items():
                rule = ThresholdRule(
                    name=rule_config['name'],
                    metric_name=rule_config['metric_name'],
                    threshold_type=ThresholdType(rule_config['threshold_type']),
                    threshold_value=rule_config['threshold_value'],
                    severity=AlertSeverity(rule_config['severity']),
                    description=rule_config['description'],
                    enabled=rule_config.get('enabled', True),
                    cooldown_minutes=rule_config.get('cooldown_minutes', 5),
                    consecutive_violations=rule_config.get('consecutive_violations', 1),
                    window_minutes=rule_config.get('window_minutes', 1),
                    created_at=datetime.fromisoformat(rule_config['created_at']),
                    trigger_count=rule_config.get('trigger_count', 0)
                )
                
                self.rules[rule_name] = rule
            
            if self.logger:
                self.logger.info(f"Imported {len(config_data.get('rules', {}))} threshold rules")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error importing configuration: {e}")
            return False


def create_performance_rules() -> List[ThresholdRule]:
    """
    Create standard performance monitoring rules.
    
    Returns:
        List of ThresholdRule objects for common performance metrics
    """
    rules = []
    
    # Success rate monitoring
    rules.append(ThresholdRule(
        name="success_rate_critical",
        metric_name="success_rate",
        threshold_type=ThresholdType.LESS_THAN,
        threshold_value=30.0,
        severity=AlertSeverity.CRITICAL,
        description="Success rate dropped below 30%",
        consecutive_violations=2,
        cooldown_minutes=10
    ))
    
    rules.append(ThresholdRule(
        name="success_rate_warning",
        metric_name="success_rate",
        threshold_type=ThresholdType.LESS_THAN,
        threshold_value=50.0,
        severity=AlertSeverity.WARNING,
        description="Success rate dropped below 50%",
        consecutive_violations=3,
        cooldown_minutes=15
    ))
    
    # Return volatility monitoring
    rules.append(ThresholdRule(
        name="high_volatility",
        metric_name="return_volatility",
        threshold_type=ThresholdType.GREATER_THAN,
        threshold_value=15.0,
        severity=AlertSeverity.WARNING,
        description="Return volatility exceeds 15%",
        consecutive_violations=2,
        cooldown_minutes=30
    ))
    
    # Large loss monitoring
    rules.append(ThresholdRule(
        name="large_loss",
        metric_name="worst_return",
        threshold_type=ThresholdType.LESS_THAN,
        threshold_value=-20.0,
        severity=AlertSeverity.CRITICAL,
        description="Large loss detected (>20%)",
        consecutive_violations=1,
        cooldown_minutes=5
    ))
    
    # System resource monitoring
    rules.append(ThresholdRule(
        name="high_memory_usage",
        metric_name="memory_usage_percent",
        threshold_type=ThresholdType.GREATER_THAN,
        threshold_value=85.0,
        severity=AlertSeverity.WARNING,
        description="Memory usage above 85%",
        consecutive_violations=3,
        cooldown_minutes=10
    ))
    
    rules.append(ThresholdRule(
        name="high_cpu_usage",
        metric_name="cpu_usage_percent",
        threshold_type=ThresholdType.GREATER_THAN,
        threshold_value=80.0,
        severity=AlertSeverity.WARNING,
        description="CPU usage above 80%",
        consecutive_violations=5,
        cooldown_minutes=15
    ))
    
    return rules


def create_trading_rules() -> List[ThresholdRule]:
    """
    Create trading-specific monitoring rules.
    
    Returns:
        List of ThresholdRule objects for trading metrics
    """
    rules = []
    
    # Drawdown monitoring
    rules.append(ThresholdRule(
        name="max_drawdown_critical",
        metric_name="max_drawdown",
        threshold_type=ThresholdType.GREATER_THAN,
        threshold_value=25.0,
        severity=AlertSeverity.CRITICAL,
        description="Maximum drawdown exceeds 25%",
        consecutive_violations=1,
        cooldown_minutes=60
    ))
    
    # Sharpe ratio monitoring
    rules.append(ThresholdRule(
        name="low_sharpe_ratio",
        metric_name="sharpe_ratio",
        threshold_type=ThresholdType.LESS_THAN,
        threshold_value=0.5,
        severity=AlertSeverity.WARNING,
        description="Sharpe ratio below 0.5",
        consecutive_violations=5,
        cooldown_minutes=120
    ))
    
    # Alert frequency monitoring
    rules.append(ThresholdRule(
        name="no_alerts_generated",
        metric_name="alerts_per_hour",
        threshold_type=ThresholdType.LESS_THAN,
        threshold_value=0.1,
        severity=AlertSeverity.WARNING,
        description="No alerts generated in the last hour",
        consecutive_violations=1,
        cooldown_minutes=30
    ))
    
    rules.append(ThresholdRule(
        name="excessive_alerts",
        metric_name="alerts_per_hour",
        threshold_type=ThresholdType.GREATER_THAN,
        threshold_value=50.0,
        severity=AlertSeverity.WARNING,
        description="Excessive alert generation (>50/hour)",
        consecutive_violations=2,
        cooldown_minutes=20
    ))
    
    return rules