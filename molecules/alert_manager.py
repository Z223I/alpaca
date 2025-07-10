"""
Alert Management Molecule.

This molecule provides comprehensive alert management capabilities including
alert generation, escalation, notification routing, and alert lifecycle management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import threading
import time
from pathlib import Path
import logging

# Import atoms
from atoms.alerting.threshold_monitor import (
    ThresholdMonitor, ThresholdRule, ThresholdViolation, 
    AlertSeverity, ThresholdType
)
from atoms.alerting.notification_sender import (
    NotificationSender, NotificationMessage, NotificationChannel,
    NotificationConfig
)


class AlertStatus(Enum):
    """Alert lifecycle status."""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertPriority(Enum):
    """Alert priority levels."""
    P1_CRITICAL = "p1_critical"
    P2_HIGH = "p2_high"
    P3_MEDIUM = "p3_medium"
    P4_LOW = "p4_low"
    P5_INFO = "p5_info"


@dataclass
class AlertRule:
    """Enhanced alert rule with escalation and routing."""
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    priority: AlertPriority
    enabled: bool = True
    
    # Escalation configuration
    escalation_timeout_minutes: int = 30
    escalation_levels: List[str] = field(default_factory=list)
    
    # Notification routing
    notification_channels: List[str] = field(default_factory=list)
    business_hours_only: bool = False
    
    # Suppression rules
    suppression_rules: List[str] = field(default_factory=list)
    max_alerts_per_hour: int = 10
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    runbook_url: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Individual alert instance."""
    id: str
    rule_name: str
    title: str
    description: str
    severity: AlertSeverity
    priority: AlertPriority
    status: AlertStatus
    
    # Timing
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Context
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: Union[float, str] = ""
    
    # Escalation
    escalation_level: int = 0
    escalated_at: Optional[datetime] = None
    
    # Notifications
    notifications_sent: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'triggered_at': self.triggered_at.isoformat(),
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'escalation_level': self.escalation_level,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None,
            'notifications_sent': self.notifications_sent,
            'tags': self.tags,
            'metadata': self.metadata
        }


class AlertManager:
    """
    Comprehensive alert management system.
    
    Provides alert generation, lifecycle management, escalation,
    notification routing, and alert analytics.
    """
    
    def __init__(self, 
                 persistence_path: str = "alerts_data",
                 enable_escalation: bool = True,
                 enable_suppression: bool = True):
        """
        Initialize the alert manager.
        
        Args:
            persistence_path: Path for alert data persistence
            enable_escalation: Whether to enable alert escalation
            enable_suppression: Whether to enable alert suppression
        """
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(exist_ok=True)
        
        self.enable_escalation = enable_escalation
        self.enable_suppression = enable_suppression
        
        # Core components
        self.threshold_monitor = ThresholdMonitor()
        self.notification_sender = NotificationSender()
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Escalation and suppression tracking
        self.escalation_queue: List[str] = []
        self.suppression_counters: Dict[str, List[datetime]] = {}
        
        # Background processing
        self.processing_active = False
        self.processing_thread = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Load persisted data
        self._load_persisted_data()
    
    def start_processing(self):
        """Start background alert processing."""
        if self.processing_active:
            return
        
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Alert manager processing started")
    
    def stop_processing(self):
        """Stop background alert processing."""
        self.processing_active = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        self.logger.info("Alert manager processing stopped")
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """
        Add an alert rule.
        
        Args:
            rule: AlertRule to add
            
        Returns:
            True if rule was added successfully
        """
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
        self._persist_rules()
        return True
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed successfully
        """
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
            self._persist_rules()
            return True
        return False
    
    def update_alert_rule(self, rule_name: str, **kwargs) -> bool:
        """
        Update an alert rule.
        
        Args:
            rule_name: Name of the rule to update
            **kwargs: Rule attributes to update
            
        Returns:
            True if rule was updated successfully
        """
        if rule_name not in self.alert_rules:
            return False
        
        rule = self.alert_rules[rule_name]
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.now()
        self.logger.info(f"Updated alert rule: {rule_name}")
        self._persist_rules()
        return True
    
    def trigger_alert(self, 
                     rule_name: str,
                     metric_name: str,
                     current_value: float,
                     threshold_value: Union[float, str],
                     metadata: Dict[str, Any] = None) -> Optional[Alert]:
        """
        Trigger an alert for a specific rule.
        
        Args:
            rule_name: Name of the alert rule
            metric_name: Name of the metric that triggered the alert
            current_value: Current value of the metric
            threshold_value: Threshold value that was violated
            metadata: Additional metadata
            
        Returns:
            Alert object if created, None if suppressed
        """
        if rule_name not in self.alert_rules:
            self.logger.error(f"Alert rule not found: {rule_name}")
            return None
        
        rule = self.alert_rules[rule_name]
        
        if not rule.enabled:
            self.logger.debug(f"Alert rule disabled: {rule_name}")
            return None
        
        # Check suppression
        if self.enable_suppression and self._is_suppressed(rule_name):
            self.logger.info(f"Alert suppressed: {rule_name}")
            return None
        
        # Check business hours
        if rule.business_hours_only and not self._is_business_hours():
            self.logger.info(f"Alert outside business hours: {rule_name}")
            return None
        
        # Create alert
        alert_id = f"{rule_name}_{int(time.time() * 1000)}"
        alert = Alert(
            id=alert_id,
            rule_name=rule_name,
            title=f"Alert: {rule.name}",
            description=rule.description,
            severity=rule.severity,
            priority=rule.priority,
            status=AlertStatus.TRIGGERED,
            triggered_at=datetime.now(),
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            tags=rule.tags.copy(),
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        self._send_alert_notifications(alert, rule)
        
        # Add to escalation queue if enabled
        if self.enable_escalation:
            self.escalation_queue.append(alert_id)
        
        # Update suppression counter
        if self.enable_suppression:
            self._update_suppression_counter(rule_name)
        
        self.logger.info(f"Alert triggered: {alert_id}")
        self._persist_alerts()
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = None) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if alert was acknowledged successfully
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        
        if acknowledged_by:
            alert.metadata['acknowledged_by'] = acknowledged_by
        
        # Remove from escalation queue
        if alert_id in self.escalation_queue:
            self.escalation_queue.remove(alert_id)
        
        self.logger.info(f"Alert acknowledged: {alert_id}")
        self._persist_alerts()
        
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Who resolved the alert
            
        Returns:
            True if alert was resolved successfully
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        if resolved_by:
            alert.metadata['resolved_by'] = resolved_by
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Remove from escalation queue
        if alert_id in self.escalation_queue:
            self.escalation_queue.remove(alert_id)
        
        self.logger.info(f"Alert resolved: {alert_id}")
        self._persist_alerts()
        
        return True
    
    def suppress_alert(self, alert_id: str, suppressed_by: str = None) -> bool:
        """
        Suppress an alert.
        
        Args:
            alert_id: ID of the alert to suppress
            suppressed_by: Who suppressed the alert
            
        Returns:
            True if alert was suppressed successfully
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        
        if suppressed_by:
            alert.metadata['suppressed_by'] = suppressed_by
        
        # Remove from escalation queue
        if alert_id in self.escalation_queue:
            self.escalation_queue.remove(alert_id)
        
        self.logger.info(f"Alert suppressed: {alert_id}")
        self._persist_alerts()
        
        return True
    
    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None,
                         priority: Optional[AlertPriority] = None,
                         status: Optional[AlertStatus] = None) -> List[Alert]:
        """
        Get active alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            priority: Filter by priority
            status: Filter by status
            
        Returns:
            List of matching alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return alerts
    
    def get_alert_history(self, 
                         hours_back: int = 24,
                         rule_name: Optional[str] = None) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            hours_back: Number of hours to look back
            rule_name: Filter by rule name
            
        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        alerts = [
            a for a in self.alert_history
            if a.triggered_at >= cutoff_time
        ]
        
        if rule_name:
            alerts = [a for a in alerts if a.rule_name == rule_name]
        
        return alerts
    
    def get_alert_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Args:
            hours_back: Number of hours to analyze
            
        Returns:
            Dictionary with alert statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Get recent alerts
        recent_alerts = [
            a for a in self.alert_history
            if a.triggered_at >= cutoff_time
        ]
        
        stats = {
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': {},
            'alerts_by_priority': {},
            'alerts_by_status': {},
            'alerts_by_rule': {},
            'resolution_times': {},
            'escalation_rate': 0.0,
            'suppression_rate': 0.0
        }
        
        # Count by severity
        for severity in AlertSeverity:
            count = len([a for a in recent_alerts if a.severity == severity])
            stats['alerts_by_severity'][severity.value] = count
        
        # Count by priority
        for priority in AlertPriority:
            count = len([a for a in recent_alerts if a.priority == priority])
            stats['alerts_by_priority'][priority.value] = count
        
        # Count by status
        for status in AlertStatus:
            count = len([a for a in recent_alerts if a.status == status])
            stats['alerts_by_status'][status.value] = count
        
        # Count by rule
        for alert in recent_alerts:
            rule_name = alert.rule_name
            if rule_name not in stats['alerts_by_rule']:
                stats['alerts_by_rule'][rule_name] = 0
            stats['alerts_by_rule'][rule_name] += 1
        
        # Calculate resolution times
        resolved_alerts = [a for a in recent_alerts if a.resolved_at]
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.triggered_at).total_seconds() / 60
                for a in resolved_alerts
            ]
            stats['resolution_times'] = {
                'avg_minutes': sum(resolution_times) / len(resolution_times),
                'min_minutes': min(resolution_times),
                'max_minutes': max(resolution_times)
            }
        
        # Calculate rates
        if stats['total_alerts'] > 0:
            escalated_count = len([a for a in recent_alerts if a.escalation_level > 0])
            suppressed_count = len([a for a in recent_alerts if a.status == AlertStatus.SUPPRESSED])
            
            stats['escalation_rate'] = escalated_count / stats['total_alerts']
            stats['suppression_rate'] = suppressed_count / stats['total_alerts']
        
        return stats
    
    def add_notification_channel(self, name: str, config: NotificationConfig):
        """Add a notification channel."""
        self.notification_sender.add_config(name, config)
    
    def test_notification_channel(self, channel_name: str) -> bool:
        """Test a notification channel."""
        return self.notification_sender.test_configuration(channel_name)
    
    def _processing_loop(self):
        """Background processing loop for escalation and cleanup."""
        while self.processing_active:
            try:
                # Process escalations
                if self.enable_escalation:
                    self._process_escalations()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Clean up suppression counters
                self._cleanup_suppression_counters()
                
                # Sleep for processing interval
                time.sleep(60)  # Process every minute
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(60)
    
    def _process_escalations(self):
        """Process alert escalations."""
        for alert_id in self.escalation_queue.copy():
            if alert_id not in self.active_alerts:
                self.escalation_queue.remove(alert_id)
                continue
            
            alert = self.active_alerts[alert_id]
            rule = self.alert_rules.get(alert.rule_name)
            
            if not rule:
                continue
            
            # Check if escalation timeout has passed
            time_since_trigger = datetime.now() - alert.triggered_at
            if time_since_trigger.total_seconds() < rule.escalation_timeout_minutes * 60:
                continue
            
            # Escalate alert
            alert.escalation_level += 1
            alert.escalated_at = datetime.now()
            alert.status = AlertStatus.ESCALATED
            
            # Send escalation notifications
            self._send_escalation_notifications(alert, rule)
            
            # Remove from queue if max escalation reached
            if alert.escalation_level >= len(rule.escalation_levels):
                self.escalation_queue.remove(alert_id)
            
            self.logger.info(f"Alert escalated: {alert_id} (level {alert.escalation_level})")
    
    def _is_suppressed(self, rule_name: str) -> bool:
        """Check if an alert should be suppressed due to rate limiting."""
        if rule_name not in self.alert_rules:
            return False
        
        rule = self.alert_rules[rule_name]
        
        # Check rate limiting
        if rule_name in self.suppression_counters:
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_alerts = [
                ts for ts in self.suppression_counters[rule_name]
                if ts > one_hour_ago
            ]
            
            if len(recent_alerts) >= rule.max_alerts_per_hour:
                return True
        
        return False
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        # Simple business hours check (9 AM to 5 PM, Monday-Friday)
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        if now.hour < 9 or now.hour >= 17:  # Outside 9-5
            return False
        
        return True
    
    def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for a triggered alert."""
        message = NotificationMessage(
            title=alert.title,
            content=f"{alert.description}\n\nMetric: {alert.metric_name}\nCurrent Value: {alert.current_value}\nThreshold: {alert.threshold_value}",
            severity=alert.severity.value,
            timestamp=alert.triggered_at,
            metadata=alert.metadata
        )
        
        # Send to configured channels
        for channel_name in rule.notification_channels:
            if channel_name in self.notification_sender.configs:
                success = self.notification_sender.send_notification(message, channel_name)
                if success:
                    alert.notifications_sent.append(channel_name)
        
        # Send to default channels if none configured
        if not rule.notification_channels:
            results = self.notification_sender.send_to_all(message)
            alert.notifications_sent.extend([k for k, v in results.items() if v])
    
    def _send_escalation_notifications(self, alert: Alert, rule: AlertRule):
        """Send escalation notifications."""
        escalation_message = NotificationMessage(
            title=f"ESCALATED: {alert.title}",
            content=f"Alert has been escalated to level {alert.escalation_level}\n\n{alert.description}\n\nMetric: {alert.metric_name}\nCurrent Value: {alert.current_value}\nThreshold: {alert.threshold_value}",
            severity="critical",
            timestamp=datetime.now(),
            metadata={**alert.metadata, 'escalation_level': alert.escalation_level}
        )
        
        # Send to escalation channels
        if alert.escalation_level <= len(rule.escalation_levels):
            escalation_channels = rule.escalation_levels[alert.escalation_level - 1].split(',')
            for channel_name in escalation_channels:
                channel_name = channel_name.strip()
                if channel_name in self.notification_sender.configs:
                    self.notification_sender.send_notification(escalation_message, channel_name)
    
    def _update_suppression_counter(self, rule_name: str):
        """Update suppression counter for a rule."""
        if rule_name not in self.suppression_counters:
            self.suppression_counters[rule_name] = []
        
        self.suppression_counters[rule_name].append(datetime.now())
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts from history."""
        # Keep only last 7 days of alerts
        cutoff_time = datetime.now() - timedelta(days=7)
        self.alert_history = [
            a for a in self.alert_history
            if a.triggered_at >= cutoff_time
        ]
    
    def _cleanup_suppression_counters(self):
        """Clean up old suppression counter entries."""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        for rule_name in self.suppression_counters:
            self.suppression_counters[rule_name] = [
                ts for ts in self.suppression_counters[rule_name]
                if ts > one_hour_ago
            ]
    
    def _persist_rules(self):
        """Persist alert rules to disk."""
        rules_data = {}
        for name, rule in self.alert_rules.items():
            rules_data[name] = {
                'name': rule.name,
                'description': rule.description,
                'condition': rule.condition,
                'severity': rule.severity.value,
                'priority': rule.priority.value,
                'enabled': rule.enabled,
                'escalation_timeout_minutes': rule.escalation_timeout_minutes,
                'escalation_levels': rule.escalation_levels,
                'notification_channels': rule.notification_channels,
                'business_hours_only': rule.business_hours_only,
                'suppression_rules': rule.suppression_rules,
                'max_alerts_per_hour': rule.max_alerts_per_hour,
                'tags': rule.tags,
                'owner': rule.owner,
                'runbook_url': rule.runbook_url,
                'created_at': rule.created_at.isoformat(),
                'updated_at': rule.updated_at.isoformat()
            }
        
        with open(self.persistence_path / "alert_rules.json", 'w') as f:
            json.dump(rules_data, f, indent=2)
    
    def _persist_alerts(self):
        """Persist active alerts to disk."""
        alerts_data = [alert.to_dict() for alert in self.active_alerts.values()]
        
        with open(self.persistence_path / "active_alerts.json", 'w') as f:
            json.dump(alerts_data, f, indent=2)
    
    def _load_persisted_data(self):
        """Load persisted data from disk."""
        # Load alert rules
        rules_file = self.persistence_path / "alert_rules.json"
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                
                for name, rule_data in rules_data.items():
                    rule = AlertRule(
                        name=rule_data['name'],
                        description=rule_data['description'],
                        condition=rule_data['condition'],
                        severity=AlertSeverity(rule_data['severity']),
                        priority=AlertPriority(rule_data['priority']),
                        enabled=rule_data['enabled'],
                        escalation_timeout_minutes=rule_data['escalation_timeout_minutes'],
                        escalation_levels=rule_data['escalation_levels'],
                        notification_channels=rule_data['notification_channels'],
                        business_hours_only=rule_data['business_hours_only'],
                        suppression_rules=rule_data['suppression_rules'],
                        max_alerts_per_hour=rule_data['max_alerts_per_hour'],
                        tags=rule_data['tags'],
                        owner=rule_data['owner'],
                        runbook_url=rule_data['runbook_url'],
                        created_at=datetime.fromisoformat(rule_data['created_at']),
                        updated_at=datetime.fromisoformat(rule_data['updated_at'])
                    )
                    self.alert_rules[name] = rule
                
                self.logger.info(f"Loaded {len(self.alert_rules)} alert rules")
                
            except Exception as e:
                self.logger.error(f"Error loading alert rules: {e}")
        
        # Load active alerts
        alerts_file = self.persistence_path / "active_alerts.json"
        if alerts_file.exists():
            try:
                with open(alerts_file, 'r') as f:
                    alerts_data = json.load(f)
                
                for alert_data in alerts_data:
                    alert = Alert(
                        id=alert_data['id'],
                        rule_name=alert_data['rule_name'],
                        title=alert_data['title'],
                        description=alert_data['description'],
                        severity=AlertSeverity(alert_data['severity']),
                        priority=AlertPriority(alert_data['priority']),
                        status=AlertStatus(alert_data['status']),
                        triggered_at=datetime.fromisoformat(alert_data['triggered_at']),
                        acknowledged_at=datetime.fromisoformat(alert_data['acknowledged_at']) if alert_data['acknowledged_at'] else None,
                        resolved_at=datetime.fromisoformat(alert_data['resolved_at']) if alert_data['resolved_at'] else None,
                        metric_name=alert_data['metric_name'],
                        current_value=alert_data['current_value'],
                        threshold_value=alert_data['threshold_value'],
                        escalation_level=alert_data['escalation_level'],
                        escalated_at=datetime.fromisoformat(alert_data['escalated_at']) if alert_data['escalated_at'] else None,
                        notifications_sent=alert_data['notifications_sent'],
                        tags=alert_data['tags'],
                        metadata=alert_data['metadata']
                    )
                    self.active_alerts[alert.id] = alert
                
                self.logger.info(f"Loaded {len(self.active_alerts)} active alerts")
                
            except Exception as e:
                self.logger.error(f"Error loading active alerts: {e}")


def create_default_alert_rules() -> List[AlertRule]:
    """Create default alert rules for common scenarios."""
    rules = []
    
    # Performance alert rules
    rules.append(AlertRule(
        name="critical_success_rate",
        description="Trading success rate dropped below critical threshold",
        condition="success_rate < 30",
        severity=AlertSeverity.CRITICAL,
        priority=AlertPriority.P1_CRITICAL,
        escalation_timeout_minutes=15,
        escalation_levels=["console", "email,slack"],
        notification_channels=["console", "email"],
        max_alerts_per_hour=5,
        tags=["performance", "trading"],
        runbook_url="https://docs.example.com/runbooks/success-rate"
    ))
    
    rules.append(AlertRule(
        name="high_system_load",
        description="System resource usage is high",
        condition="cpu_usage > 80 OR memory_usage > 85",
        severity=AlertSeverity.WARNING,
        priority=AlertPriority.P2_HIGH,
        escalation_timeout_minutes=30,
        escalation_levels=["console"],
        notification_channels=["console"],
        max_alerts_per_hour=10,
        tags=["system", "resources"]
    ))
    
    rules.append(AlertRule(
        name="large_loss_detected",
        description="Large trading loss detected",
        condition="worst_return < -20",
        severity=AlertSeverity.CRITICAL,
        priority=AlertPriority.P1_CRITICAL,
        escalation_timeout_minutes=5,
        escalation_levels=["console,email", "slack"],
        notification_channels=["console", "email"],
        max_alerts_per_hour=3,
        tags=["trading", "risk"],
        runbook_url="https://docs.example.com/runbooks/large-loss"
    ))
    
    return rules