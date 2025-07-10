"""
System Monitoring Molecule.

This molecule combines monitoring, alerting, and dashboard atoms to provide
comprehensive system health monitoring and alerting capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import asyncio
import threading
import time
import json
from pathlib import Path

# Import atoms
from atoms.monitoring.performance_tracker import (
    PerformanceTracker, calculate_real_time_metrics, 
    monitor_system_health, calculate_performance_benchmarks
)
from atoms.alerting.threshold_monitor import (
    ThresholdMonitor, ThresholdRule, AlertSeverity, ThresholdType,
    create_performance_rules, create_trading_rules
)
from atoms.alerting.notification_sender import (
    NotificationSender, NotificationMessage, NotificationChannel,
    create_console_config, create_file_config
)
from atoms.dashboard.dashboard_generator import (
    DashboardGenerator, create_sample_dashboard_data
)


class SystemMonitor:
    """
    Comprehensive system monitoring and alerting molecule.
    
    This molecule combines performance tracking, threshold monitoring,
    alerting, and dashboard generation for complete system oversight.
    """
    
    def __init__(self, 
                 monitoring_interval: int = 30,
                 dashboard_update_interval: int = 60,
                 enable_notifications: bool = True,
                 enable_dashboard: bool = True,
                 output_dir: str = "monitoring_output"):
        """
        Initialize the system monitor.
        
        Args:
            monitoring_interval: How often to collect metrics (seconds)
            dashboard_update_interval: How often to update dashboard (seconds)
            enable_notifications: Whether to enable alert notifications
            enable_dashboard: Whether to generate dashboards
            output_dir: Directory for output files
        """
        self.monitoring_interval = monitoring_interval
        self.dashboard_update_interval = dashboard_update_interval
        self.enable_notifications = enable_notifications
        self.enable_dashboard = enable_dashboard
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.performance_tracker = PerformanceTracker()
        self.threshold_monitor = ThresholdMonitor()
        self.notification_sender = NotificationSender() if enable_notifications else None
        self.dashboard_generator = DashboardGenerator(output_dir=str(self.output_dir / "dashboards")) if enable_dashboard else None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.dashboard_thread = None
        
        # Data storage
        self.latest_metrics = {}
        self.alert_history = []
        self.system_health_history = []
        
        # Configuration
        self.setup_default_configuration()
    
    def setup_default_configuration(self):
        """Set up default monitoring configuration."""
        # Add default performance rules
        performance_rules = create_performance_rules()
        for rule in performance_rules:
            self.threshold_monitor.add_rule(rule)
        
        # Add default trading rules
        trading_rules = create_trading_rules()
        for rule in trading_rules:
            self.threshold_monitor.add_rule(rule)
        
        # Set up default notifications
        if self.notification_sender:
            # Console notifications
            console_config = create_console_config()
            self.notification_sender.add_config("console", console_config)
            
            # File notifications
            file_config = create_file_config(str(self.output_dir / "alerts.log"))
            self.notification_sender.add_config("file", file_config)
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start dashboard thread if enabled
        if self.enable_dashboard:
            self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
            self.dashboard_thread.start()
        
        print("System monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5)
        
        print("System monitoring stopped")
    
    def add_custom_rule(self, rule: ThresholdRule):
        """Add a custom threshold rule."""
        self.threshold_monitor.add_rule(rule)
    
    def add_notification_config(self, name: str, config):
        """Add a notification configuration."""
        if self.notification_sender:
            self.notification_sender.add_config(name, config)
    
    def record_trading_metrics(self, metrics: Dict[str, Any]):
        """Record trading-specific metrics."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.performance_tracker.record_metric(
                    metric_name=metric_name,
                    metric_value=float(value),
                    metric_type="trading"
                )
    
    def record_alert_generated(self, 
                             symbol: str, 
                             priority: str, 
                             confidence: float,
                             generation_time_ms: float):
        """Record alert generation metrics."""
        self.performance_tracker.record_alert_generated(
            priority=priority,
            confidence_score=confidence,
            generation_time_ms=generation_time_ms,
            symbol=symbol
        )
    
    def record_alert_outcome(self, 
                           symbol: str, 
                           success: bool, 
                           profit_loss: float = 0.0):
        """Record alert outcome for effectiveness tracking."""
        self.performance_tracker.record_alert_outcome(
            symbol=symbol,
            success=success,
            profit_loss=profit_loss
        )
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        # Get performance summary
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Get alert performance
        alert_performance = self.performance_tracker.get_alert_performance()
        
        # Get system health
        system_health = self.performance_tracker.get_system_health()
        
        # Get threshold violations
        recent_violations = self.threshold_monitor.get_all_violations(hours_back=1)
        
        # Get notification stats
        notification_stats = {}
        if self.notification_sender:
            notification_stats = self.notification_sender.get_notification_stats()
        
        return {
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance_summary,
            'alert_performance': alert_performance,
            'system_health': system_health.__dict__,
            'recent_violations': recent_violations,
            'notification_stats': notification_stats,
            'latest_metrics': self.latest_metrics
        }
    
    def generate_monitoring_report(self, 
                                 hours_back: int = 24,
                                 output_filename: str = None) -> str:
        """Generate comprehensive monitoring report."""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"monitoring_report_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        
        # Collect comprehensive data
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'hours_covered': hours_back,
                'monitoring_interval': self.monitoring_interval
            },
            'system_status': self.get_current_status(),
            'performance_benchmarks': calculate_performance_benchmarks(
                self.performance_tracker, 
                benchmark_period_days=max(1, hours_back // 24)
            ),
            'threshold_violations': self.threshold_monitor.get_all_violations(hours_back=hours_back),
            'metrics_summary': self.threshold_monitor.get_metrics_summary(),
            'rule_statuses': {}
        }
        
        # Get status for each rule
        for rule_name in self.threshold_monitor.rules:
            report_data['rule_statuses'][rule_name] = self.threshold_monitor.get_rule_status(rule_name)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(output_path)
    
    def generate_dashboard(self, output_filename: str = None) -> str:
        """Generate monitoring dashboard."""
        if not self.enable_dashboard:
            return ""
        
        if output_filename is None:
            output_filename = "system_monitoring_dashboard.html"
        
        # Prepare dashboard data
        dashboard_data = self._prepare_dashboard_data()
        
        # Generate dashboard
        dashboard_path = self.dashboard_generator.generate_performance_dashboard(
            dashboard_data, 
            output_filename
        )
        
        return dashboard_path
    
    def export_configuration(self, filename: str = None) -> str:
        """Export monitoring configuration."""
        if filename is None:
            filename = "monitoring_config.json"
        
        config_path = self.output_dir / filename
        
        # Export threshold rules
        rules_path = self.output_dir / "threshold_rules.json"
        self.threshold_monitor.export_configuration(str(rules_path))
        
        # Export monitoring configuration
        config_data = {
            'monitoring_interval': self.monitoring_interval,
            'dashboard_update_interval': self.dashboard_update_interval,
            'enable_notifications': self.enable_notifications,
            'enable_dashboard': self.enable_dashboard,
            'output_dir': str(self.output_dir),
            'threshold_rules_file': str(rules_path),
            'performance_targets': self.performance_tracker.performance_targets
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return str(config_path)
    
    def import_configuration(self, config_path: str) -> bool:
        """Import monitoring configuration."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update settings
            self.monitoring_interval = config_data.get('monitoring_interval', 30)
            self.dashboard_update_interval = config_data.get('dashboard_update_interval', 60)
            
            # Import threshold rules if specified
            if 'threshold_rules_file' in config_data:
                rules_path = config_data['threshold_rules_file']
                if Path(rules_path).exists():
                    self.threshold_monitor.import_configuration(rules_path)
            
            return True
            
        except Exception as e:
            print(f"Error importing configuration: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()
                
                # Update latest metrics
                self.latest_metrics = current_metrics
                
                # Check thresholds
                violations = self.threshold_monitor.check_multiple_metrics(current_metrics)
                
                # Send notifications for violations
                if violations and self.notification_sender:
                    self._send_violation_notifications(violations)
                
                # Store violations
                self.alert_history.extend(violations)
                
                # Record performance snapshot
                self._record_performance_snapshot(current_metrics)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _dashboard_loop(self):
        """Dashboard update loop."""
        while self.monitoring_active:
            try:
                # Generate dashboard
                self.generate_dashboard()
                
                # Sleep until next dashboard update
                time.sleep(self.dashboard_update_interval)
                
            except Exception as e:
                print(f"Error in dashboard loop: {e}")
                time.sleep(self.dashboard_update_interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        # Get system health
        system_health = self.performance_tracker.get_system_health()
        metrics['cpu_usage_percent'] = system_health.cpu_percent
        metrics['memory_usage_percent'] = system_health.memory_percent
        metrics['disk_usage_percent'] = system_health.disk_usage_percent
        
        # Get performance summary
        performance_summary = self.performance_tracker.get_performance_summary()
        if performance_summary:
            metrics['success_rate'] = performance_summary.get('success_rate', 0)
            metrics['avg_duration_ms'] = performance_summary.get('avg_duration_ms', 0)
            metrics['operations_per_minute'] = performance_summary.get('operations_per_minute', 0)
        
        # Get alert performance
        alert_performance = self.performance_tracker.get_alert_performance()
        if alert_performance:
            metrics['alerts_per_hour'] = alert_performance.get('alerts_per_hour', 0)
            metrics['avg_confidence_score'] = alert_performance.get('avg_confidence_score', 0)
            metrics['alert_success_rate'] = alert_performance.get('success_rate', 0)
        
        return metrics
    
    def _send_violation_notifications(self, violations: List):
        """Send notifications for threshold violations."""
        for violation in violations:
            message = NotificationMessage(
                title=f"Threshold Violation: {violation.rule_name}",
                content=f"{violation.description}\nCurrent value: {violation.current_value}\nThreshold: {violation.threshold_value}",
                severity=violation.severity.value,
                timestamp=violation.timestamp,
                metadata={
                    'rule_name': violation.rule_name,
                    'metric_name': violation.metric_name,
                    'threshold_type': violation.threshold_type.value,
                    'consecutive_count': violation.consecutive_count
                }
            )
            
            # Send to all configured channels
            self.notification_sender.send_to_all(message)
    
    def _record_performance_snapshot(self, metrics: Dict[str, float]):
        """Record performance snapshot."""
        snapshot_data = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': metrics.get('total_operations', 0),
            'success_rate': metrics.get('success_rate', 0),
            'total_return': 0,  # Would be calculated from trading data
            'sharpe_ratio': 0,  # Would be calculated from trading data
            'max_drawdown': 0,  # Would be calculated from trading data
            'var_5_percent': 0,  # Would be calculated from trading data
            'win_rate': metrics.get('alert_success_rate', 0),
            'profit_factor': 0,  # Would be calculated from trading data
        }
        
        self.performance_tracker.record_performance_snapshot(snapshot_data)
    
    def _prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prepare data for dashboard generation."""
        # Get current status
        status = self.get_current_status()
        
        # Format for dashboard
        dashboard_data = {
            'summary_metrics': {
                'success_rate': status.get('performance_summary', {}).get('success_rate', 0),
                'total_return': 0,  # Would come from trading data
                'sharpe_ratio': 0,  # Would come from trading data
                'max_drawdown': 0   # Would come from trading data
            },
            'recent_alerts': [
                {
                    'title': v.get('rule_name', 'Unknown'),
                    'severity': v.get('severity', 'info'),
                    'timestamp': v.get('timestamp', ''),
                    'content': v.get('description', '')
                }
                for v in status.get('recent_violations', [])[:10]
            ],
            'system_health': {
                'status': 'healthy',  # Would be calculated
                'cpu_usage': status.get('system_health', {}).get('cpu_percent', 0),
                'memory_usage': status.get('system_health', {}).get('memory_percent', 0),
                'disk_usage': status.get('system_health', {}).get('disk_usage_percent', 0),
                'uptime': f"{status.get('system_health', {}).get('process_uptime_seconds', 0) / 3600:.1f} hours"
            }
        }
        
        return dashboard_data
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return {
            'monitoring_uptime': time.time() - (self.performance_tracker.process_start_time if hasattr(self.performance_tracker, 'process_start_time') else time.time()),
            'total_alerts_processed': len(self.alert_history),
            'unique_metrics_monitored': len(self.latest_metrics),
            'threshold_rules_active': len([r for r in self.threshold_monitor.rules.values() if r.enabled]),
            'notification_channels': len(self.notification_sender.configs) if self.notification_sender else 0,
            'performance_snapshots': len(self.performance_tracker.get_performance_snapshots()),
            'dashboard_updates': 0,  # Would track dashboard generations
            'last_update': datetime.now().isoformat()
        }