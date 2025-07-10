"""
Integration tests for Phase 3 monitoring and alerting components.

These tests verify that atoms and molecules work together correctly
for monitoring, alerting, and dashboard generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import json
import time
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Phase 3 components
from atoms.monitoring.performance_tracker import (
    PerformanceTracker, calculate_real_time_metrics, monitor_system_health
)
from atoms.alerting.threshold_monitor import (
    ThresholdMonitor, ThresholdRule, AlertSeverity, ThresholdType
)
from atoms.alerting.notification_sender import (
    NotificationSender, NotificationMessage, NotificationChannel,
    create_console_config, create_file_config
)
from atoms.dashboard.dashboard_generator import (
    DashboardGenerator, create_sample_dashboard_data
)
from molecules.system_monitor import SystemMonitor
from molecules.alert_manager import (
    AlertManager, AlertRule, AlertPriority, AlertStatus, create_default_alert_rules
)


class TestPerformanceTracker:
    """Test PerformanceTracker atom."""
    
    @pytest.fixture
    def tracker(self):
        """Create a temporary performance tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield PerformanceTracker()
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.performance_targets is not None
        assert len(tracker.performance_targets) > 0
        assert tracker.metrics_history is not None
    
    def test_metric_recording(self, tracker):
        """Test metric recording and retrieval."""
        # Record some metrics
        tracker.record_metric("test_metric", 42.0, "test")
        tracker.record_metric("test_metric", 45.0, "test")
        
        # Retrieve metric history
        history = tracker.get_metric_history("test_metric")
        assert len(history) == 2
        assert history['metric_value'].iloc[0] == 42.0
        assert history['metric_value'].iloc[1] == 45.0
    
    def test_performance_snapshot(self, tracker):
        """Test performance snapshot recording."""
        snapshot_data = {
            'success_rate': 75.0,
            'total_return': 5.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': 3.2
        }
        
        tracker.record_performance_snapshot(snapshot_data)
        
        # Retrieve snapshots
        snapshots = tracker.get_performance_snapshots(days=1)
        assert len(snapshots) == 1
        assert snapshots.iloc[0]['success_rate'] == 75.0
    
    def test_trend_analysis(self, tracker):
        """Test trend analysis calculation."""
        # Record trending data
        for i in range(5):
            tracker.record_metric("trending_metric", float(i * 10), "test")
        
        # Calculate trend
        trend = tracker.calculate_trend_analysis("trending_metric", window_days=1)
        assert trend['status'] == 'success'
        assert trend['trend_direction'] == 'increasing'
        assert trend['current_value'] == 40.0


class TestThresholdMonitor:
    """Test ThresholdMonitor atom."""
    
    @pytest.fixture
    def monitor(self):
        """Create a threshold monitor."""
        return ThresholdMonitor()
    
    def test_rule_management(self, monitor):
        """Test adding and managing threshold rules."""
        rule = ThresholdRule(
            name="test_rule",
            metric_name="test_metric",
            threshold_type=ThresholdType.GREATER_THAN,
            threshold_value=50.0,
            severity=AlertSeverity.WARNING,
            description="Test rule"
        )
        
        # Add rule
        assert monitor.add_rule(rule) == True
        assert "test_rule" in monitor.rules
        
        # Update rule
        assert monitor.update_rule("test_rule", threshold_value=60.0) == True
        assert monitor.rules["test_rule"].threshold_value == 60.0
        
        # Remove rule
        assert monitor.remove_rule("test_rule") == True
        assert "test_rule" not in monitor.rules
    
    def test_threshold_checking(self, monitor):
        """Test threshold violation detection."""
        rule = ThresholdRule(
            name="test_rule",
            metric_name="test_metric",
            threshold_type=ThresholdType.GREATER_THAN,
            threshold_value=50.0,
            severity=AlertSeverity.WARNING,
            description="Test rule"
        )
        
        monitor.add_rule(rule)
        
        # Test value below threshold (no violation)
        violations = monitor.check_metric("test_metric", 40.0)
        assert len(violations) == 0
        
        # Test value above threshold (violation)
        violations = monitor.check_metric("test_metric", 60.0)
        assert len(violations) == 1
        assert violations[0].rule_name == "test_rule"
        assert violations[0].current_value == 60.0
    
    def test_multiple_metrics_checking(self, monitor):
        """Test checking multiple metrics simultaneously."""
        # Add multiple rules
        rule1 = ThresholdRule(
            name="cpu_rule",
            metric_name="cpu_usage",
            threshold_type=ThresholdType.GREATER_THAN,
            threshold_value=80.0,
            severity=AlertSeverity.WARNING,
            description="CPU usage high"
        )
        
        rule2 = ThresholdRule(
            name="memory_rule",
            metric_name="memory_usage",
            threshold_type=ThresholdType.GREATER_THAN,
            threshold_value=85.0,
            severity=AlertSeverity.CRITICAL,
            description="Memory usage critical"
        )
        
        monitor.add_rule(rule1)
        monitor.add_rule(rule2)
        
        # Test multiple metrics
        metrics = {
            "cpu_usage": 85.0,      # Violation
            "memory_usage": 90.0,   # Violation
            "disk_usage": 50.0      # No rule, no violation
        }
        
        violations = monitor.check_multiple_metrics(metrics)
        assert len(violations) == 2
        
        # Check that both violations were detected
        rule_names = [v.rule_name for v in violations]
        assert "cpu_rule" in rule_names
        assert "memory_rule" in rule_names


class TestNotificationSender:
    """Test NotificationSender atom."""
    
    @pytest.fixture
    def sender(self):
        """Create a notification sender."""
        return NotificationSender()
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    def test_configuration_management(self, sender):
        """Test notification configuration management."""
        # Add console config
        console_config = create_console_config()
        assert sender.add_config("console", console_config) == True
        assert "console" in sender.configs
        
        # Remove config
        assert sender.remove_config("console") == True
        assert "console" not in sender.configs
    
    def test_console_notification(self, sender):
        """Test console notification sending."""
        # Add console config
        console_config = create_console_config()
        sender.add_config("console", console_config)
        
        # Create test message
        message = NotificationMessage(
            title="Test Alert",
            content="This is a test notification",
            severity="warning",
            timestamp=datetime.now()
        )
        
        # Send notification
        success = sender.send_notification(message, "console")
        assert success == True
    
    def test_file_notification(self, sender, temp_file):
        """Test file notification sending."""
        # Add file config
        file_config = create_file_config(temp_file)
        sender.add_config("file", file_config)
        
        # Create test message
        message = NotificationMessage(
            title="Test Alert",
            content="This is a test notification",
            severity="info",
            timestamp=datetime.now()
        )
        
        # Send notification
        success = sender.send_notification(message, "file")
        assert success == True
        
        # Verify file was written
        assert os.path.exists(temp_file)
        with open(temp_file, 'r') as f:
            content = f.read()
            assert "Test Alert" in content
    
    def test_notification_statistics(self, sender):
        """Test notification statistics tracking."""
        # Add console config
        console_config = create_console_config()
        sender.add_config("console", console_config)
        
        # Send some notifications
        for i in range(3):
            message = NotificationMessage(
                title=f"Test Alert {i}",
                content="Test content",
                severity="info",
                timestamp=datetime.now()
            )
            sender.send_notification(message, "console")
        
        # Check statistics
        stats = sender.get_notification_stats(hours_back=1)
        assert stats['total_notifications'] == 3
        assert stats['successful_notifications'] == 3
        assert stats['success_rate'] == 1.0


class TestDashboardGenerator:
    """Test DashboardGenerator atom."""
    
    @pytest.fixture
    def generator(self):
        """Create a dashboard generator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield DashboardGenerator(output_dir=temp_dir)
    
    def test_dashboard_generation(self, generator):
        """Test dashboard generation."""
        # Create sample data
        dashboard_data = create_sample_dashboard_data()
        
        # Generate dashboard
        dashboard_path = generator.generate_performance_dashboard(
            dashboard_data, 
            "test_dashboard.html"
        )
        
        # Verify dashboard was created
        assert os.path.exists(dashboard_path)
        
        # Verify HTML content
        with open(dashboard_path, 'r') as f:
            html_content = f.read()
            assert "<html" in html_content
            assert "Alert Performance Dashboard" in html_content
    
    def test_trading_dashboard(self, generator):
        """Test trading dashboard generation."""
        # Create sample trading data
        trading_data = {
            'trading_metrics': {
                'total_trades': 100,
                'win_rate': 65.5,
                'profit_factor': 1.45,
                'avg_trade': 25.50
            },
            'equity_curve': {
                'dates': [datetime.now() - timedelta(days=i) for i in range(10)],
                'values': [10000 + i * 100 for i in range(10)]
            },
            'positions': [
                {
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'entry_price': 150.0,
                    'current_price': 155.0,
                    'unrealized_pnl': 500.0,
                    'unrealized_pnl_pct': 3.33
                }
            ]
        }
        
        # Generate trading dashboard
        dashboard_path = generator.generate_trading_dashboard(
            trading_data, 
            "test_trading_dashboard.html"
        )
        
        # Verify dashboard was created
        assert os.path.exists(dashboard_path)
        
        # Verify content
        with open(dashboard_path, 'r') as f:
            html_content = f.read()
            assert "Trading Performance Dashboard" in html_content


class TestSystemMonitor:
    """Test SystemMonitor molecule integration."""
    
    @pytest.fixture
    def monitor(self):
        """Create a system monitor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield SystemMonitor(
                monitoring_interval=1,  # Fast for testing
                dashboard_update_interval=2,
                output_dir=temp_dir
            )
    
    def test_monitor_initialization(self, monitor):
        """Test system monitor initialization."""
        assert monitor.performance_tracker is not None
        assert monitor.threshold_monitor is not None
        assert monitor.notification_sender is not None
        assert monitor.dashboard_generator is not None
        assert len(monitor.threshold_monitor.rules) > 0  # Default rules loaded
    
    def test_custom_rule_addition(self, monitor):
        """Test adding custom monitoring rules."""
        from atoms.alerting.threshold_monitor import ThresholdRule, ThresholdType, AlertSeverity
        
        custom_rule = ThresholdRule(
            name="custom_test_rule",
            metric_name="custom_metric",
            threshold_type=ThresholdType.GREATER_THAN,
            threshold_value=100.0,
            severity=AlertSeverity.WARNING,
            description="Custom test rule"
        )
        
        monitor.add_custom_rule(custom_rule)
        assert "custom_test_rule" in monitor.threshold_monitor.rules
    
    def test_trading_metrics_recording(self, monitor):
        """Test recording trading metrics."""
        trading_metrics = {
            'success_rate': 72.5,
            'avg_return': 2.3,
            'max_drawdown': 5.1,
            'sharpe_ratio': 1.4
        }
        
        monitor.record_trading_metrics(trading_metrics)
        
        # Verify metrics were recorded
        history = monitor.performance_tracker.get_metric_history("success_rate")
        assert len(history) == 1
        assert history.iloc[0]['metric_value'] == 72.5
    
    def test_alert_generation_tracking(self, monitor):
        """Test alert generation tracking."""
        # Record alert generation
        monitor.record_alert_generated(
            symbol="AAPL",
            priority="HIGH",
            confidence=0.85,
            generation_time_ms=125.0
        )
        
        # Check alert performance
        alert_performance = monitor.performance_tracker.get_alert_performance()
        assert alert_performance['total_alerts'] == 1
        assert alert_performance['avg_confidence_score'] == 0.85
    
    def test_current_status_retrieval(self, monitor):
        """Test current status retrieval."""
        # Record some metrics
        monitor.record_trading_metrics({'success_rate': 75.0})
        
        # Get current status
        status = monitor.get_current_status()
        
        assert 'monitoring_active' in status
        assert 'timestamp' in status
        assert 'performance_summary' in status
        assert 'system_health' in status
        assert 'latest_metrics' in status
    
    def test_monitoring_report_generation(self, monitor):
        """Test monitoring report generation."""
        # Record some data
        monitor.record_trading_metrics({'success_rate': 70.0})
        
        # Generate report
        report_path = monitor.generate_monitoring_report(hours_back=1)
        
        # Verify report was created
        assert os.path.exists(report_path)
        
        # Verify report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
            assert 'report_metadata' in report_data
            assert 'system_status' in report_data
    
    def test_dashboard_generation(self, monitor):
        """Test dashboard generation."""
        # Record some data
        monitor.record_trading_metrics({'success_rate': 65.0})
        
        # Generate dashboard
        dashboard_path = monitor.generate_dashboard()
        
        # Verify dashboard was created
        assert os.path.exists(dashboard_path)
        
        # Verify HTML content
        with open(dashboard_path, 'r') as f:
            html_content = f.read()
            assert "System Monitoring Dashboard" in html_content or "Alert Performance Dashboard" in html_content


class TestAlertManager:
    """Test AlertManager molecule integration."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create an alert manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield AlertManager(persistence_path=temp_dir)
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert alert_manager.threshold_monitor is not None
        assert alert_manager.notification_sender is not None
        assert alert_manager.alert_rules is not None
        assert alert_manager.active_alerts is not None
    
    def test_alert_rule_management(self, alert_manager):
        """Test alert rule management."""
        from molecules.alert_manager import AlertRule, AlertSeverity, AlertPriority
        
        # Create test rule
        rule = AlertRule(
            name="test_performance_rule",
            description="Test performance alert",
            condition="success_rate < 50",
            severity=AlertSeverity.WARNING,
            priority=AlertPriority.P2_HIGH,
            notification_channels=["console"]
        )
        
        # Add rule
        assert alert_manager.add_alert_rule(rule) == True
        assert "test_performance_rule" in alert_manager.alert_rules
        
        # Update rule
        assert alert_manager.update_alert_rule("test_performance_rule", enabled=False) == True
        assert alert_manager.alert_rules["test_performance_rule"].enabled == False
        
        # Remove rule
        assert alert_manager.remove_alert_rule("test_performance_rule") == True
        assert "test_performance_rule" not in alert_manager.alert_rules
    
    def test_alert_triggering(self, alert_manager):
        """Test alert triggering and lifecycle."""
        from molecules.alert_manager import AlertRule, AlertSeverity, AlertPriority
        
        # Add console notification
        from atoms.alerting.notification_sender import create_console_config
        console_config = create_console_config()
        alert_manager.add_notification_channel("console", console_config)
        
        # Create test rule
        rule = AlertRule(
            name="test_alert_rule",
            description="Test alert rule",
            condition="test_metric > 50",
            severity=AlertSeverity.WARNING,
            priority=AlertPriority.P3_MEDIUM,
            notification_channels=["console"]
        )
        
        alert_manager.add_alert_rule(rule)
        
        # Trigger alert
        alert = alert_manager.trigger_alert(
            rule_name="test_alert_rule",
            metric_name="test_metric",
            current_value=75.0,
            threshold_value=50.0
        )
        
        # Verify alert was created
        assert alert is not None
        assert alert.rule_name == "test_alert_rule"
        assert alert.current_value == 75.0
        assert alert.status == AlertStatus.TRIGGERED
        
        # Test alert acknowledgment
        assert alert_manager.acknowledge_alert(alert.id) == True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        
        # Test alert resolution
        assert alert_manager.resolve_alert(alert.id) == True
        assert alert.status == AlertStatus.RESOLVED
        assert alert.id not in alert_manager.active_alerts
    
    def test_alert_statistics(self, alert_manager):
        """Test alert statistics calculation."""
        from molecules.alert_manager import AlertRule, AlertSeverity, AlertPriority
        
        # Add console notification
        from atoms.alerting.notification_sender import create_console_config
        console_config = create_console_config()
        alert_manager.add_notification_channel("console", console_config)
        
        # Create and add test rules
        for i in range(3):
            rule = AlertRule(
                name=f"test_rule_{i}",
                description=f"Test rule {i}",
                condition="test_metric > 50",
                severity=AlertSeverity.WARNING,
                priority=AlertPriority.P3_MEDIUM,
                notification_channels=["console"]
            )
            alert_manager.add_alert_rule(rule)
        
        # Trigger multiple alerts
        for i in range(3):
            alert_manager.trigger_alert(
                rule_name=f"test_rule_{i}",
                metric_name="test_metric",
                current_value=75.0,
                threshold_value=50.0
            )
        
        # Get statistics
        stats = alert_manager.get_alert_statistics(hours_back=1)
        
        assert stats['total_alerts'] == 3
        assert stats['active_alerts'] == 3
        assert stats['alerts_by_severity']['warning'] == 3
        assert stats['alerts_by_priority']['p3_medium'] == 3
    
    def test_default_alert_rules(self, alert_manager):
        """Test default alert rules creation."""
        default_rules = create_default_alert_rules()
        
        # Add default rules
        for rule in default_rules:
            alert_manager.add_alert_rule(rule)
        
        # Verify rules were added
        assert len(alert_manager.alert_rules) == len(default_rules)
        assert "critical_success_rate" in alert_manager.alert_rules
        assert "high_system_load" in alert_manager.alert_rules
        assert "large_loss_detected" in alert_manager.alert_rules


class TestEndToEndIntegration:
    """Test end-to-end integration of all Phase 3 components."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create an integrated monitoring system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system_monitor = SystemMonitor(
                monitoring_interval=1,
                dashboard_update_interval=2,
                output_dir=temp_dir
            )
            
            alert_manager = AlertManager(persistence_path=temp_dir)
            
            yield {
                'system_monitor': system_monitor,
                'alert_manager': alert_manager,
                'temp_dir': temp_dir
            }
    
    def test_full_monitoring_workflow(self, integrated_system):
        """Test complete monitoring workflow."""
        system_monitor = integrated_system['system_monitor']
        alert_manager = integrated_system['alert_manager']
        
        # 1. Record trading metrics
        trading_metrics = {
            'success_rate': 45.0,  # Below threshold
            'avg_return': 1.5,
            'max_drawdown': 8.2,
            'sharpe_ratio': 0.9
        }
        system_monitor.record_trading_metrics(trading_metrics)
        
        # 2. Check thresholds (should trigger alert)
        violations = system_monitor.threshold_monitor.check_multiple_metrics(trading_metrics)
        assert len(violations) > 0  # Should have violations for low success rate
        
        # 3. Generate monitoring report
        report_path = system_monitor.generate_monitoring_report(hours_back=1)
        assert os.path.exists(report_path)
        
        # 4. Generate dashboard
        dashboard_path = system_monitor.generate_dashboard()
        assert os.path.exists(dashboard_path)
        
        # 5. Verify system status
        status = system_monitor.get_current_status()
        assert 'performance_summary' in status
        assert 'system_health' in status
        assert 'recent_violations' in status
    
    def test_alert_escalation_workflow(self, integrated_system):
        """Test alert escalation workflow."""
        alert_manager = integrated_system['alert_manager']
        
        # Add console notification
        from atoms.alerting.notification_sender import create_console_config
        console_config = create_console_config()
        alert_manager.add_notification_channel("console", console_config)
        
        # Create rule with escalation
        from molecules.alert_manager import AlertRule, AlertSeverity, AlertPriority
        rule = AlertRule(
            name="escalation_test_rule",
            description="Test escalation rule",
            condition="test_metric > 100",
            severity=AlertSeverity.CRITICAL,
            priority=AlertPriority.P1_CRITICAL,
            escalation_timeout_minutes=0,  # Immediate escalation for testing
            escalation_levels=["console"],
            notification_channels=["console"]
        )
        
        alert_manager.add_alert_rule(rule)
        
        # Trigger alert
        alert = alert_manager.trigger_alert(
            rule_name="escalation_test_rule",
            metric_name="test_metric",
            current_value=150.0,
            threshold_value=100.0
        )
        
        # Verify alert was created
        assert alert is not None
        assert alert.id in alert_manager.active_alerts
        
        # Check escalation queue
        if alert_manager.enable_escalation:
            assert alert.id in alert_manager.escalation_queue
    
    def test_performance_benchmarking(self, integrated_system):
        """Test performance benchmarking integration."""
        system_monitor = integrated_system['system_monitor']
        
        # Record historical performance data
        for i in range(10):
            metrics = {
                'success_rate': 60.0 + i * 2,
                'avg_return': 1.0 + i * 0.1,
                'sharpe_ratio': 0.8 + i * 0.05
            }
            system_monitor.record_trading_metrics(metrics)
            
            # Record performance snapshot
            system_monitor.performance_tracker.record_performance_snapshot(metrics)
        
        # Calculate benchmarks
        from atoms.monitoring.performance_tracker import calculate_performance_benchmarks
        benchmarks = calculate_performance_benchmarks(system_monitor.performance_tracker)
        
        assert benchmarks['status'] == 'success'
        assert 'benchmarks' in benchmarks
        assert benchmarks['sample_size'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])