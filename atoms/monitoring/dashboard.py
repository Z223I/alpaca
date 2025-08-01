"""
Real-time Monitoring Dashboard

This module provides a comprehensive monitoring dashboard for the ORB Alert system,
displaying real-time performance metrics, system health, and alert statistics.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import threading

from atoms.monitoring.performance_tracker import performance_tracker
from atoms.optimization.performance_optimizer import performance_optimizer
from atoms.validation.data_validator import data_validator
from atoms.config.alert_config import config


@dataclass
class DashboardMetrics:
    """Consolidated metrics for dashboard display."""
    timestamp: datetime
    system_health: Dict[str, Any]
    performance_summary: Dict[str, Any]
    alert_metrics: Dict[str, Any]
    optimization_stats: Dict[str, Any]
    validation_summary: Dict[str, Any]
    uptime_seconds: float
    alert_rate: float
    error_rate: float
    symbols_monitored: int
    memory_usage_mb: float
    cpu_usage_percent: float


class DashboardLogger:
    """Specialized logger for dashboard events."""

    def __init__(self, log_file: str = "logs/dashboard.log"):
        """Initialize dashboard logger."""
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger("dashboard")
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_alert_generated(self, symbol: str, priority: str, confidence: float) -> None:
        """Log alert generation event."""
        self.logger.info(f"ALERT_GENERATED,{symbol},{priority},{confidence:.3f}")

    def log_performance_issue(self, operation: str, duration_ms: float, threshold_ms: float) -> None:
        """Log performance issue."""
        self.logger.warning(f"PERFORMANCE_ISSUE,{operation},{duration_ms:.2f},{threshold_ms}")

    def log_system_health(self, memory_percent: float, cpu_percent: float) -> None:
        """Log system health metrics."""
        self.logger.info(f"SYSTEM_HEALTH,{memory_percent:.1f},{cpu_percent:.1f}")

    def log_error(self, error_type: str, message: str) -> None:
        """Log error event."""
        self.logger.error(f"ERROR,{error_type},{message}")


class WebDashboard:
    """Simple web-based dashboard for monitoring."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Initialize web dashboard.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.running = False
        self.server = None
        self.logger = logging.getLogger(__name__)

    def generate_html_dashboard(self, metrics: DashboardMetrics) -> str:
        """
        Generate HTML dashboard.

        Args:
            metrics: Dashboard metrics to display

        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ORB Alert System Dashboard</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #2c3e50; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .status-good {{ color: #27ae60; }}
                .status-warning {{ color: #f39c12; }}
                .status-error {{ color: #e74c3c; }}
                .progress-bar {{ width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }}
                .progress-fill {{ height: 100%; background: #3498db; transition: width 0.3s; }}
                .alert-list {{ max-height: 300px; overflow-y: auto; }}
                .alert-item {{ padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; background: #f8f9fa; }}
                .alert-high {{ border-left-color: #e74c3c; }}
                .alert-medium {{ border-left-color: #f39c12; }}
                .alert-low {{ border-left-color: #27ae60; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ORB Alert System Dashboard</h1>
                    <p>Last Updated: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Uptime: {self._format_uptime(metrics.uptime_seconds)}</p>
                </div>

                <div class="metrics-grid">
                    {self._generate_system_health_card(metrics)}
                    {self._generate_performance_card(metrics)}
                    {self._generate_alerts_card(metrics)}
                    {self._generate_symbols_card(metrics)}
                    {self._generate_validation_card(metrics)}
                    {self._generate_optimization_card(metrics)}
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def _generate_system_health_card(self, metrics: DashboardMetrics) -> str:
        """Generate system health card."""
        memory_status = self._get_status_class(metrics.memory_usage_mb, 800, 900)  # Warning at 800MB, Error at 900MB
        cpu_status = self._get_status_class(metrics.cpu_usage_percent, 70, 85)

        return f"""
        <div class="metric-card">
            <div class="metric-title">System Health</div>
            <div class="metric-value {memory_status}">Memory: {metrics.memory_usage_mb:.1f} MB</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(metrics.memory_usage_mb/1024*100, 100):.1f}%"></div>
            </div>
            <div class="metric-value {cpu_status}">CPU: {metrics.cpu_usage_percent:.1f}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {min(metrics.cpu_usage_percent, 100):.1f}%"></div>
            </div>
            <div>Symbols Monitored: {metrics.symbols_monitored}</div>
            <div>Error Rate: {metrics.error_rate:.2%}</div>
        </div>
        """

    def _generate_performance_card(self, metrics: DashboardMetrics) -> str:
        """Generate performance metrics card."""
        perf = metrics.performance_summary

        return f"""
        <div class="metric-card">
            <div class="metric-title">Performance Metrics</div>
            <div class="metric-value">Avg Processing: {perf.get('avg_duration_ms', 0):.1f}ms</div>
            <div class="metric-value">P95 Latency: {perf.get('p95_duration_ms', 0):.1f}ms</div>
            <div class="metric-value">P99 Latency: {perf.get('p99_duration_ms', 0):.1f}ms</div>
            <div>Operations/min: {perf.get('operations_per_minute', 0):.1f}</div>
            <div>Success Rate: {perf.get('success_rate', 0):.2%}</div>
            <div>Total Operations: {perf.get('total_operations', 0)}</div>
        </div>
        """

    def _generate_alerts_card(self, metrics: DashboardMetrics) -> str:
        """Generate alerts metrics card."""
        alerts = metrics.alert_metrics

        return f"""
        <div class="metric-card">
            <div class="metric-title">Alert Statistics</div>
            <div class="metric-value">Total Alerts: {alerts.get('total_alerts', 0)}</div>
            <div class="metric-value">Alerts/Hour: {alerts.get('alerts_per_hour', 0):.1f}</div>
            <div>High Priority: {alerts.get('high_priority_count', 0)}</div>
            <div>Medium Priority: {alerts.get('medium_priority_count', 0)}</div>
            <div>Low Priority: {alerts.get('low_priority_count', 0)}</div>
            <div>Avg Confidence: {alerts.get('avg_confidence_score', 0):.3f}</div>
            <div>Avg Gen Time: {alerts.get('avg_generation_time_ms', 0):.1f}ms</div>
        </div>
        """

    def _generate_symbols_card(self, metrics: DashboardMetrics) -> str:
        """Generate symbols monitoring card."""
        opt_stats = metrics.optimization_stats

        return f"""
        <div class="metric-card">
            <div class="metric-title">Symbol Management</div>
            <div class="metric-value">Active Symbols: {opt_stats.get('active_symbols', 0)}</div>
            <div class="metric-value">Utilization: {opt_stats.get('symbol_utilization', 0):.1%}</div>
            <div>Max Symbols: {opt_stats.get('max_symbols', 0)}</div>
            <div>Processing Rate: {opt_stats.get('processing_metrics', {}).get('throughput_symbols_per_second', 0):.2f}/s</div>
            <div>Cache Hit Rate: {opt_stats.get('cache_stats', {}).get('hit_rate', 0):.2%}</div>
            <div>Worker Threads: {opt_stats.get('worker_stats', {}).get('max_workers', 0)}</div>
        </div>
        """

    def _generate_validation_card(self, metrics: DashboardMetrics) -> str:
        """Generate data validation card."""
        validation = metrics.validation_summary

        return f"""
        <div class="metric-card">
            <div class="metric-title">Data Validation</div>
            <div class="metric-value">Quality Score: {validation.get('validation_stats', {}).get('quality_score', 0):.2%}</div>
            <div class="metric-value">Total Issues: {validation.get('total_issues', 0)}</div>
            <div>Errors: {validation.get('severity_counts', {}).get('error', 0)}</div>
            <div>Warnings: {validation.get('severity_counts', {}).get('warning', 0)}</div>
            <div>Anomalies: {validation.get('severity_counts', {}).get('anomaly', 0)}</div>
            <div>Error Rate: {validation.get('validation_stats', {}).get('error_rate', 0):.2%}</div>
        </div>
        """

    def _generate_optimization_card(self, metrics: DashboardMetrics) -> str:
        """Generate optimization metrics card."""
        opt = metrics.optimization_stats
        memory_stats = opt.get('memory_stats', {})

        return f"""
        <div class="metric-card">
            <div class="metric-title">Optimization Stats</div>
            <div class="metric-value">Peak Memory: {memory_stats.get('peak_memory_usage_mb', 0):.1f}MB</div>
            <div class="metric-value">GC Collections: {memory_stats.get('gc_collections', 0)}</div>
            <div>Thread Utilization: {opt.get('worker_stats', {}).get('thread_pool_utilization', 0):.1%}</div>
            <div>Cache Size: {opt.get('cache_stats', {}).get('size', 0)}</div>
            <div>Cache Utilization: {opt.get('cache_stats', {}).get('utilization', 0):.1%}</div>
        </div>
        """

    def _get_status_class(self, value: float, warning_threshold: float, error_threshold: float) -> str:
        """Get CSS class based on value and thresholds."""
        if value >= error_threshold:
            return "status-error"
        elif value >= warning_threshold:
            return "status-warning"
        else:
            return "status-good"

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for the ORB Alert system.

    Provides real-time monitoring, alerting, and performance visualization.
    """

    def __init__(self, update_interval: int = 5, export_interval: int = 60):
        """
        Initialize monitoring dashboard.

        Args:
            update_interval: Metrics update interval in seconds
            export_interval: Metrics export interval in seconds
        """
        self.update_interval = update_interval
        self.export_interval = export_interval
        self.running = False
        self.start_time = time.time()

        # Components
        self.dashboard_logger = DashboardLogger()
        self.web_dashboard = WebDashboard()

        # Metrics storage
        self.current_metrics: Optional[DashboardMetrics] = None
        self.metrics_history: List[DashboardMetrics] = []
        self.max_history_size = 1000

        # Background tasks
        self.update_task = None
        self.export_task = None

        # Logging
        self.logger = logging.getLogger(__name__)

        # Alert thresholds
        self.alert_thresholds = {
            'memory_mb': 900,
            'cpu_percent': 85,
            'error_rate': 0.05,  # 5%
            'latency_ms': 500
        }

    async def start(self) -> None:
        """Start the monitoring dashboard."""
        if self.running:
            return

        self.running = True
        self.logger.info("Starting monitoring dashboard")

        # Start background tasks
        self.update_task = asyncio.create_task(self._update_loop())
        self.export_task = asyncio.create_task(self._export_loop())

        # Start performance tracking
        await performance_tracker.start_health_monitoring()

    async def stop(self) -> None:
        """Stop the monitoring dashboard."""
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping monitoring dashboard")

        # Cancel background tasks
        if self.update_task:
            self.update_task.cancel()
        if self.export_task:
            self.export_task.cancel()

        # Stop performance tracking
        performance_tracker.stop_health_monitoring()

    async def _update_loop(self) -> None:
        """Background metrics update loop."""
        while self.running:
            try:
                # Collect metrics from all sources
                metrics = await self._collect_metrics()

                # Update current metrics
                self.current_metrics = metrics

                # Add to history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]

                # Check for alerts
                self._check_alert_thresholds(metrics)

                # Log system health
                self.dashboard_logger.log_system_health(
                    metrics.memory_usage_mb / 1024 * 100,  # Convert to percentage
                    metrics.cpu_usage_percent
                )

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _export_loop(self) -> None:
        """Background metrics export loop."""
        while self.running:
            try:
                if self.current_metrics:
                    # Export to JSON
                    await self._export_metrics_to_file()

                    # Export to HTML dashboard
                    await self._export_html_dashboard()

                await asyncio.sleep(self.export_interval)

            except Exception as e:
                self.logger.error(f"Error in dashboard export loop: {e}")
                await asyncio.sleep(self.export_interval)

    async def _collect_metrics(self) -> DashboardMetrics:
        """Collect metrics from all monitoring sources."""
        timestamp = datetime.now()
        uptime = time.time() - self.start_time

        # Get system health
        system_health = performance_tracker.get_system_health().__dict__
        system_health['timestamp'] = system_health['timestamp'].isoformat()

        # Get performance summary
        performance_summary = performance_tracker.get_performance_summary(lookback_minutes=60)

        # Get alert metrics
        alert_metrics = performance_tracker.get_alert_performance()

        # Get optimization stats
        optimization_stats = performance_optimizer.get_optimization_stats()

        # Get validation summary
        validation_summary = data_validator.get_validation_summary(hours=24)

        # Calculate derived metrics
        alert_rate = alert_metrics.get('alerts_per_hour', 0)
        error_rate = performance_summary.get('error_count', 0) / max(performance_summary.get('total_operations', 1), 1)
        symbols_monitored = optimization_stats.get('active_symbols', 0)
        memory_usage_mb = optimization_stats.get('memory_stats', {}).get('current_memory_mb', 0)
        cpu_usage_percent = system_health.get('cpu_percent', 0)

        return DashboardMetrics(
            timestamp=timestamp,
            system_health=system_health,
            performance_summary=performance_summary,
            alert_metrics=alert_metrics,
            optimization_stats=optimization_stats,
            validation_summary=validation_summary,
            uptime_seconds=uptime,
            alert_rate=alert_rate,
            error_rate=error_rate,
            symbols_monitored=symbols_monitored,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent
        )

    def _check_alert_thresholds(self, metrics: DashboardMetrics) -> None:
        """Check metrics against alert thresholds."""
        # Memory threshold
        if metrics.memory_usage_mb > self.alert_thresholds['memory_mb']:
            self.dashboard_logger.log_error(
                "MEMORY_THRESHOLD", 
                f"Memory usage {metrics.memory_usage_mb:.1f}MB exceeds threshold {self.alert_thresholds['memory_mb']}MB"
            )

        # CPU threshold
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_percent']:
            self.dashboard_logger.log_error(
                "CPU_THRESHOLD",
                f"CPU usage {metrics.cpu_usage_percent:.1f}% exceeds threshold {self.alert_thresholds['cpu_percent']}%"
            )

        # Error rate threshold
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            self.dashboard_logger.log_error(
                "ERROR_RATE_THRESHOLD",
                f"Error rate {metrics.error_rate:.2%} exceeds threshold {self.alert_thresholds['error_rate']:.2%}"
            )

        # Latency threshold
        avg_latency = metrics.performance_summary.get('avg_duration_ms', 0)
        if avg_latency > self.alert_thresholds['latency_ms']:
            self.dashboard_logger.log_performance_issue(
                "LATENCY_THRESHOLD",
                avg_latency,
                self.alert_thresholds['latency_ms']
            )

    async def _export_metrics_to_file(self) -> None:
        """Export current metrics to JSON file."""
        if not self.current_metrics:
            return

        # Convert to dictionary
        metrics_dict = asdict(self.current_metrics)
        metrics_dict['timestamp'] = self.current_metrics.timestamp.isoformat()

        # Export path
        export_path = Path("monitoring/dashboard_metrics.json")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(export_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    async def _export_html_dashboard(self) -> None:
        """Export HTML dashboard."""
        if not self.current_metrics:
            return

        # Generate HTML
        html_content = self.web_dashboard.generate_html_dashboard(self.current_metrics)

        # Export path
        export_path = Path("monitoring/dashboard.html")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(export_path, 'w') as f:
            f.write(html_content)

    def get_current_metrics(self) -> Optional[DashboardMetrics]:
        """Get current dashboard metrics."""
        return self.current_metrics

    def get_metrics_history(self, hours: int = 24) -> List[DashboardMetrics]:
        """
        Get metrics history for specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            List of DashboardMetrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def generate_status_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive status report.

        Returns:
            Dictionary with status report
        """
        if not self.current_metrics:
            return {'status': 'No metrics available'}

        metrics = self.current_metrics

        # Overall system status
        status = "HEALTHY"
        if (metrics.memory_usage_mb > self.alert_thresholds['memory_mb'] or
            metrics.cpu_usage_percent > self.alert_thresholds['cpu_percent'] or
            metrics.error_rate > self.alert_thresholds['error_rate']):
            status = "WARNING"

        # Performance grade
        avg_latency = metrics.performance_summary.get('avg_duration_ms', 0)
        if avg_latency < 100:
            performance_grade = "EXCELLENT"
        elif avg_latency < 250:
            performance_grade = "GOOD"
        elif avg_latency < 500:
            performance_grade = "ACCEPTABLE"
        else:
            performance_grade = "POOR"

        return {
            'timestamp': metrics.timestamp.isoformat(),
            'overall_status': status,
            'performance_grade': performance_grade,
            'uptime_hours': metrics.uptime_seconds / 3600,
            'symbols_monitored': metrics.symbols_monitored,
            'alerts_generated': metrics.alert_metrics.get('total_alerts', 0),
            'system_health': {
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'error_rate': metrics.error_rate
            },
            'performance_metrics': {
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': metrics.performance_summary.get('p95_duration_ms', 0),
                'throughput_ops_per_min': metrics.performance_summary.get('operations_per_minute', 0)
            },
            'quality_metrics': {
                'data_quality_score': metrics.validation_summary.get('validation_stats', {}).get('quality_score', 0),
                'cache_hit_rate': metrics.optimization_stats.get('cache_stats', {}).get('hit_rate', 0)
            }
        }


# Global monitoring dashboard instance
monitoring_dashboard = MonitoringDashboard()