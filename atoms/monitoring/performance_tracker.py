"""
Performance Monitoring and Metrics Collection

This module implements comprehensive performance tracking for the ORB Alert system,
targeting Phase 3 production-ready requirements:
- Data Ingestion: < 100ms
- Processing: < 50ms  
- Alert Generation: < 200ms
- Total Latency: < 500ms
"""

import time
import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from threading import Lock
import json
from pathlib import Path

from atoms.config.alert_config import config


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool = True
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    def __post_init__(self):
        """Calculate duration if not provided."""
        if self.duration_ms == 0.0:
            self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class SystemHealthMetrics:
    """System health and resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    thread_count: int
    process_uptime_seconds: float


@dataclass
class AlertMetrics:
    """Alert generation and effectiveness metrics."""
    total_alerts: int = 0
    high_priority_alerts: int = 0
    medium_priority_alerts: int = 0
    low_priority_alerts: int = 0
    avg_confidence_score: float = 0.0
    avg_generation_time_ms: float = 0.0
    success_rate: float = 0.0
    false_positive_rate: float = 0.0
    symbols_monitored: int = 0
    alerts_per_hour: float = 0.0


class PerformanceTracker:
    """
    Comprehensive performance tracking system for ORB Alert engine.

    Tracks latency, throughput, resource usage, and alert effectiveness.
    """

    def __init__(self, max_history_size: int = 10000):
        """
        Initialize performance tracker.

        Args:
            max_history_size: Maximum number of metrics to keep in memory
        """
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.health_history: deque = deque(maxlen=1000)  # Keep 1000 health snapshots
        self.alert_metrics = AlertMetrics()

        # Thread-safe access
        self.metrics_lock = Lock()
        self.health_lock = Lock()

        # Performance targets (from Phase 3 requirements)
        self.performance_targets = {
            'data_ingestion_ms': 100,
            'processing_ms': 50,
            'alert_generation_ms': 200,
            'total_latency_ms': 500,
            'memory_usage_mb': 1024,  # 1GB limit
            'cpu_usage_percent': 80,
            'success_rate_percent': 90
        }

        # Operation timers
        self.active_operations: Dict[str, float] = {}
        self.operation_counts: Dict[str, int] = {}

        # Logging
        self.logger = logging.getLogger(__name__)

        # Health monitoring
        self.health_monitor_running = False
        self.health_monitor_interval = 5.0  # seconds

        # Process start time
        self.process_start_time = time.time()

    def start_operation(self, operation_name: str, operation_id: str = None) -> str:
        """
        Start timing an operation.

        Args:
            operation_name: Name of the operation
            operation_id: Optional unique ID for this operation instance

        Returns:
            Operation ID for use with end_operation
        """
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

        start_time = time.time()
        self.active_operations[operation_id] = start_time

        # Increment operation count
        self.operation_counts[operation_name] = self.operation_counts.get(operation_name, 0) + 1

        return operation_id

    def end_operation(self, operation_id: str, operation_name: str = None, 
                     success: bool = True, error_message: str = None) -> PerformanceMetrics:
        """
        End timing an operation and record metrics.

        Args:
            operation_id: ID returned from start_operation
            operation_name: Name of the operation (extracted from ID if not provided)
            success: Whether the operation succeeded
            error_message: Error message if operation failed

        Returns:
            PerformanceMetrics object
        """
        end_time = time.time()
        start_time = self.active_operations.pop(operation_id, end_time)

        if operation_name is None:
            operation_name = operation_id.split('_')[0]

        # Get current system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=(end_time - start_time) * 1000,
            success=success,
            error_message=error_message,
            memory_usage_mb=memory_info.rss / 1024 / 1024,
            cpu_usage_percent=cpu_percent
        )

        # Store metrics
        with self.metrics_lock:
            self.metrics_history.append(metrics)

        # Log performance issues
        self._check_performance_thresholds(metrics)

        return metrics

    def record_alert_generated(self, priority: str, confidence_score: float, 
                              generation_time_ms: float, symbol: str) -> None:
        """
        Record alert generation metrics.

        Args:
            priority: Alert priority (HIGH, MEDIUM, LOW)
            confidence_score: Confidence score (0.0 to 1.0)
            generation_time_ms: Time taken to generate alert
            symbol: Trading symbol
        """
        self.alert_metrics.total_alerts += 1

        if priority == "HIGH":
            self.alert_metrics.high_priority_alerts += 1
        elif priority == "MEDIUM":
            self.alert_metrics.medium_priority_alerts += 1
        else:
            self.alert_metrics.low_priority_alerts += 1

        # Update averages
        total = self.alert_metrics.total_alerts
        self.alert_metrics.avg_confidence_score = (
            (self.alert_metrics.avg_confidence_score * (total - 1) + confidence_score) / total
        )
        self.alert_metrics.avg_generation_time_ms = (
            (self.alert_metrics.avg_generation_time_ms * (total - 1) + generation_time_ms) / total
        )

    def record_alert_outcome(self, symbol: str, success: bool, 
                           profit_loss: float = 0.0) -> None:
        """
        Record alert trading outcome for effectiveness tracking.

        Args:
            symbol: Trading symbol
            success: Whether the alert led to a profitable trade
            profit_loss: Profit/loss amount
        """
        # This would be implemented with actual trading outcome tracking
        # For now, we'll track basic success/failure rates
        if success:
            self.alert_metrics.success_rate = (
                self.alert_metrics.success_rate * 0.9 + 1.0 * 0.1  # Moving average
            )
        else:
            self.alert_metrics.false_positive_rate = (
                self.alert_metrics.false_positive_rate * 0.9 + 1.0 * 0.1
            )

    def get_system_health(self) -> SystemHealthMetrics:
        """
        Get current system health metrics.

        Returns:
            SystemHealthMetrics object
        """
        # System-wide metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network metrics
        network = psutil.net_io_counters()

        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        return SystemHealthMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            active_connections=len(psutil.net_connections()),
            thread_count=process.num_threads(),
            process_uptime_seconds=time.time() - self.process_start_time
        )

    async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.health_monitor_running:
            return

        self.health_monitor_running = True
        self.logger.info("Started health monitoring")

        while self.health_monitor_running:
            try:
                health_metrics = self.get_system_health()

                with self.health_lock:
                    self.health_history.append(health_metrics)

                # Check for health issues
                self._check_health_thresholds(health_metrics)

                await asyncio.sleep(self.health_monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.health_monitor_interval)

    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self.health_monitor_running = False
        self.logger.info("Stopped health monitoring")

    def get_performance_summary(self, operation_name: str = None, 
                              lookback_minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance summary for specified operation or all operations.

        Args:
            operation_name: Specific operation to analyze (None for all)
            lookback_minutes: Minutes of history to analyze

        Returns:
            Dictionary with performance summary
        """
        cutoff_time = time.time() - (lookback_minutes * 60)

        with self.metrics_lock:
            relevant_metrics = [
                m for m in self.metrics_history 
                if m.end_time >= cutoff_time and 
                (operation_name is None or m.operation_name == operation_name)
            ]

        if not relevant_metrics:
            return {}

        # Calculate statistics
        durations = [m.duration_ms for m in relevant_metrics]
        success_count = sum(1 for m in relevant_metrics if m.success)

        return {
            'operation_name': operation_name or 'all_operations',
            'total_operations': len(relevant_metrics),
            'success_count': success_count,
            'success_rate': success_count / len(relevant_metrics) if relevant_metrics else 0,
            'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
            'min_duration_ms': min(durations) if durations else 0,
            'max_duration_ms': max(durations) if durations else 0,
            'p95_duration_ms': self._calculate_percentile(durations, 95),
            'p99_duration_ms': self._calculate_percentile(durations, 99),
            'operations_per_minute': len(relevant_metrics) / lookback_minutes,
            'error_count': len(relevant_metrics) - success_count,
            'avg_memory_mb': sum(m.memory_usage_mb for m in relevant_metrics) / len(relevant_metrics),
            'avg_cpu_percent': sum(m.cpu_usage_percent for m in relevant_metrics) / len(relevant_metrics)
        }

    def get_alert_performance(self) -> Dict[str, Any]:
        """
        Get alert generation performance metrics.

        Returns:
            Dictionary with alert performance data
        """
        # Calculate alerts per hour
        if self.alert_metrics.total_alerts > 0:
            uptime_hours = (time.time() - self.process_start_time) / 3600
            alerts_per_hour = self.alert_metrics.total_alerts / uptime_hours if uptime_hours > 0 else 0
        else:
            alerts_per_hour = 0

        return {
            'total_alerts': self.alert_metrics.total_alerts,
            'high_priority_count': self.alert_metrics.high_priority_alerts,
            'medium_priority_count': self.alert_metrics.medium_priority_alerts,
            'low_priority_count': self.alert_metrics.low_priority_alerts,
            'avg_confidence_score': self.alert_metrics.avg_confidence_score,
            'avg_generation_time_ms': self.alert_metrics.avg_generation_time_ms,
            'success_rate': self.alert_metrics.success_rate,
            'false_positive_rate': self.alert_metrics.false_positive_rate,
            'alerts_per_hour': alerts_per_hour,
            'priority_distribution': {
                'HIGH': self.alert_metrics.high_priority_alerts,
                'MEDIUM': self.alert_metrics.medium_priority_alerts,
                'LOW': self.alert_metrics.low_priority_alerts
            }
        }

    def export_metrics_to_file(self, filepath: str) -> None:
        """
        Export performance metrics to JSON file.

        Args:
            filepath: Path to output file
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': self.get_performance_summary(),
            'alert_performance': self.get_alert_performance(),
            'system_health': self.get_system_health().__dict__,
            'performance_targets': self.performance_targets,
            'operation_counts': self.operation_counts
        }

        # Convert datetime objects to ISO format
        if 'system_health' in export_data:
            export_data['system_health']['timestamp'] = export_data['system_health']['timestamp'].isoformat()

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Exported performance metrics to {filepath}")

    def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check if performance metrics exceed thresholds."""
        operation_target = self.performance_targets.get(f"{metrics.operation_name}_ms")

        if operation_target and metrics.duration_ms > operation_target:
            self.logger.warning(
                f"Performance threshold exceeded: {metrics.operation_name} "
                f"took {metrics.duration_ms:.2f}ms (target: {operation_target}ms)"
            )

        if metrics.memory_usage_mb > self.performance_targets['memory_usage_mb']:
            self.logger.warning(
                f"Memory usage high: {metrics.memory_usage_mb:.2f}MB "
                f"(target: {self.performance_targets['memory_usage_mb']}MB)"
            )

        if metrics.cpu_usage_percent > self.performance_targets['cpu_usage_percent']:
            self.logger.warning(
                f"CPU usage high: {metrics.cpu_usage_percent:.2f}% "
                f"(target: {self.performance_targets['cpu_usage_percent']}%)"
            )

    def _check_health_thresholds(self, health: SystemHealthMetrics) -> None:
        """Check if system health metrics exceed thresholds."""
        if health.memory_percent > 90:
            self.logger.error(f"Critical memory usage: {health.memory_percent:.1f}%")
            self.logger.warning(f"High memory usage: {health.memory_percent:.1f}%")

        if health.cpu_percent > 90:
            self.logger.error(f"Critical CPU usage: {health.cpu_percent:.1f}%")
        elif health.cpu_percent > 80:
            self.logger.warning(f"High CPU usage: {health.cpu_percent:.1f}%")

        if health.disk_usage_percent > 90:
            self.logger.error(f"Critical disk usage: {health.disk_usage_percent:.1f}%")

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile for a list of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def reset_metrics(self) -> None:
        """Reset all metrics and history."""
        with self.metrics_lock:
            self.metrics_history.clear()

        with self.health_lock:
            self.health_history.clear()

        self.alert_metrics = AlertMetrics()
        self.operation_counts.clear()
        self.active_operations.clear()

        self.logger.info("Reset all performance metrics")


# Global performance tracker instance
performance_tracker = PerformanceTracker()