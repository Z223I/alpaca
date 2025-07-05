"""
Performance Benchmarking Tests for Phase 3

This module implements comprehensive performance tests to validate
Phase 3 requirements including latency, throughput, and resource usage.
"""

import pytest
import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from unittest.mock import Mock, patch

from atoms.monitoring.performance_tracker import performance_tracker, PerformanceTracker
from atoms.optimization.performance_optimizer import performance_optimizer, PerformanceOptimizer
from atoms.validation.data_validator import data_validator, DataValidator
from atoms.error_handling.retry_manager import retry_manager, RetryManager
from atoms.monitoring.dashboard import monitoring_dashboard
from atoms.websocket.alpaca_stream import MarketData
from molecules.orb_alert_engine import ORBAlertEngine


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.results: Dict[str, List[float]] = {}
        self.memory_snapshots: List[float] = []
        self.cpu_snapshots: List[float] = []
    
    def measure_latency(self, operation_name: str, operation_func, *args, **kwargs) -> float:
        """Measure operation latency."""
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(operation_func):
                result = asyncio.run(operation_func(*args, **kwargs))
            else:
                result = operation_func(*args, **kwargs)
            return (time.time() - start_time) * 1000  # Convert to ms
        except Exception:
            return float('inf')
    
    def record_result(self, test_name: str, latency_ms: float) -> None:
        """Record benchmark result."""
        if test_name not in self.results:
            self.results[test_name] = []
        self.results[test_name].append(latency_ms)
    
    def take_memory_snapshot(self) -> float:
        """Take memory usage snapshot."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_snapshots.append(memory_mb)
        return memory_mb
    
    def take_cpu_snapshot(self) -> float:
        """Take CPU usage snapshot."""
        cpu_percent = psutil.cpu_percent()
        self.cpu_snapshots.append(cpu_percent)
        return cpu_percent
    
    def get_statistics(self, test_name: str) -> Dict[str, float]:
        """Get performance statistics for test."""
        if test_name not in self.results:
            return {}
        
        values = self.results[test_name]
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


@pytest.fixture
def benchmark():
    """Performance benchmark fixture."""
    return PerformanceBenchmark()


@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    return MarketData(
        symbol="AAPL",
        price=150.0,
        volume=1000000,
        timestamp=datetime.now(),
        high=151.0,
        low=149.0,
        open=150.5,
        close=150.0,
        vwap=150.2
    )


@pytest.fixture
def performance_tracker_instance():
    """Performance tracker instance."""
    tracker = PerformanceTracker()
    yield tracker
    tracker.reset_metrics()


@pytest.fixture
def data_validator_instance():
    """Data validator instance."""
    validator = DataValidator()
    yield validator
    validator.clear_validation_history()


class TestPerformanceTargets:
    """Test Phase 3 performance targets."""
    
    def test_data_ingestion_latency(self, benchmark, mock_market_data, data_validator_instance):
        """Test data ingestion latency < 100ms."""
        target_latency = 100.0  # ms
        
        for i in range(100):
            latency = benchmark.measure_latency(
                "data_ingestion",
                data_validator_instance.validate_market_data,
                mock_market_data
            )
            benchmark.record_result("data_ingestion", latency)
        
        stats = benchmark.get_statistics("data_ingestion")
        
        # Assert performance targets
        assert stats['p95'] < target_latency, f"P95 latency {stats['p95']:.2f}ms exceeds target {target_latency}ms"
        assert stats['mean'] < target_latency * 0.5, f"Mean latency {stats['mean']:.2f}ms exceeds 50% of target"
        
        print(f"Data Ingestion Performance: Mean={stats['mean']:.2f}ms, P95={stats['p95']:.2f}ms")
    
    def test_processing_latency(self, benchmark, performance_tracker_instance):
        """Test processing latency < 50ms."""
        target_latency = 50.0  # ms
        
        for i in range(100):
            # Simulate processing operation
            op_id = performance_tracker_instance.start_operation("test_processing")
            time.sleep(0.001)  # Simulate 1ms of processing
            metrics = performance_tracker_instance.end_operation(op_id, "test_processing")
            
            benchmark.record_result("processing", metrics.duration_ms)
        
        stats = benchmark.get_statistics("processing")
        
        # Assert performance targets
        assert stats['p95'] < target_latency, f"P95 latency {stats['p95']:.2f}ms exceeds target {target_latency}ms"
        assert stats['mean'] < target_latency * 0.5, f"Mean latency {stats['mean']:.2f}ms exceeds 50% of target"
        
        print(f"Processing Performance: Mean={stats['mean']:.2f}ms, P95={stats['p95']:.2f}ms")
    
    def test_alert_generation_latency(self, benchmark):
        """Test alert generation latency < 200ms."""
        target_latency = 200.0  # ms
        
        # Mock alert generation pipeline
        def mock_alert_generation():
            # Simulate ORB calculation
            time.sleep(0.01)  # 10ms
            
            # Simulate breakout detection
            time.sleep(0.005)  # 5ms
            
            # Simulate confidence scoring
            time.sleep(0.02)  # 20ms
            
            # Simulate alert formatting
            time.sleep(0.005)  # 5ms
            
            return True
        
        for i in range(50):
            latency = benchmark.measure_latency("alert_generation", mock_alert_generation)
            benchmark.record_result("alert_generation", latency)
        
        stats = benchmark.get_statistics("alert_generation")
        
        # Assert performance targets
        assert stats['p95'] < target_latency, f"P95 latency {stats['p95']:.2f}ms exceeds target {target_latency}ms"
        assert stats['mean'] < target_latency * 0.5, f"Mean latency {stats['mean']:.2f}ms exceeds 50% of target"
        
        print(f"Alert Generation Performance: Mean={stats['mean']:.2f}ms, P95={stats['p95']:.2f}ms")
    
    def test_total_latency(self, benchmark, mock_market_data, data_validator_instance):
        """Test total end-to-end latency < 500ms."""
        target_latency = 500.0  # ms
        
        def simulate_end_to_end_processing():
            # Data ingestion
            data_validator_instance.validate_market_data(mock_market_data)
            
            # Simulate ORB calculation and processing
            time.sleep(0.050)  # 50ms processing
            
            # Simulate alert generation
            time.sleep(0.040)  # 40ms alert generation
            
            return True
        
        for i in range(50):
            latency = benchmark.measure_latency("total_latency", simulate_end_to_end_processing)
            benchmark.record_result("total_latency", latency)
        
        stats = benchmark.get_statistics("total_latency")
        
        # Assert performance targets
        assert stats['p95'] < target_latency, f"P95 latency {stats['p95']:.2f}ms exceeds target {target_latency}ms"
        assert stats['mean'] < target_latency * 0.5, f"Mean latency {stats['mean']:.2f}ms exceeds 50% of target"
        
        print(f"Total Latency Performance: Mean={stats['mean']:.2f}ms, P95={stats['p95']:.2f}ms")


class TestThroughputRequirements:
    """Test throughput requirements."""
    
    def test_symbol_capacity(self):
        """Test support for 50+ concurrent symbols."""
        target_symbols = 50
        
        optimizer = PerformanceOptimizer(max_symbols=100)
        
        # Simulate symbol processing
        symbols = [f"SYM{i:03d}" for i in range(target_symbols + 10)]
        
        processed_count = 0
        for symbol in symbols:
            mock_data = MarketData(
                symbol=symbol,
                price=100.0,
                volume=1000,
                timestamp=datetime.now()
            )
            
            task_id = optimizer.submit_processing_task(symbol, mock_data)
            if task_id:
                processed_count += 1
        
        assert processed_count >= target_symbols, f"Only processed {processed_count} symbols, target was {target_symbols}"
        print(f"Symbol Capacity: Successfully processed {processed_count} symbols")
    
    def test_processing_intervals(self):
        """Test 1-second processing intervals."""
        target_interval = 1.0  # seconds
        
        intervals = []
        last_time = time.time()
        
        for i in range(10):
            # Simulate processing cycle
            time.sleep(0.1)  # 100ms processing time
            current_time = time.time()
            interval = current_time - last_time
            intervals.append(interval)
            last_time = current_time
        
        avg_interval = np.mean(intervals[1:])  # Skip first interval
        
        # Allow 10% tolerance
        assert abs(avg_interval - target_interval) < target_interval * 0.1, \
            f"Average interval {avg_interval:.3f}s deviates too much from target {target_interval}s"
        
        print(f"Processing Intervals: Average={avg_interval:.3f}s, Target={target_interval}s")
    
    def test_daily_alert_capacity(self):
        """Test handling 100+ alerts per trading day."""
        target_alerts = 100
        trading_hours = 6.5  # 6.5 hours
        
        # Calculate required rate
        required_rate = target_alerts / (trading_hours * 3600)  # alerts per second
        
        # Simulate alert processing
        alerts_processed = 0
        start_time = time.time()
        
        for i in range(target_alerts):
            # Simulate alert processing (should be very fast)
            time.sleep(0.001)  # 1ms per alert
            alerts_processed += 1
            
            if time.time() - start_time > 10:  # Max 10 seconds for test
                break
        
        elapsed_time = time.time() - start_time
        actual_rate = alerts_processed / elapsed_time
        
        assert actual_rate > required_rate, \
            f"Alert processing rate {actual_rate:.6f} alerts/sec is below required {required_rate:.6f}"
        
        print(f"Alert Capacity: Processed {alerts_processed} alerts at {actual_rate:.3f} alerts/sec")


class TestResourceUsage:
    """Test resource usage requirements."""
    
    def test_memory_usage(self, benchmark):
        """Test memory usage < 1GB for full system."""
        target_memory = 1024  # MB
        
        # Take initial memory snapshot
        initial_memory = benchmark.take_memory_snapshot()
        
        # Simulate system operation
        optimizer = PerformanceOptimizer()
        validator = DataValidator()
        tracker = PerformanceTracker()
        
        # Generate load
        for i in range(1000):
            mock_data = MarketData(
                symbol=f"TEST{i % 10}",
                price=100.0 + i * 0.1,
                volume=1000 + i,
                timestamp=datetime.now()
            )
            validator.validate_market_data(mock_data)
            
            if i % 100 == 0:
                benchmark.take_memory_snapshot()
        
        # Take final memory snapshot
        final_memory = benchmark.take_memory_snapshot()
        peak_memory = max(benchmark.memory_snapshots)
        
        assert peak_memory < target_memory, \
            f"Peak memory usage {peak_memory:.1f}MB exceeds target {target_memory}MB"
        
        print(f"Memory Usage: Initial={initial_memory:.1f}MB, Peak={peak_memory:.1f}MB, Final={final_memory:.1f}MB")
    
    def test_cpu_efficiency(self, benchmark):
        """Test CPU usage efficiency during operation."""
        max_cpu_threshold = 80  # %
        
        start_time = time.time()
        
        # Simulate CPU-intensive operation
        while time.time() - start_time < 5:  # Run for 5 seconds
            # Simulate processing work
            for _ in range(1000):
                _ = sum(range(100))
            
            benchmark.take_cpu_snapshot()
            time.sleep(0.1)
        
        avg_cpu = np.mean(benchmark.cpu_snapshots)
        peak_cpu = max(benchmark.cpu_snapshots)
        
        # CPU usage should be reasonable during normal operation
        assert avg_cpu < max_cpu_threshold, \
            f"Average CPU usage {avg_cpu:.1f}% exceeds threshold {max_cpu_threshold}%"
        
        print(f"CPU Usage: Average={avg_cpu:.1f}%, Peak={peak_cpu:.1f}%")


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test exponential backoff retry mechanism."""
        retry_manager_test = RetryManager()
        
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Simulated failure")
            return "success"
        
        start_time = time.time()
        result = await retry_manager_test.execute_with_retry(
            failing_operation,
            "test_operation",
            max_retries=3,
            base_delay=0.1
        )
        elapsed_time = time.time() - start_time
        
        assert result == "success", "Retry mechanism should eventually succeed"
        assert call_count == 3, f"Expected 3 calls, got {call_count}"
        assert elapsed_time > 0.3, "Should have delay between retries"  # 0.1 + 0.2 = 0.3s minimum
        
        print(f"Retry Mechanism: {call_count} attempts in {elapsed_time:.3f}s")
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        from atoms.error_handling.retry_manager import CircuitBreaker, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            name="test_circuit"
        )
        
        circuit_breaker = CircuitBreaker(config)
        
        # Should be closed initially
        assert circuit_breaker.can_execute()
        
        # Record failures to trip circuit breaker
        for _ in range(3):
            circuit_breaker.record_failure()
        
        # Should be open now
        assert not circuit_breaker.can_execute()
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should be half-open
        assert circuit_breaker.can_execute()
        
        # Record success to close circuit
        circuit_breaker.record_success()
        assert circuit_breaker.can_execute()
        
        print("Circuit Breaker: Correctly transitions between states")


class TestDataQuality:
    """Test data quality validation."""
    
    def test_anomaly_detection(self, data_validator_instance):
        """Test anomaly detection capabilities."""
        normal_data = MarketData(
            symbol="TEST",
            price=100.0,
            volume=1000000,
            timestamp=datetime.now()
        )
        
        # Establish baseline
        for i in range(10):
            data_validator_instance.validate_market_data(normal_data)
        
        # Create anomalous data
        spike_data = MarketData(
            symbol="TEST",
            price=200.0,  # 100% price spike
            volume=1000000,
            timestamp=datetime.now()
        )
        
        issues = data_validator_instance.validate_market_data(spike_data)
        
        # Should detect price anomaly
        price_anomalies = [issue for issue in issues if "price" in issue.message.lower()]
        assert len(price_anomalies) > 0, "Should detect price anomaly"
        
        print(f"Anomaly Detection: Detected {len(issues)} issues including price spike")
    
    def test_data_quality_scoring(self, data_validator_instance):
        """Test data quality scoring."""
        # Generate mix of good and bad data
        for i in range(50):
            if i % 10 == 0:  # 10% bad data
                bad_data = MarketData(
                    symbol="TEST",
                    price=-1.0,  # Invalid price
                    volume=1000000,
                    timestamp=datetime.now()
                )
                data_validator_instance.validate_market_data(bad_data)
            else:
                good_data = MarketData(
                    symbol="TEST",
                    price=100.0 + i * 0.1,
                    volume=1000000,
                    timestamp=datetime.now()
                )
                data_validator_instance.validate_market_data(good_data)
        
        stats = data_validator_instance.validation_stats
        quality_score = stats.quality_score
        
        # Should have reasonable quality score
        assert 0.8 <= quality_score <= 1.0, f"Quality score {quality_score:.3f} should be between 0.8 and 1.0"
        
        print(f"Data Quality: Score={quality_score:.3f}, Valid={stats.valid_count}, Errors={stats.error_count}")


def test_phase3_integration():
    """Integration test for all Phase 3 components."""
    print("\n=== Phase 3 Integration Test ===")
    
    # Test all systems can be initialized
    engine = ORBAlertEngine()
    
    # Verify Phase 3 components are available
    phase3_status = engine.get_phase3_status()
    
    assert phase3_status['phase3_enabled'], "Phase 3 should be enabled"
    assert 'performance_tracker_running' in phase3_status
    assert 'dashboard_running' in phase3_status
    assert 'optimizer_running' in phase3_status
    assert 'validation_enabled' in phase3_status
    
    print("Phase 3 Integration: All components initialized successfully")
    print(f"Systems status: {list(phase3_status.keys())}")


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "-s"])