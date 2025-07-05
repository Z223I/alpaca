"""
Performance Optimization System

This module implements performance optimizations for handling 50+ concurrent symbols
with sub-500ms latency requirements and efficient resource utilization.
"""

import asyncio
import threading
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import concurrent.futures
import weakref
import logging
from queue import Queue, PriorityQueue
import psutil
import numpy as np

from atoms.websocket.alpaca_stream import MarketData
from atoms.config.alert_config import config
from atoms.monitoring.performance_tracker import performance_tracker


@dataclass
class ProcessingTask:
    """Task for processing in the optimization pipeline."""
    priority: int
    timestamp: datetime
    symbol: str
    data: MarketData
    task_id: str
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority < other.priority


@dataclass
class OptimizationMetrics:
    """Metrics for performance optimization."""
    symbols_processed: int = 0
    total_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    queue_depth: int = 0
    thread_pool_utilization: float = 0.0
    gc_collections: int = 0
    
    @property
    def avg_processing_time(self) -> float:
        """Average processing time per symbol."""
        return self.total_processing_time / self.symbols_processed if self.symbols_processed > 0 else 0.0


class SymbolLoadBalancer:
    """Load balancer for distributing symbols across processing threads."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize load balancer.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.worker_loads: Dict[int, int] = {i: 0 for i in range(max_workers)}
        self.symbol_assignments: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
    
    def assign_symbol(self, symbol: str) -> int:
        """
        Assign symbol to least loaded worker.
        
        Args:
            symbol: Symbol to assign
            
        Returns:
            Worker ID
        """
        if symbol in self.symbol_assignments:
            return self.symbol_assignments[symbol]
        
        # Find least loaded worker
        min_load = min(self.worker_loads.values())
        worker_id = next(worker for worker, load in self.worker_loads.items() if load == min_load)
        
        # Assign symbol to worker
        self.symbol_assignments[symbol] = worker_id
        self.worker_loads[worker_id] += 1
        
        self.logger.debug(f"Assigned {symbol} to worker {worker_id} (load: {self.worker_loads[worker_id]})")
        return worker_id
    
    def get_worker_load(self, worker_id: int) -> int:
        """Get current load for worker."""
        return self.worker_loads.get(worker_id, 0)
    
    def get_symbol_worker(self, symbol: str) -> Optional[int]:
        """Get assigned worker for symbol."""
        return self.symbol_assignments.get(symbol)
    
    def rebalance(self) -> None:
        """Rebalance symbol assignments across workers."""
        if not self.symbol_assignments:
            return
        
        # Calculate target load per worker
        total_symbols = len(self.symbol_assignments)
        target_load = total_symbols // self.max_workers
        
        # Reset loads
        self.worker_loads = {i: 0 for i in range(self.max_workers)}
        
        # Redistribute symbols
        symbols = list(self.symbol_assignments.keys())
        for i, symbol in enumerate(symbols):
            worker_id = i % self.max_workers
            self.symbol_assignments[symbol] = worker_id
            self.worker_loads[worker_id] += 1
        
        self.logger.info(f"Rebalanced {total_symbols} symbols across {self.max_workers} workers")


class MemoryOptimizer:
    """Memory optimization and garbage collection management."""
    
    def __init__(self, max_memory_mb: int = 1024):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.gc_threshold = max_memory_mb * 0.8  # Trigger GC at 80% of max
        self.last_gc_time = time.time()
        self.gc_interval = 300  # 5 minutes
        self.logger = logging.getLogger(__name__)
        
        # Weak references for automatic cleanup
        self.weak_refs: Set[weakref.ref] = set()
    
    def check_memory_usage(self) -> float:
        """
        Check current memory usage.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    
    def should_run_gc(self) -> bool:
        """Check if garbage collection should be run."""
        current_memory = self.check_memory_usage()
        time_since_gc = time.time() - self.last_gc_time
        
        return (current_memory > self.gc_threshold or 
                time_since_gc > self.gc_interval)
    
    def run_gc(self) -> Dict[str, Any]:
        """
        Run garbage collection and return statistics.
        
        Returns:
            Dictionary with GC statistics
        """
        start_time = time.time()
        memory_before = self.check_memory_usage()
        
        # Clean up weak references
        self.clean_weak_refs()
        
        # Run garbage collection
        collected_objects = gc.collect()
        
        memory_after = self.check_memory_usage()
        gc_time = time.time() - start_time
        self.last_gc_time = time.time()
        
        stats = {
            'collected_objects': collected_objects,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_before - memory_after,
            'gc_time_seconds': gc_time
        }
        
        self.logger.info(f"GC completed: freed {stats['memory_freed_mb']:.2f}MB in {gc_time:.3f}s")
        return stats
    
    def clean_weak_refs(self) -> None:
        """Clean up dead weak references."""
        dead_refs = [ref for ref in self.weak_refs if ref() is None]
        for ref in dead_refs:
            self.weak_refs.discard(ref)
    
    def add_weak_ref(self, obj: Any) -> None:
        """Add weak reference for automatic cleanup."""
        self.weak_refs.add(weakref.ref(obj))
    
    def optimize_data_structures(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize data structures for memory efficiency.
        
        Args:
            data_dict: Dictionary to optimize
            
        Returns:
            Optimized dictionary
        """
        # Convert lists to numpy arrays for better memory efficiency
        optimized = {}
        for key, value in data_dict.items():
            if isinstance(value, list) and len(value) > 100:
                # Convert large lists to numpy arrays
                optimized[key] = np.array(value)
            elif isinstance(value, dict) and len(value) > 50:
                # Recursively optimize nested dictionaries
                optimized[key] = self.optimize_data_structures(value)
            else:
                optimized[key] = value
        
        return optimized


class CacheManager:
    """High-performance cache manager with TTL and memory limits."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if time.time() > entry['expires_at']:
            self.delete(key)
            self.miss_count += 1
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        self.hit_count += 1
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        ttl = ttl or self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
        self.access_times[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        self.delete(lru_key)
    
    def clear_expired(self) -> int:
        """
        Clear expired entries.
        
        Returns:
            Number of entries cleared
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.max_size
        }


class PerformanceOptimizer:
    """
    Main performance optimization system for handling 50+ concurrent symbols.
    
    Implements:
    - Multi-threaded processing with load balancing
    - Memory optimization and garbage collection
    - Intelligent caching with TTL
    - Priority-based task queuing
    - Resource monitoring and throttling
    """
    
    def __init__(self, max_symbols: int = 100, max_workers: int = 4):
        """
        Initialize performance optimizer.
        
        Args:
            max_symbols: Maximum number of symbols to handle
            max_workers: Maximum number of worker threads
        """
        self.max_symbols = max_symbols
        self.max_workers = max_workers
        
        # Components
        self.load_balancer = SymbolLoadBalancer(max_workers)
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager()
        
        # Task processing
        self.task_queue = PriorityQueue()
        self.processing_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Metrics
        self.metrics = OptimizationMetrics()
        self.start_time = time.time()
        
        # Symbol processing state
        self.active_symbols: Set[str] = set()
        self.symbol_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Resource monitoring
        self.cpu_threshold = 80.0  # CPU usage threshold (%)
        self.memory_threshold = 80.0  # Memory usage threshold (%)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Background tasks
        self.cleanup_task = None
        self.monitoring_task = None
        self.running = False
    
    async def start(self) -> None:
        """Start the performance optimizer."""
        if self.running:
            return
        
        self.running = True
        self.logger.info(f"Starting performance optimizer (max_symbols={self.max_symbols}, max_workers={self.max_workers})")
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop(self) -> None:
        """Stop the performance optimizer."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping performance optimizer")
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Shutdown thread pool
        self.processing_pool.shutdown(wait=True)
    
    def submit_processing_task(self, symbol: str, market_data: MarketData, 
                             priority: int = 1, processor_func: Callable = None) -> str:
        """
        Submit processing task for symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Market data to process
            priority: Task priority (lower = higher priority)
            processor_func: Function to process the data
            
        Returns:
            Task ID
        """
        task_id = f"{symbol}_{int(time.time() * 1000000)}"
        
        # Check if we can handle more symbols
        if len(self.active_symbols) >= self.max_symbols:
            self.logger.warning(f"Maximum symbols ({self.max_symbols}) reached, dropping task for {symbol}")
            return task_id
        
        # Add symbol to active set
        self.active_symbols.add(symbol)
        
        # Create processing task
        task = ProcessingTask(
            priority=priority,
            timestamp=datetime.now(),
            symbol=symbol,
            data=market_data,
            task_id=task_id
        )
        
        # Submit to appropriate worker
        worker_id = self.load_balancer.assign_symbol(symbol)
        
        # Submit to processing pool
        future = self.processing_pool.submit(self._process_task, task, processor_func)
        
        # Track task
        self.symbol_stats[symbol]['last_task_id'] = task_id
        self.symbol_stats[symbol]['last_submission_time'] = time.time()
        
        return task_id
    
    def _process_task(self, task: ProcessingTask, processor_func: Callable = None) -> Any:
        """
        Process individual task.
        
        Args:
            task: Processing task
            processor_func: Function to process the data
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Record operation start
            op_id = performance_tracker.start_operation(f"process_{task.symbol}")
            
            # Check cache first
            cache_key = f"{task.symbol}_{task.data.timestamp.isoformat()}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                # Cache hit
                self.metrics.cache_hit_rate = (
                    self.metrics.cache_hit_rate * 0.9 + 1.0 * 0.1  # Moving average
                )
                return cached_result
            
            # Process data
            if processor_func:
                result = processor_func(task.data)
            else:
                result = self._default_processor(task.data)
            
            # Cache result
            self.cache_manager.set(cache_key, result, ttl=60)  # 1 minute TTL
            
            # Update cache miss rate
            self.metrics.cache_hit_rate = (
                self.metrics.cache_hit_rate * 0.9 + 0.0 * 0.1  # Moving average
            )
            
            # Record operation end
            performance_tracker.end_operation(op_id, f"process_{task.symbol}", success=True)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.symbols_processed += 1
            self.metrics.total_processing_time += processing_time
            
            # Update symbol stats
            self.symbol_stats[task.symbol]['last_processing_time'] = processing_time
            self.symbol_stats[task.symbol]['total_processed'] = self.symbol_stats[task.symbol].get('total_processed', 0) + 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            
            # Record operation end with error
            performance_tracker.end_operation(op_id, f"process_{task.symbol}", success=False, error_message=str(e))
            
            raise
        finally:
            # Update processing time
            processing_time = time.time() - start_time
            self.symbol_stats[task.symbol]['last_processing_time'] = processing_time
    
    def _default_processor(self, market_data: MarketData) -> Dict[str, Any]:
        """Default data processor."""
        return {
            'symbol': market_data.symbol,
            'price': market_data.price,
            'volume': market_data.volume,
            'timestamp': market_data.timestamp.isoformat(),
            'processed_at': datetime.now().isoformat()
        }
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.running:
            try:
                # Memory optimization
                if self.memory_optimizer.should_run_gc():
                    gc_stats = self.memory_optimizer.run_gc()
                    self.metrics.gc_collections += 1
                    self.metrics.peak_memory_usage = max(
                        self.metrics.peak_memory_usage,
                        gc_stats['memory_after_mb']
                    )
                
                # Cache cleanup
                expired_count = self.cache_manager.clear_expired()
                if expired_count > 0:
                    self.logger.debug(f"Cleared {expired_count} expired cache entries")
                
                # Symbol cleanup (remove inactive symbols)
                current_time = time.time()
                inactive_symbols = [
                    symbol for symbol, stats in self.symbol_stats.items()
                    if current_time - stats.get('last_submission_time', 0) > 600  # 10 minutes
                ]
                
                for symbol in inactive_symbols:
                    self.active_symbols.discard(symbol)
                    del self.symbol_stats[symbol]
                    self.logger.debug(f"Removed inactive symbol {symbol}")
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.running:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Throttle if resources are high
                if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
                    self.logger.warning(f"High resource usage: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%")
                    
                    # Reduce worker count temporarily
                    if self.max_workers > 1:
                        self.max_workers -= 1
                        self.logger.info(f"Reduced worker count to {self.max_workers}")
                elif cpu_percent < self.cpu_threshold * 0.5 and memory_percent < self.memory_threshold * 0.5:
                    # Restore worker count if resources are low
                    if self.max_workers < 4:
                        self.max_workers += 1
                        self.logger.info(f"Increased worker count to {self.max_workers}")
                
                # Update metrics
                self.metrics.queue_depth = self.task_queue.qsize()
                self.metrics.thread_pool_utilization = len(self.processing_pool._threads) / self.max_workers
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization statistics.
        
        Returns:
            Dictionary with optimization statistics
        """
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'active_symbols': len(self.active_symbols),
            'max_symbols': self.max_symbols,
            'symbol_utilization': len(self.active_symbols) / self.max_symbols,
            'processing_metrics': {
                'symbols_processed': self.metrics.symbols_processed,
                'avg_processing_time': self.metrics.avg_processing_time,
                'total_processing_time': self.metrics.total_processing_time,
                'throughput_symbols_per_second': self.metrics.symbols_processed / uptime if uptime > 0 else 0
            },
            'cache_stats': self.cache_manager.get_stats(),
            'memory_stats': {
                'peak_memory_usage_mb': self.metrics.peak_memory_usage,
                'gc_collections': self.metrics.gc_collections,
                'current_memory_mb': self.memory_optimizer.check_memory_usage()
            },
            'worker_stats': {
                'max_workers': self.max_workers,
                'worker_loads': self.load_balancer.worker_loads,
                'thread_pool_utilization': self.metrics.thread_pool_utilization
            },
            'symbol_stats': dict(self.symbol_stats)
        }
    
    def optimize_for_latency(self) -> None:
        """Optimize configuration for minimum latency."""
        # Increase worker count for parallel processing
        self.max_workers = min(8, psutil.cpu_count())
        
        # Reduce cache TTL for fresher data
        self.cache_manager.default_ttl = 30
        
        # Increase memory threshold for less frequent GC
        self.memory_optimizer.gc_threshold = self.memory_optimizer.max_memory_mb * 0.9
        
        self.logger.info("Optimized for latency")
    
    def optimize_for_throughput(self) -> None:
        """Optimize configuration for maximum throughput."""
        # Increase symbol capacity
        self.max_symbols = 150
        
        # Longer cache TTL for better hit rates
        self.cache_manager.default_ttl = 300
        
        # More aggressive memory management
        self.memory_optimizer.gc_threshold = self.memory_optimizer.max_memory_mb * 0.7
        
        self.logger.info("Optimized for throughput")
    
    def get_symbol_performance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get performance statistics for specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol performance data
        """
        if symbol not in self.symbol_stats:
            return None
        
        stats = self.symbol_stats[symbol]
        return {
            'symbol': symbol,
            'total_processed': stats.get('total_processed', 0),
            'last_processing_time': stats.get('last_processing_time', 0),
            'last_submission_time': stats.get('last_submission_time', 0),
            'assigned_worker': self.load_balancer.get_symbol_worker(symbol),
            'avg_processing_time': stats.get('total_processing_time', 0) / stats.get('total_processed', 1)
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()