"""
Advanced Error Handling and Retry Management

This module implements sophisticated error handling with exponential backoff,
circuit breaker patterns, and automatic recovery mechanisms for production reliability.
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import json
from pathlib import Path

from atoms.config.alert_config import config


class ErrorType(Enum):
    """Classification of error types for different handling strategies."""
    NETWORK_ERROR = "network_error"
    API_RATE_LIMIT = "api_rate_limit"
    AUTHENTICATION_ERROR = "authentication_error"
    DATA_QUALITY_ERROR = "data_quality_error"
    SYSTEM_RESOURCE_ERROR = "system_resource_error"
    UNKNOWN_ERROR = "unknown_error"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, not allowing calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_max: float = 0.1  # 10% jitter
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])
    backoff_strategy: str = "exponential"  # "exponential", "linear", "fixed"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception
    name: str = "default"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: datetime
    error_type: ErrorType
    error_message: str
    operation_name: str
    retry_count: int
    recovery_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


class RetryManager:
    """
    Advanced retry manager with exponential backoff and circuit breaker.
    
    Provides sophisticated error handling for production systems with:
    - Exponential backoff with jitter
    - Circuit breaker pattern
    - Error classification and tracking
    - Automatic recovery mechanisms
    """
    
    def __init__(self, default_config: RetryConfig = None):
        """
        Initialize retry manager.
        
        Args:
            default_config: Default retry configuration
        """
        self.default_config = default_config or RetryConfig()
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.error_history: List[ErrorRecord] = []
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Error classification patterns
        self.error_patterns = {
            ErrorType.NETWORK_ERROR: [
                "connection", "timeout", "network", "socket", "dns", "unreachable"
            ],
            ErrorType.API_RATE_LIMIT: [
                "rate limit", "too many requests", "quota", "throttle", "429"
            ],
            ErrorType.AUTHENTICATION_ERROR: [
                "authentication", "unauthorized", "invalid token", "403", "401"
            ],
            ErrorType.DATA_QUALITY_ERROR: [
                "invalid data", "parse error", "format error", "validation"
            ],
            ErrorType.SYSTEM_RESOURCE_ERROR: [
                "memory", "disk", "cpu", "resource", "capacity"
            ]
        }
    
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify error type for appropriate handling.
        
        Args:
            error: Exception to classify
            
        Returns:
            ErrorType enum
        """
        error_message = str(error).lower()
        
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type
        
        return ErrorType.UNKNOWN_ERROR
    
    def get_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """
        Calculate retry delay based on configuration.
        
        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration
            
        Returns:
            Delay in seconds
        """
        if config.backoff_strategy == "exponential":
            delay = config.base_delay * (config.exponential_base ** attempt)
        elif config.backoff_strategy == "linear":
            delay = config.base_delay * (attempt + 1)
        else:  # fixed
            delay = config.base_delay
        
        # Apply maximum delay
        delay = min(delay, config.max_delay)
        
        # Apply jitter if enabled
        if config.jitter:
            jitter_amount = delay * config.jitter_max
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    async def execute_with_retry(self, 
                                operation: Callable,
                                operation_name: str,
                                config: RetryConfig = None,
                                circuit_breaker_config: CircuitBreakerConfig = None,
                                *args, **kwargs) -> Any:
        """
        Execute operation with retry logic and circuit breaker.
        
        Args:
            operation: Function to execute
            operation_name: Name for logging and stats
            config: Retry configuration (uses default if None)
            circuit_breaker_config: Circuit breaker configuration
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Result of operation
            
        Raises:
            Exception: If all retries exhausted or circuit breaker open
        """
        config = config or self.default_config
        
        # Get or create circuit breaker
        circuit_breaker = None
        if circuit_breaker_config:
            circuit_breaker = self.get_circuit_breaker(circuit_breaker_config)
        
        # Check circuit breaker state
        if circuit_breaker and not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker open for {operation_name}")
        
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Record success
                self._record_success(operation_name, attempt)
                
                # Reset circuit breaker on success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self.classify_error(e)
                
                # Check if this exception should trigger retry
                if not any(isinstance(e, exc_type) for exc_type in config.retry_on_exceptions):
                    # Don't retry this type of exception
                    self._record_error(operation_name, error_type, str(e), attempt)
                    raise e
                
                # Record error
                self._record_error(operation_name, error_type, str(e), attempt)
                
                # Record circuit breaker failure
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Don't retry on last attempt
                if attempt >= config.max_retries:
                    break
                
                # Calculate delay for next attempt
                delay = self.get_retry_delay(attempt, config)
                
                self.logger.warning(
                    f"Retry {attempt + 1}/{config.max_retries} for {operation_name} "
                    f"after {delay:.2f}s delay. Error: {str(e)}"
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries exhausted
        self.logger.error(f"All retries exhausted for {operation_name}")
        raise last_exception
    
    def get_circuit_breaker(self, config: CircuitBreakerConfig) -> 'CircuitBreaker':
        """
        Get or create circuit breaker for operation.
        
        Args:
            config: Circuit breaker configuration
            
        Returns:
            CircuitBreaker instance
        """
        if config.name not in self.circuit_breakers:
            self.circuit_breakers[config.name] = CircuitBreaker(config)
        
        return self.circuit_breakers[config.name]
    
    def _record_success(self, operation_name: str, attempts: int) -> None:
        """Record successful operation."""
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_retry_attempts': 0,
                'avg_attempts': 0.0
            }
        
        stats = self.operation_stats[operation_name]
        stats['total_calls'] += 1
        stats['successful_calls'] += 1
        stats['total_retry_attempts'] += attempts
        stats['avg_attempts'] = stats['total_retry_attempts'] / stats['total_calls']
    
    def _record_error(self, operation_name: str, error_type: ErrorType, 
                     error_message: str, attempt: int) -> None:
        """Record error occurrence."""
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            error_message=error_message,
            operation_name=operation_name,
            retry_count=attempt
        )
        
        self.error_history.append(error_record)
        
        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Update operation stats
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_retry_attempts': 0,
                'avg_attempts': 0.0
            }
        
        stats = self.operation_stats[operation_name]
        if attempt == 0:  # Only count as failed call on first attempt
            stats['total_calls'] += 1
            stats['failed_calls'] += 1
        stats['total_retry_attempts'] += 1
        stats['avg_attempts'] = stats['total_retry_attempts'] / stats['total_calls']
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get error statistics for the specified time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with error statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {'total_errors': 0}
        
        # Count by error type
        error_type_counts = {}
        for error in recent_errors:
            error_type_counts[error.error_type.value] = error_type_counts.get(error.error_type.value, 0) + 1
        
        # Count by operation
        operation_counts = {}
        for error in recent_errors:
            operation_counts[error.operation_name] = operation_counts.get(error.operation_name, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'error_types': error_type_counts,
            'operations': operation_counts,
            'avg_retry_count': sum(e.retry_count for e in recent_errors) / len(recent_errors),
            'time_window_hours': hours
        }
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """
        Get operation statistics.
        
        Returns:
            Dictionary with operation statistics
        """
        return self.operation_stats.copy()
    
    def export_error_report(self, filepath: str) -> None:
        """
        Export error report to JSON file.
        
        Args:
            filepath: Path to output file
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'error_statistics': self.get_error_statistics(),
            'operation_statistics': self.get_operation_statistics(),
            'circuit_breaker_states': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'last_failure_time': cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            },
            'recent_errors': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'error_type': e.error_type.value,
                    'error_message': e.error_message,
                    'operation_name': e.operation_name,
                    'retry_count': e.retry_count
                }
                for e in self.error_history[-100:]  # Last 100 errors
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Exported error report to {filepath}")


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by stopping requests to failing services.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self) -> bool:
        """
        Check if circuit breaker allows execution.
        
        Returns:
            True if execution is allowed
        """
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.config.name} transitioning to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self) -> None:
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.logger.info(f"Circuit breaker {self.config.name} recovered - CLOSED")
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.config.name} failed in HALF_OPEN - OPEN")
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.error(f"Circuit breaker {self.config.name} threshold exceeded - OPEN")


# Decorator for easy retry functionality
def retry_on_failure(max_retries: int = 3, 
                    base_delay: float = 1.0,
                    exponential_base: float = 2.0,
                    max_delay: float = 60.0,
                    jitter: bool = True,
                    operation_name: str = None):
    """
    Decorator to add retry functionality to functions.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        exponential_base: Exponential backoff multiplier
        max_delay: Maximum delay between retries
        jitter: Whether to add jitter to delays
        operation_name: Name for logging (uses function name if None)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                exponential_base=exponential_base,
                max_delay=max_delay,
                jitter=jitter
            )
            
            op_name = operation_name or func.__name__
            retry_manager = RetryManager()
            
            return await retry_manager.execute_with_retry(
                func, op_name, retry_config, None, *args, **kwargs
            )
        
        return wrapper
    return decorator


# Global retry manager instance
retry_manager = RetryManager()