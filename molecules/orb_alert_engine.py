"""
ORB Alert Engine - Phase 3 Production Ready

This module orchestrates the complete ORB trading alerts system,
integrating breakout detection, confidence scoring, alert formatting,
and Phase 3 production-ready features including performance monitoring,
error handling, data validation, and optimization.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from atoms.websocket.alpaca_stream import AlpacaStreamClient, MarketData
from atoms.websocket.data_buffer import DataBuffer
from atoms.indicators.orb_calculator import ORBCalculator
from atoms.alerts.breakout_detector import BreakoutDetector, BreakoutSignal
from atoms.alerts.confidence_scorer import ConfidenceScorer, ConfidenceComponents
from atoms.alerts.alert_formatter import AlertFormatter, ORBAlert
from atoms.config.symbol_manager import SymbolManager
from atoms.config.alert_config import config

# Phase 3 Production Components
from atoms.monitoring.performance_tracker import performance_tracker
from atoms.monitoring.dashboard import monitoring_dashboard
from atoms.error_handling.retry_manager import retry_manager, RetryConfig, CircuitBreakerConfig
from atoms.validation.data_validator import data_validator
from atoms.optimization.performance_optimizer import performance_optimizer


@dataclass
class AlertEngineStats:
    """Statistics for the alert engine."""
    total_alerts_generated: int = 0
    high_priority_alerts: int = 0
    medium_priority_alerts: int = 0
    low_priority_alerts: int = 0
    symbols_monitored: int = 0
    uptime_seconds: int = 0
    avg_confidence_score: float = 0.0
    last_alert_time: Optional[datetime] = None


class ORBAlertEngine:
    """Main orchestrator for ORB trading alerts system."""
    
    def __init__(self, symbols_file: str = None, output_dir: str = "alerts"):
        """
        Initialize ORB alert engine.
        
        Args:
            symbols_file: Path to symbols CSV file
            output_dir: Directory for alert output files
        """
        # Initialize components
        self.symbol_manager = SymbolManager(symbols_file or config.symbols_file)
        self.stream_client = AlpacaStreamClient()
        self.data_buffer = DataBuffer()
        self.orb_calculator = ORBCalculator()
        self.breakout_detector = BreakoutDetector(self.orb_calculator)
        self.confidence_scorer = ConfidenceScorer()
        self.alert_formatter = AlertFormatter(output_dir)
        
        # State management
        self.is_running = False
        self.start_time = None
        self.stats = AlertEngineStats()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[ORBAlert], None]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Setup data handler
        self.stream_client.add_data_handler(self._handle_market_data)
        
    def add_alert_callback(self, callback: Callable[[ORBAlert], None]) -> None:
        """
        Add callback function to be called when alerts are generated.
        
        Args:
            callback: Function to call with new ORBAlert
        """
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[ORBAlert], None]) -> None:
        """
        Remove alert callback function.
        
        Args:
            callback: Function to remove
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def start(self) -> None:
        """Start the ORB alert engine with Phase 3 production features."""
        if self.is_running:
            self.logger.warning("Alert engine is already running")
            return
        
        self.logger.info("Starting ORB Alert Engine (Phase 3 Production Ready)")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Start Phase 3 monitoring and optimization systems
            await self._start_phase3_systems()
            
            # Connect to data stream with retry and circuit breaker
            await self._connect_with_retry()
            
            # Subscribe to symbols
            symbols = self.symbol_manager.get_symbols()
            self.stats.symbols_monitored = len(symbols)
            
            # Configure performance optimizer for symbol count
            if len(symbols) > 50:
                performance_optimizer.optimize_for_throughput()
            else:
                performance_optimizer.optimize_for_latency()
            
            success = await self._subscribe_with_retry(symbols)
            if not success:
                raise RuntimeError("Failed to subscribe to market data")
            
            self.logger.info(f"Monitoring {len(symbols)} symbols for ORB alerts with Phase 3 optimizations")
            
            # Start listening for market data
            await self.stream_client.listen()
            
        except Exception as e:
            self.logger.error(f"Error in alert engine: {e}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop the ORB alert engine and Phase 3 systems."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping ORB Alert Engine and Phase 3 systems")
        self.is_running = False
        
        # Stop Phase 3 systems
        await self._stop_phase3_systems()
        
        await self.stream_client.disconnect()
        
        # Update stats
        if self.start_time:
            self.stats.uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
    
    async def _stop_phase3_systems(self) -> None:
        """Stop all Phase 3 production systems."""
        self.logger.info("Stopping Phase 3 production systems")
        
        # Stop performance monitoring
        performance_tracker.stop_health_monitoring()
        
        # Stop monitoring dashboard
        await monitoring_dashboard.stop()
        
        # Stop performance optimizer
        await performance_optimizer.stop()
        
        self.logger.info("Phase 3 systems stopped successfully")
    
    def _handle_market_data(self, market_data: MarketData) -> None:
        """
        Handle incoming market data with Phase 3 optimizations.
        
        Args:
            market_data: MarketData from websocket stream
        """
        try:
            # Use Phase 3 enhanced processing
            self._handle_market_data_with_validation(market_data)
            
        except Exception as e:
            self.logger.error(f"Error handling market data for {market_data.symbol}: {e}")
    
    async def _process_potential_alert(self, market_data: MarketData) -> None:
        """
        Process market data for potential alert generation.
        
        Args:
            market_data: MarketData to process
        """
        symbol = market_data.symbol
        
        try:
            # Get historical data for ORB calculation
            historical_data = self.data_buffer.get_symbol_data(symbol)
            if historical_data.empty:
                return
            
            # Calculate ORB levels if not already cached
            orb_level = self.orb_calculator.get_orb_level(symbol)
            if orb_level is None:
                orb_level = self.orb_calculator.calculate_orb_levels(symbol, historical_data)
                if orb_level is None:
                    return
            
            # Check if we're in alert window
            if not self.breakout_detector.is_within_alert_window():
                return
            
            # Calculate volume ratio
            volume_ratio = self._calculate_volume_ratio(symbol, market_data)
            
            # Detect breakout
            breakout_signal = self.breakout_detector.detect_breakout(
                symbol=symbol,
                current_price=market_data.price,
                volume_ratio=volume_ratio,
                timestamp=market_data.timestamp
            )
            
            if breakout_signal is None:
                return
            
            # Calculate technical indicators
            technical_indicators = self.breakout_detector.calculate_technical_indicators(historical_data)
            
            # Calculate confidence score
            confidence = self.confidence_scorer.calculate_confidence_score(
                breakout_signal, technical_indicators
            )
            
            # Check if alert should be generated
            if not self.confidence_scorer.should_generate_alert(confidence):
                return
            
            # Create and process alert
            alert = self.alert_formatter.create_alert(breakout_signal, confidence)
            await self._process_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error processing potential alert for {symbol}: {e}")
    
    def _calculate_volume_ratio(self, symbol: str, market_data: MarketData) -> float:
        """
        Calculate volume ratio vs average volume.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            Volume ratio (current / average)
        """
        # Get average volume from buffer
        avg_volume = self.data_buffer.get_average_volume(symbol, lookback_minutes=20)
        
        if avg_volume is None or avg_volume == 0:
            return 1.0
        
        return market_data.volume / avg_volume
    
    async def _process_alert(self, alert: ORBAlert) -> None:
        """
        Process generated alert through all output channels.
        
        Args:
            alert: ORBAlert to process
        """
        try:
            # Update statistics
            self.stats.total_alerts_generated += 1
            self.stats.last_alert_time = alert.timestamp
            
            if alert.priority.value == "HIGH":
                self.stats.high_priority_alerts += 1
            elif alert.priority.value == "MEDIUM":
                self.stats.medium_priority_alerts += 1
            else:
                self.stats.low_priority_alerts += 1
            
            # Update average confidence score
            total_alerts = self.stats.total_alerts_generated
            self.stats.avg_confidence_score = (
                (self.stats.avg_confidence_score * (total_alerts - 1) + alert.confidence_score) / total_alerts
            )
            
            # Console output
            console_output = self.alert_formatter.format_console_output(alert)
            print(console_output)
            self.logger.info(f"Generated alert: {alert.symbol} - {alert.priority.value}")
            
            # File output
            json_file = self.alert_formatter.save_alert_to_file(alert, "json")
            self.logger.debug(f"Saved alert to: {json_file}")
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing alert: {e}")
    
    def get_stats(self) -> AlertEngineStats:
        """
        Get current engine statistics.
        
        Returns:
            AlertEngineStats object
        """
        # Update uptime
        if self.start_time:
            self.stats.uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
        
        return self.stats
    
    def get_recent_alerts(self, limit: int = 10) -> List[ORBAlert]:
        """
        Get recent alerts from the formatter.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent ORBAlert objects
        """
        return self.alert_formatter.get_recent_alerts(limit)
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Get daily alert summary.
        
        Returns:
            Dictionary with daily summary data
        """
        return self.alert_formatter.get_daily_summary()
    
    def is_market_hours(self) -> bool:
        """
        Check if current time is within market hours.
        
        Returns:
            True if within market hours
        """
        current_time = datetime.now().time()
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        
        return market_open <= current_time <= market_close
    
    def reload_symbols(self) -> None:
        """Reload symbols from configuration file."""
        self.symbol_manager.reload_symbols()
        self.stats.symbols_monitored = len(self.symbol_manager.get_symbols())
        self.logger.info(f"Reloaded {self.stats.symbols_monitored} symbols")
    
    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self.orb_calculator.clear_cache()
        self.breakout_detector.clear_recent_signals()
        self.confidence_scorer.clear_history()
        self.data_buffer.clear_all_data()
        self.logger.info("Cleared all internal caches")
    
    def get_monitored_symbols(self) -> List[str]:
        """
        Get list of currently monitored symbols.
        
        Returns:
            List of symbol strings
        """
        return self.symbol_manager.get_symbols()
    
    # Phase 3 Production Methods
    
    async def _start_phase3_systems(self) -> None:
        """Start all Phase 3 production systems."""
        self.logger.info("Starting Phase 3 production systems")
        
        # Start performance monitoring
        await performance_tracker.start_health_monitoring()
        
        # Start monitoring dashboard
        await monitoring_dashboard.start()
        
        # Start performance optimizer
        await performance_optimizer.start()
        
        self.logger.info("Phase 3 systems started successfully")
    
    async def _connect_with_retry(self) -> None:
        """Connect to data stream with retry logic."""
        retry_config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            name="stream_connection"
        )
        
        async def connect_operation():
            return await self.stream_client.connect()
        
        success = await retry_manager.execute_with_retry(
            connect_operation,
            "stream_connection",
            retry_config,
            circuit_breaker_config
        )
        
        if not success:
            raise RuntimeError("Failed to connect to data stream after retries")
    
    async def _subscribe_with_retry(self, symbols: List[str]) -> bool:
        """Subscribe to symbols with retry logic."""
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0
        )
        
        async def subscribe_operation():
            return await self.stream_client.subscribe_bars(symbols)
        
        return await retry_manager.execute_with_retry(
            subscribe_operation,
            "symbol_subscription",
            retry_config
        )
    
    def _handle_market_data_with_validation(self, market_data: MarketData) -> None:
        """Handle market data with Phase 3 validation and optimization."""
        try:
            # Phase 3: Data validation
            validation_issues = data_validator.validate_market_data(market_data)
            
            # Skip processing if data quality is unacceptable
            if not data_validator.is_data_quality_acceptable(market_data.symbol):
                self.logger.warning(f"Skipping {market_data.symbol} due to poor data quality")
                return
            
            # Phase 3: Performance optimization - submit to processing pool
            def process_data(data):
                return self._process_market_data_optimized(data)
            
            performance_optimizer.submit_processing_task(
                symbol=market_data.symbol,
                market_data=market_data,
                priority=1,
                processor_func=process_data
            )
            
        except Exception as e:
            self.logger.error(f"Error in Phase 3 market data handling for {market_data.symbol}: {e}")
    
    def _process_market_data_optimized(self, market_data: MarketData) -> None:
        """Process market data with Phase 3 optimizations."""
        # Track operation timing
        op_id = performance_tracker.start_operation(f"process_market_data_{market_data.symbol}")
        
        try:
            # Add to data buffer
            self.data_buffer.add_market_data(market_data)
            
            # Process potential alert with async task
            asyncio.create_task(self._process_potential_alert_optimized(market_data))
            
            # Record successful processing
            performance_tracker.end_operation(op_id, success=True)
            
        except Exception as e:
            # Record failed processing
            performance_tracker.end_operation(op_id, success=False, error_message=str(e))
            raise
    
    async def _process_potential_alert_optimized(self, market_data: MarketData) -> None:
        """Process potential alert with Phase 3 optimizations and monitoring."""
        symbol = market_data.symbol
        
        # Track alert processing time
        alert_start_time = time.time()
        op_id = performance_tracker.start_operation(f"alert_processing_{symbol}")
        
        try:
            # Get historical data for ORB calculation
            historical_data = self.data_buffer.get_symbol_data(symbol)
            if historical_data.empty:
                return
            
            # Calculate ORB levels if not already cached
            orb_level = self.orb_calculator.get_orb_level(symbol)
            if orb_level is None:
                orb_level = self.orb_calculator.calculate_orb_levels(symbol, historical_data)
                if orb_level is None:
                    return
            
            # Check if we're in alert window
            if not self.breakout_detector.is_within_alert_window():
                return
            
            # Calculate volume ratio
            volume_ratio = self._calculate_volume_ratio(symbol, market_data)
            
            # Detect breakout
            breakout_signal = self.breakout_detector.detect_breakout(
                symbol=symbol,
                current_price=market_data.price,
                volume_ratio=volume_ratio,
                timestamp=market_data.timestamp
            )
            
            if breakout_signal is None:
                return
            
            # Calculate technical indicators
            technical_indicators = self.breakout_detector.calculate_technical_indicators(historical_data)
            
            # Calculate confidence score
            confidence = self.confidence_scorer.calculate_confidence_score(
                breakout_signal, technical_indicators
            )
            
            # Check if alert should be generated
            if not self.confidence_scorer.should_generate_alert(confidence):
                return
            
            # Create and process alert
            alert = self.alert_formatter.create_alert(breakout_signal, confidence)
            
            # Calculate alert generation time
            alert_generation_time = (time.time() - alert_start_time) * 1000  # Convert to ms
            
            # Record alert metrics
            performance_tracker.record_alert_generated(
                alert.priority.value,
                alert.confidence_score,
                alert_generation_time,
                symbol
            )
            
            await self._process_alert_with_monitoring(alert, alert_generation_time)
            
            # Record successful operation
            performance_tracker.end_operation(op_id, success=True)
            
        except Exception as e:
            self.logger.error(f"Error processing potential alert for {symbol}: {e}")
            performance_tracker.end_operation(op_id, success=False, error_message=str(e))
    
    async def _process_alert_with_monitoring(self, alert: ORBAlert, generation_time_ms: float) -> None:
        """Process alert with Phase 3 monitoring and logging."""
        try:
            # Update statistics
            self.stats.total_alerts_generated += 1
            self.stats.last_alert_time = alert.timestamp
            
            if alert.priority.value == "HIGH":
                self.stats.high_priority_alerts += 1
            elif alert.priority.value == "MEDIUM":
                self.stats.medium_priority_alerts += 1
            else:
                self.stats.low_priority_alerts += 1
            
            # Update average confidence score
            total_alerts = self.stats.total_alerts_generated
            self.stats.avg_confidence_score = (
                (self.stats.avg_confidence_score * (total_alerts - 1) + alert.confidence_score) / total_alerts
            )
            
            # Console output
            console_output = self.alert_formatter.format_console_output(alert)
            print(console_output)
            
            # Log to dashboard
            monitoring_dashboard.dashboard_logger.log_alert_generated(
                alert.symbol, alert.priority.value, alert.confidence_score
            )
            
            self.logger.info(f"Generated alert: {alert.symbol} - {alert.priority.value} (gen_time: {generation_time_ms:.1f}ms)")
            
            # File output
            json_file = self.alert_formatter.save_alert_to_file(alert, "json")
            self.logger.debug(f"Saved alert to: {json_file}")
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing alert with monitoring: {e}")
    
    def get_phase3_status(self) -> Dict[str, Any]:
        """Get comprehensive Phase 3 status report."""
        return {
            'phase3_enabled': True,
            'performance_tracker_running': performance_tracker.health_monitor_running,
            'dashboard_running': monitoring_dashboard.running,
            'optimizer_running': performance_optimizer.running,
            'validation_enabled': True,
            'retry_manager_stats': retry_manager.get_operation_statistics(),
            'error_stats': retry_manager.get_error_statistics(),
            'validation_stats': data_validator.get_validation_summary(),
            'optimization_stats': performance_optimizer.get_optimization_stats(),
            'current_metrics': monitoring_dashboard.get_current_metrics()
        }