"""
ORB Alert Engine

This module orchestrates the complete ORB trading alerts system,
integrating breakout detection, confidence scoring, and alert formatting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from ..atoms.websocket.alpaca_stream import AlpacaStreamClient, MarketData
from ..atoms.websocket.data_buffer import DataBuffer
from ..atoms.indicators.orb_calculator import ORBCalculator
from ..atoms.alerts.breakout_detector import BreakoutDetector, BreakoutSignal
from ..atoms.alerts.confidence_scorer import ConfidenceScorer, ConfidenceComponents
from ..atoms.alerts.alert_formatter import AlertFormatter, ORBAlert
from ..atoms.config.symbol_manager import SymbolManager
from ..atoms.config.alert_config import config


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
        """Start the ORB alert engine."""
        if self.is_running:
            self.logger.warning("Alert engine is already running")
            return
        
        self.logger.info("Starting ORB Alert Engine")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Connect to data stream
            success = await self.stream_client.connect()
            if not success:
                raise RuntimeError("Failed to connect to data stream")
            
            # Subscribe to symbols
            symbols = self.symbol_manager.get_symbols()
            self.stats.symbols_monitored = len(symbols)
            
            success = await self.stream_client.subscribe_bars(symbols)
            if not success:
                raise RuntimeError("Failed to subscribe to market data")
            
            self.logger.info(f"Monitoring {len(symbols)} symbols for ORB alerts")
            
            # Start listening for market data
            await self.stream_client.listen()
            
        except Exception as e:
            self.logger.error(f"Error in alert engine: {e}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop the ORB alert engine."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping ORB Alert Engine")
        self.is_running = False
        
        await self.stream_client.disconnect()
        
        # Update stats
        if self.start_time:
            self.stats.uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
    
    def _handle_market_data(self, market_data: MarketData) -> None:
        """
        Handle incoming market data and process for alerts.
        
        Args:
            market_data: MarketData from websocket stream
        """
        try:
            # Add to data buffer
            self.data_buffer.add_data(market_data)
            
            # Process potential alert
            asyncio.create_task(self._process_potential_alert(market_data))
            
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