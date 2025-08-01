"""
Websocket Connection Manager

This module manages websocket connections with automatic reconnection,
error handling, and state management for the ORB alerts system.
"""

import asyncio
import logging
from typing import List, Optional, Callable, Any
from enum import Enum
from datetime import datetime

from .alpaca_stream import AlpacaStreamClient, StreamState, MarketData
from .data_buffer import DataBuffer
from ..config.alert_config import config


class ConnectionManagerState(Enum):
    """Connection manager states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ConnectionManager:
    """Manages websocket connections and data flow."""

    def __init__(self, symbols: List[str]):
        """
        Initialize connection manager.

        Args:
            symbols: List of symbols to monitor
        """
        self.symbols = symbols
        self.state = ConnectionManagerState.STOPPED

        # Components
        self.stream_client = AlpacaStreamClient()
        self.data_buffer = DataBuffer()

        # Event handlers
        self.data_handlers: List[Callable[[MarketData], None]] = []
        self.state_handlers: List[Callable[[ConnectionManagerState], None]] = []

        # Monitoring
        self.start_time = None
        self.last_heartbeat = None
        self.reconnect_count = 0

        # Logging
        self.logger = logging.getLogger(__name__)

        # Setup data handling
        self.stream_client.add_data_handler(self._handle_market_data)

    def _handle_market_data(self, market_data: MarketData) -> None:
        """
        Handle incoming market data.

        Args:
            market_data: Market data from stream
        """
        # Add to buffer
        self.data_buffer.add_market_data(market_data)

        # Update heartbeat
        self.last_heartbeat = datetime.now()

        # Call external handlers
        for handler in self.data_handlers:
            try:
                handler(market_data)
            except Exception as e:
                self.logger.error(f"Error in data handler: {e}")

    def add_data_handler(self, handler: Callable[[MarketData], None]) -> None:
        """
        Add a data handler function.

        Args:
            handler: Function to call with new market data
        """
        self.data_handlers.append(handler)

    def remove_data_handler(self, handler: Callable[[MarketData], None]) -> None:
        """
        Remove a data handler function.

        Args:
            handler: Function to remove
        """
        if handler in self.data_handlers:
            self.data_handlers.remove(handler)

    def add_state_handler(self, handler: Callable[[ConnectionManagerState], None]) -> None:
        """
        Add a state change handler.

        Args:
            handler: Function to call on state changes
        """
        self.state_handlers.append(handler)

    def _set_state(self, new_state: ConnectionManagerState) -> None:
        """
        Set new state and notify handlers.

        Args:
            new_state: New state to set
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state

            self.logger.info(f"State changed: {old_state.value} -> {new_state.value}")

            # Notify handlers
            for handler in self.state_handlers:
                try:
                    handler(new_state)
                except Exception as e:
                    self.logger.error(f"Error in state handler: {e}")

    async def start(self) -> bool:
        """
        Start the connection manager.

        Returns:
            True if started successfully
        """
        if self.state == ConnectionManagerState.RUNNING:
            self.logger.warning("Connection manager already running")
            return True

        self._set_state(ConnectionManagerState.STARTING)

        try:
            # Connect to stream
            if not await self.stream_client.connect():
                self._set_state(ConnectionManagerState.ERROR)
                return False

            # Subscribe to symbols
            if not await self.stream_client.subscribe_bars(self.symbols):
                self._set_state(ConnectionManagerState.ERROR)
                return False

            # Start listening
            self.start_time = datetime.now()
            self.last_heartbeat = datetime.now()
            self._set_state(ConnectionManagerState.RUNNING)

            # Start background tasks
            asyncio.create_task(self._monitor_connection())
            asyncio.create_task(self.stream_client.listen())

            self.logger.info(f"Connection manager started for {len(self.symbols)} symbols")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start connection manager: {e}")
            self._set_state(ConnectionManagerState.ERROR)
            return False

    async def stop(self) -> None:
        """Stop the connection manager."""
        if self.state == ConnectionManagerState.STOPPED:
            return

        self._set_state(ConnectionManagerState.STOPPING)

        try:
            await self.stream_client.disconnect()
            self._set_state(ConnectionManagerState.STOPPED)
            self.logger.info("Connection manager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping connection manager: {e}")
            self._set_state(ConnectionManagerState.ERROR)

    async def _monitor_connection(self) -> None:
        """Monitor connection health and handle reconnections."""
        while self.state == ConnectionManagerState.RUNNING:
            try:
                # Check connection health
                if not self.stream_client.is_connected():
                    self.logger.warning("Stream client disconnected, attempting reconnection")

                    if await self.stream_client.reconnect():
                        self.reconnect_count += 1
                        self.logger.info(f"Reconnected successfully (attempt {self.reconnect_count})")
                    else:
                        self.logger.error("Failed to reconnect, setting error state")
                        self._set_state(ConnectionManagerState.ERROR)
                        break

                # Check heartbeat
                if self.last_heartbeat:
                    heartbeat_age = (datetime.now() - self.last_heartbeat).total_seconds()
                    if heartbeat_age > 300:  # 5 minutes without data
                        self.logger.warning(f"No data received for {heartbeat_age:.1f} seconds")

                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(5)

    def get_data_buffer(self) -> DataBuffer:
        """Get the data buffer instance."""
        return self.data_buffer

    def get_symbol_data(self, symbol: str) -> Optional[Any]:
        """Get all data for a symbol."""
        return self.data_buffer.get_symbol_data(symbol)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        return self.data_buffer.get_current_price(symbol)

    def get_recent_data(self, symbol: str, minutes: int = 30) -> Optional[Any]:
        """Get recent data for a symbol."""
        return self.data_buffer.get_recent_data(symbol, minutes)

    def get_opening_range_data(self, symbol: str, orb_minutes: int = 15) -> Optional[Any]:
        """Get opening range data for a symbol."""
        return self.data_buffer.get_opening_range_data(symbol, orb_minutes)

    def is_running(self) -> bool:
        """Check if connection manager is running."""
        return self.state == ConnectionManagerState.RUNNING

    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return (self.state == ConnectionManagerState.RUNNING and 
                self.stream_client.is_connected() and
                self.last_heartbeat is not None and
                (datetime.now() - self.last_heartbeat).total_seconds() < 300)

    def get_statistics(self) -> dict:
        """Get connection statistics."""
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'state': self.state.value,
            'uptime_seconds': uptime,
            'reconnect_count': self.reconnect_count,
            'symbols_count': len(self.symbols),
            'stream_state': self.stream_client.get_state().value,
            'last_heartbeat': self.last_heartbeat,
            'data_buffer_stats': self.data_buffer.get_statistics()
        }

    def add_symbols(self, new_symbols: List[str]) -> None:
        """
        Add new symbols to monitor.

        Args:
            new_symbols: List of new symbols to add
        """
        for symbol in new_symbols:
            if symbol not in self.symbols:
                self.symbols.append(symbol)

        self.logger.info(f"Added {len(new_symbols)} new symbols")

    def remove_symbols(self, symbols_to_remove: List[str]) -> None:
        """
        Remove symbols from monitoring.

        Args:
            symbols_to_remove: List of symbols to remove
        """
        for symbol in symbols_to_remove:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                self.data_buffer.clear_symbol(symbol)

        self.logger.info(f"Removed {len(symbols_to_remove)} symbols")

    def get_tracked_symbols(self) -> List[str]:
        """Get list of currently tracked symbols."""
        return self.symbols.copy()