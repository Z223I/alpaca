"""
Alpaca Websocket Streaming Client

This module provides real-time market data streaming using Alpaca's websocket API.
Handles connection management, data buffering, and reconnection logic.
"""

import json
import asyncio
import websockets
import logging
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from ..config.alert_config import config


class StreamState(Enum):
    """Websocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    ERROR = "error"


@dataclass
class MarketData:
    """Market data point from websocket stream."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    high: float
    low: float
    close: float
    trade_count: int
    vwap: float


class AlpacaStreamClient:
    """Alpaca websocket streaming client for real-time market data."""
    
    def __init__(self, 
                 api_key: str = None,
                 secret_key: str = None,
                 base_url: str = None):
        """
        Initialize Alpaca stream client.
        
        Args:
            api_key: Alpaca API key (defaults to config)
            secret_key: Alpaca secret key (defaults to config)
            base_url: Alpaca base URL (defaults to config)
        """
        self.api_key = api_key or config.api_key
        self.secret_key = secret_key or config.secret_key
        self.base_url = base_url or config.base_url
        
        # Connection state
        self.state = StreamState.DISCONNECTED
        self.websocket = None
        self.reconnect_count = 0
        
        # Subscriptions
        self.subscribed_symbols: List[str] = []
        self.data_handlers: List[Callable[[MarketData], None]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Connection parameters
        self.websocket_url = self._get_websocket_url()
        self.timeout = config.websocket_timeout
        self.reconnect_delay = config.reconnect_delay
        self.max_reconnect_attempts = config.max_reconnect_attempts
    
    def _get_websocket_url(self) -> str:
        """Get websocket URL based on base URL."""
        if "paper" in self.base_url:
            return "wss://stream.data.alpaca.markets/v2/iex"
        else:
            return "wss://stream.data.alpaca.markets/v2/sip"
    
    async def connect(self) -> bool:
        """
        Connect to Alpaca websocket stream.
        
        Returns:
            True if connected successfully
        """
        try:
            self.state = StreamState.CONNECTING
            self.logger.info(f"Connecting to Alpaca websocket: {self.websocket_url}")
            
            self.websocket = await websockets.connect(
                self.websocket_url,
                timeout=self.timeout
            )
            
            self.state = StreamState.CONNECTED
            self.logger.info("Connected to Alpaca websocket")
            
            # Authenticate
            await self._authenticate()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to websocket: {e}")
            self.state = StreamState.ERROR
            return False
    
    async def _authenticate(self) -> bool:
        """
        Authenticate with Alpaca websocket.
        
        Returns:
            True if authenticated successfully
        """
        try:
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for authentication response
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            # Handle both single message and list of messages
            if isinstance(auth_response, list):
                auth_response = auth_response[0] if auth_response else {}
            
            if auth_response.get("T") == "success":
                self.state = StreamState.AUTHENTICATED
                self.logger.info("Authenticated with Alpaca websocket")
                return True
            else:
                self.logger.error(f"Authentication failed: {auth_response}")
                self.state = StreamState.ERROR
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            self.state = StreamState.ERROR
            return False
    
    async def subscribe_bars(self, symbols: List[str]) -> bool:
        """
        Subscribe to minute bars for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            True if subscribed successfully
        """
        if self.state != StreamState.AUTHENTICATED:
            self.logger.error("Not authenticated, cannot subscribe")
            return False
        
        try:
            subscribe_message = {
                "action": "subscribe",
                "bars": symbols
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Wait for subscription response
            response = await self.websocket.recv()
            sub_response = json.loads(response)
            
            # Handle both single message and list of messages
            if isinstance(sub_response, list):
                sub_response = sub_response[0] if sub_response else {}
            
            if sub_response.get("T") in ["subscription", "success"]:
                self.subscribed_symbols = symbols
                self.state = StreamState.SUBSCRIBED
                self.logger.info(f"Subscribed to bars for {len(symbols)} symbols")
                return True
            else:
                self.logger.error(f"Subscription failed: {sub_response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Subscription error: {e}")
            return False
    
    async def listen(self) -> None:
        """
        Listen for incoming market data and process messages.
        """
        if self.state != StreamState.SUBSCRIBED:
            self.logger.error("Not subscribed, cannot listen")
            return
        
        try:
            self.logger.info("Starting to listen for market data")
            
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON message: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Websocket connection closed")
            self.state = StreamState.DISCONNECTED
        except Exception as e:
            self.logger.error(f"Error in listen loop: {e}")
            self.state = StreamState.ERROR
    
    async def _process_message(self, data: Dict[str, Any]) -> None:
        """
        Process incoming websocket message.
        
        Args:
            data: Parsed JSON message
        """
        # Handle different message types
        if isinstance(data, list):
            for item in data:
                await self._process_single_message(item)
        else:
            await self._process_single_message(data)
    
    async def _process_single_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single message from the websocket.
        
        Args:
            message: Single message dictionary
        """
        msg_type = message.get("T")
        
        if msg_type == "b":  # Bar data
            await self._process_bar_data(message)
        elif msg_type == "error":
            self.logger.error(f"Received error message: {message}")
        elif msg_type == "subscription":
            self.logger.info(f"Subscription update: {message}")
    
    async def _process_bar_data(self, bar_data: Dict[str, Any]) -> None:
        """
        Process bar data message.
        
        Args:
            bar_data: Bar data from websocket
        """
        try:
            market_data = MarketData(
                symbol=bar_data.get("S", ""),
                timestamp=datetime.fromisoformat(bar_data.get("t", "").replace("Z", "+00:00")),
                price=float(bar_data.get("c", 0)),
                volume=int(bar_data.get("v", 0)),
                high=float(bar_data.get("h", 0)),
                low=float(bar_data.get("l", 0)),
                close=float(bar_data.get("c", 0)),
                trade_count=int(bar_data.get("n", 0)),
                vwap=float(bar_data.get("vw", 0))
            )
            
            # Call all registered data handlers
            for handler in self.data_handlers:
                try:
                    handler(market_data)
                except Exception as e:
                    self.logger.error(f"Error in data handler: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing bar data: {e}")
    
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
    
    async def disconnect(self) -> None:
        """Disconnect from websocket."""
        if self.websocket:
            await self.websocket.close()
            self.state = StreamState.DISCONNECTED
            self.logger.info("Disconnected from Alpaca websocket")
    
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to websocket.
        
        Returns:
            True if reconnected successfully
        """
        if self.reconnect_count >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return False
        
        self.reconnect_count += 1
        self.logger.info(f"Reconnection attempt {self.reconnect_count}")
        
        await asyncio.sleep(self.reconnect_delay)
        
        success = await self.connect()
        if success and self.subscribed_symbols:
            success = await self.subscribe_bars(self.subscribed_symbols)
        
        if success:
            self.reconnect_count = 0
            
        return success
    
    def is_connected(self) -> bool:
        """Check if websocket is connected."""
        return self.state in [StreamState.CONNECTED, StreamState.AUTHENTICATED, StreamState.SUBSCRIBED]
    
    def get_state(self) -> StreamState:
        """Get current connection state."""
        return self.state