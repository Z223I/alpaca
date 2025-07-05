"""
Real-time Data Buffer for ORB Alerts

This module provides data buffering and management for real-time market data.
Maintains recent price history for ORB calculations and technical indicators.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import logging
from dataclasses import dataclass

from .alpaca_stream import MarketData


@dataclass
class SymbolBuffer:
    """Buffer for a single symbol's market data."""
    symbol: str
    max_size: int
    data: deque
    
    def __post_init__(self):
        """Initialize data buffer."""
        self.data = deque(maxlen=self.max_size)
    
    def add_data(self, market_data: MarketData) -> None:
        """Add new market data point."""
        self.data.append(market_data)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert buffer to pandas DataFrame."""
        if not self.data:
            return pd.DataFrame()
        
        records = []
        for data_point in self.data:
            records.append({
                'timestamp': data_point.timestamp,
                'symbol': data_point.symbol,
                'high': data_point.high,
                'low': data_point.low,
                'close': data_point.close,
                'volume': data_point.volume,
                'vwap': data_point.vwap,
                'trade_count': data_point.trade_count
            })
        
        return pd.DataFrame(records)
    
    def get_recent_data(self, minutes: int = 30) -> pd.DataFrame:
        """Get data from the last N minutes."""
        if not self.data:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_data = []
        for data_point in self.data:
            if data_point.timestamp >= cutoff_time:
                recent_data.append({
                    'timestamp': data_point.timestamp,
                    'symbol': data_point.symbol,
                    'high': data_point.high,
                    'low': data_point.low,
                    'close': data_point.close,
                    'volume': data_point.volume,
                    'vwap': data_point.vwap,
                    'trade_count': data_point.trade_count
                })
        
        return pd.DataFrame(recent_data)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.data) == 0
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.data)
    
    def clear(self) -> None:
        """Clear all data from buffer."""
        self.data.clear()


class DataBuffer:
    """Main data buffer managing all symbols."""
    
    def __init__(self, max_buffer_size: int = 1000):
        """
        Initialize data buffer.
        
        Args:
            max_buffer_size: Maximum number of data points per symbol
        """
        self.max_buffer_size = max_buffer_size
        self.symbol_buffers: Dict[str, SymbolBuffer] = {}
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.total_messages_received = 0
        self.last_update_time = None
    
    def add_market_data(self, market_data: MarketData) -> None:
        """
        Add market data to appropriate symbol buffer.
        
        Args:
            market_data: Market data to add
        """
        symbol = market_data.symbol
        
        # Create buffer if it doesn't exist
        if symbol not in self.symbol_buffers:
            self.symbol_buffers[symbol] = SymbolBuffer(
                symbol=symbol,
                max_size=self.max_buffer_size,
                data=deque(maxlen=self.max_buffer_size)
            )
        
        # Add data to buffer
        self.symbol_buffers[symbol].add_data(market_data)
        
        # Update statistics
        self.total_messages_received += 1
        self.last_update_time = datetime.now()
        
        self.logger.debug(f"Added data for {symbol}: {market_data.close}")
    
    def get_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get all data for a specific symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            DataFrame with symbol data or None if not found
        """
        if symbol not in self.symbol_buffers:
            return None
        
        return self.symbol_buffers[symbol].get_dataframe()
    
    def get_recent_data(self, symbol: str, minutes: int = 30) -> Optional[pd.DataFrame]:
        """
        Get recent data for a specific symbol.
        
        Args:
            symbol: Symbol to get data for
            minutes: Number of minutes of recent data
            
        Returns:
            DataFrame with recent data or None if not found
        """
        if symbol not in self.symbol_buffers:
            return None
        
        return self.symbol_buffers[symbol].get_recent_data(minutes)
    
    def get_opening_range_data(self, symbol: str, orb_minutes: int = 15) -> Optional[pd.DataFrame]:
        """
        Get opening range data for ORB calculation.
        
        Args:
            symbol: Symbol to get data for
            orb_minutes: Opening range period in minutes
            
        Returns:
            DataFrame with opening range data or None if not found
        """
        if symbol not in self.symbol_buffers:
            return None
        
        df = self.symbol_buffers[symbol].get_dataframe()
        if df.empty:
            return None
        
        # Get today's data starting from 9:30 AM
        today = datetime.now().date()
        market_open = datetime.combine(today, datetime.min.time().replace(hour=9, minute=30))
        orb_end = market_open + timedelta(minutes=orb_minutes)
        
        # Filter for opening range period
        orb_data = df[
            (df['timestamp'] >= market_open) & 
            (df['timestamp'] <= orb_end)
        ]
        
        return orb_data if not orb_data.empty else None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if not found
        """
        if symbol not in self.symbol_buffers:
            return None
        
        buffer = self.symbol_buffers[symbol]
        if buffer.is_empty():
            return None
        
        return buffer.data[-1].close
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get latest market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Latest MarketData or None if not found
        """
        if symbol not in self.symbol_buffers:
            return None
        
        buffer = self.symbol_buffers[symbol]
        if buffer.is_empty():
            return None
        
        return buffer.data[-1]
    
    def get_tracked_symbols(self) -> List[str]:
        """
        Get list of currently tracked symbols.
        
        Returns:
            List of symbol strings
        """
        return list(self.symbol_buffers.keys())
    
    def get_buffer_sizes(self) -> Dict[str, int]:
        """
        Get buffer sizes for all symbols.
        
        Returns:
            Dictionary mapping symbols to buffer sizes
        """
        return {symbol: buffer.size() for symbol, buffer in self.symbol_buffers.items()}
    
    def clear_symbol(self, symbol: str) -> None:
        """
        Clear data for a specific symbol.
        
        Args:
            symbol: Symbol to clear
        """
        if symbol in self.symbol_buffers:
            self.symbol_buffers[symbol].clear()
    
    def clear_all(self) -> None:
        """Clear all data from all buffers."""
        for buffer in self.symbol_buffers.values():
            buffer.clear()
        self.total_messages_received = 0
        self.last_update_time = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            'total_symbols': len(self.symbol_buffers),
            'total_messages_received': self.total_messages_received,
            'last_update_time': self.last_update_time,
            'buffer_sizes': self.get_buffer_sizes(),
            'average_buffer_size': sum(self.get_buffer_sizes().values()) / len(self.symbol_buffers) if self.symbol_buffers else 0
        }
    
    def is_symbol_active(self, symbol: str, minutes: int = 5) -> bool:
        """
        Check if a symbol has received data recently.
        
        Args:
            symbol: Symbol to check
            minutes: Recent activity window in minutes
            
        Returns:
            True if symbol has recent activity
        """
        if symbol not in self.symbol_buffers:
            return False
        
        buffer = self.symbol_buffers[symbol]
        if buffer.is_empty():
            return False
        
        latest_data = buffer.data[-1]
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return latest_data.timestamp >= cutoff_time