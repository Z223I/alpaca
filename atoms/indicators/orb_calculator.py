"""
Opening Range Breakout (ORB) Calculator

This module calculates ORB levels based on the first 15 minutes of trading.
Implements core ORB logic based on PCA analysis showing 82.31% variance 
explained by ORB patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time
import pandas as pd


@dataclass
class ORBLevel:
    """Represents ORB high and low levels for a symbol."""
    symbol: str
    orb_high: float
    orb_low: float
    orb_range: float
    calculation_time: datetime
    sample_count: int
    
    def __post_init__(self):
        """Calculate derived values."""
        self.orb_range = self.orb_high - self.orb_low
        self.orb_midpoint = (self.orb_high + self.orb_low) / 2
    
    def get_breakout_threshold(self, threshold_pct: float = 0.002) -> float:
        """
        Calculate breakout threshold above ORB high.
        
        Args:
            threshold_pct: Percentage threshold (default 0.2%)
            
        Returns:
            Price level for breakout confirmation
        """
        return self.orb_high * (1 + threshold_pct)
    
    def get_breakdown_threshold(self, threshold_pct: float = 0.002) -> float:
        """
        Calculate breakdown threshold below ORB low.
        
        Args:
            threshold_pct: Percentage threshold (default 0.2%)
            
        Returns:
            Price level for breakdown confirmation
        """
        return self.orb_low * (1 - threshold_pct)


class ORBCalculator:
    """Calculator for Opening Range Breakout levels."""
    
    def __init__(self, orb_period_minutes: int = 15):
        """
        Initialize ORB calculator.
        
        Args:
            orb_period_minutes: Opening range period in minutes (default 15)
        """
        self.orb_period_minutes = orb_period_minutes
        self.market_open = time(9, 30)  # 9:30 AM ET
        self._orb_levels: Dict[str, ORBLevel] = {}
    
    def calculate_orb_levels(self, 
                            symbol: str, 
                            price_data: pd.DataFrame) -> Optional[ORBLevel]:
        """
        Calculate ORB levels for a symbol.
        
        Args:
            symbol: Stock symbol
            price_data: DataFrame with columns ['timestamp', 'high', 'low', 'close', 'volume']
            
        Returns:
            ORBLevel object or None if insufficient data
        """
        debug = False  # Enable debug for ORB calculation
        
        if debug:
            print(f"\n=== ORB Calculator Debug for {symbol} ===")
            print(f"Input data shape: {price_data.shape}")
            print(f"Input data columns: {list(price_data.columns)}")
            if not price_data.empty:
                print(f"Data range: {price_data['timestamp'].min()} to {price_data['timestamp'].max()}")
        
        if price_data.empty:
            if debug:
                print("DEBUG: No data available")
            return None
        
        # Ensure timestamp column is datetime
        if 'timestamp' not in price_data.columns:
            if debug:
                print("DEBUG: No timestamp column found")
            return None
        
        price_data = price_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(price_data['timestamp']):
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        # Filter for opening range period
        orb_data = self._filter_opening_range(price_data)
        
        if debug:
            print(f"ORB data shape after filtering: {orb_data.shape}")
            if not orb_data.empty:
                print(f"ORB data range: {orb_data['timestamp'].min()} to {orb_data['timestamp'].max()}")
                print(f"ORB data high values: {orb_data['high'].tolist()}")
                print(f"ORB data low values: {orb_data['low'].tolist()}")
        
        if orb_data.empty or len(orb_data) < 5:  # Need minimum data points
            if debug:
                print(f"DEBUG: Insufficient ORB data - got {len(orb_data)} points, need at least 5")
            return None
        
        # Calculate ORB levels
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        
        if debug:
            print(f"Calculated ORB High: {orb_high}")
            print(f"Calculated ORB Low: {orb_low}")
            print(f"ORB Range: {orb_high - orb_low}")
            print("=== End ORB Calculator Debug ===\n")
        
        orb_level = ORBLevel(
            symbol=symbol,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_range=orb_high - orb_low,
            calculation_time=datetime.now(),
            sample_count=len(orb_data)
        )
        
        # Cache the ORB level
        self._orb_levels[symbol] = orb_level
        
        return orb_level
    
    def _filter_opening_range(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter price data for opening range period.
        
        Args:
            price_data: DataFrame with timestamp column
            
        Returns:
            Filtered DataFrame for opening range period
        """
        # Convert timestamps to Eastern Time for filtering
        import pytz
        price_data = price_data.copy()
        
        # Handle Eastern Time timestamps (now timezone-naive Eastern Time from websocket)
        if price_data['timestamp'].dt.tz is None:
            # Timestamps are already timezone-naive Eastern Time (from websocket), use directly
            price_data['timestamp_et'] = price_data['timestamp']
        else:
            # If timezone-aware, convert to Eastern Time
            et_tz = pytz.timezone('US/Eastern')
            price_data['timestamp_et'] = price_data['timestamp'].dt.tz_convert(et_tz).dt.tz_localize(None)
        
        price_data['time_et'] = price_data['timestamp_et'].dt.time
        
        # Calculate end time for opening range
        market_open_minutes = self.market_open.hour * 60 + self.market_open.minute
        orb_end_minutes = market_open_minutes + self.orb_period_minutes
        orb_end_time = time(orb_end_minutes // 60, orb_end_minutes % 60)
        
        # Filter for opening range period using Eastern Time
        orb_mask = (price_data['time_et'] >= self.market_open) & (price_data['time_et'] <= orb_end_time)
        
        return price_data[orb_mask]
    
    def get_orb_level(self, symbol: str) -> Optional[ORBLevel]:
        """
        Get cached ORB level for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ORBLevel object or None if not calculated
        """
        return self._orb_levels.get(symbol)
    
    def is_breakout(self, 
                   symbol: str, 
                   current_price: float, 
                   threshold_pct: float = 0.002) -> bool:
        """
        Check if current price represents a breakout above ORB high.
        
        Args:
            symbol: Stock symbol
            current_price: Current price to check
            threshold_pct: Percentage threshold for breakout (default 0.2%)
            
        Returns:
            True if price is above ORB high + threshold
        """
        orb_level = self.get_orb_level(symbol)
        if not orb_level:
            return False
        
        breakout_threshold = orb_level.get_breakout_threshold(threshold_pct)
        return current_price >= breakout_threshold
    
    def is_breakdown(self, 
                    symbol: str, 
                    current_price: float, 
                    threshold_pct: float = 0.002) -> bool:
        """
        Check if current price represents a breakdown below ORB low.
        
        Args:
            symbol: Stock symbol
            current_price: Current price to check
            threshold_pct: Percentage threshold for breakdown (default 0.2%)
            
        Returns:
            True if price is below ORB low - threshold
        """
        orb_level = self.get_orb_level(symbol)
        if not orb_level:
            return False
        
        breakdown_threshold = orb_level.get_breakdown_threshold(threshold_pct)
        return current_price <= breakdown_threshold
    
    def get_price_position(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Get price position relative to ORB range (0=low, 1=high).
        
        Args:
            symbol: Stock symbol
            current_price: Current price
            
        Returns:
            Position ratio (0-1) or None if ORB not calculated
        """
        orb_level = self.get_orb_level(symbol)
        if not orb_level or orb_level.orb_range == 0:
            return None
        
        return (current_price - orb_level.orb_low) / orb_level.orb_range
    
    def clear_cache(self) -> None:
        """Clear all cached ORB levels."""
        self._orb_levels.clear()
    
    def get_cached_symbols(self) -> List[str]:
        """Get list of symbols with cached ORB levels."""
        return list(self._orb_levels.keys())