"""
ORB Alert Configuration Management

This module provides configuration management for the ORB trading alerts system.
Based on PCA analysis showing 82.31% variance explained by ORB patterns.
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ORBAlertConfig:
    """Configuration class for ORB trading alerts system."""
    
    # Primary ORB Settings
    orb_period_minutes: int = 15          # Opening range period
    breakout_threshold: float = 0.002     # 0.2% above ORB high
    volume_multiplier: float = 1.5        # 1.5x average volume required
    
    # Statistical Confidence (based on PCA analysis)
    pc1_weight: float = 0.8231           # PC1 variance weight
    pc2_weight: float = 0.0854           # PC2 variance weight
    pc3_weight: float = 0.0378           # PC3 variance weight
    min_confidence_score: float = 0.70   # Minimum confidence threshold
    
    # Risk Management
    min_price: float = 0.01              # Minimum stock price
    max_price: float = 50.00             # Maximum stock price
    min_volume: int = 1000000            # Minimum daily volume
    stop_loss_percent: float = 7.5       # Stop loss percentage (7.5%)
    take_profit_percent: float = 4.0     # Take profit percentage (4%)
    
    # Alert Timing
    alert_window_start: str = "09:45"    # Post-ORB period
    alert_window_end: str = "20:00"      # Extended for testing
    
    # Alpaca API Configuration
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    
    # Websocket Configuration
    websocket_timeout: int = 60          # Websocket timeout in seconds
    reconnect_delay: int = 5             # Reconnection delay in seconds
    max_reconnect_attempts: int = 10     # Maximum reconnection attempts
    
    # Symbol Configuration  
    symbols_file: str = ""  # Empty string means use current date file (data/YYYYMMDD.csv)
    
    # Data Collection Configuration
    data_save_interval_minutes: int = 10  # Save historical data every N minutes (configurable)
    market_open_time: str = "09:30"       # Market open time (ET) - 09:30 for NYSE/NASDAQ
    start_collection_at_open: bool = True # Wait until market open to start data collection
                                          # This ensures ORB data is captured from market open
    
    def __post_init__(self):
        """Load API credentials from environment variables."""
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv("ALPACA_BASE_URL", self.base_url)
        
        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment")
    
    def validate(self) -> List[str]:
        """Validate configuration settings and return list of errors."""
        errors = []
        
        if self.orb_period_minutes <= 0:
            errors.append("orb_period_minutes must be positive")
        
        if self.breakout_threshold <= 0:
            errors.append("breakout_threshold must be positive")
        
        if self.volume_multiplier <= 0:
            errors.append("volume_multiplier must be positive")
        
        if self.min_price <= 0:
            errors.append("min_price must be positive")
        
        if self.min_price > 0 and self.max_price <= self.min_price:
            errors.append("max_price must be greater than min_price")
        
        if self.min_volume <= 0:
            errors.append("min_volume must be positive")
        
        if not (0 <= self.min_confidence_score <= 1):
            errors.append("min_confidence_score must be between 0 and 1")
        
        if self.stop_loss_percent <= 0:
            errors.append("stop_loss_percent must be positive")
        
        if self.take_profit_percent <= 0:
            errors.append("take_profit_percent must be positive")
        
        if self.data_save_interval_minutes <= 0:
            errors.append("data_save_interval_minutes must be positive")
        
        return errors


# Global configuration instance
config = ORBAlertConfig()