"""
Alert Configuration - Centralized momentum thresholds and alert settings

This module provides centralized configuration for all alert momentum thresholds
to ensure consistency across alert generation and notification systems.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HistoricalRootDir:
    """
    Historical data root directory configuration.
    
    This allows configuring where the historical_data directory structure
    should be rooted, making it easier to test or relocate data storage.
    """
    
    # Root directory where historical_data folder will be located
    root_path: str = "."
    
    def get_historical_data_path(self, date: str) -> Path:
        """
        Get the full path to historical data for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Path to historical_data/{date}
        """
        return Path(self.root_path) / "historical_data" / date
    
    def get_alerts_dir(self, date: str, alert_type: str = "bullish") -> Path:
        """Get alerts directory for a specific date and type."""
        return self.get_historical_data_path(date) / "alerts" / alert_type
    
    def get_super_alerts_dir(self, date: str, alert_type: str = "bullish") -> Path:
        """Get super alerts directory for a specific date and type."""
        return self.get_historical_data_path(date) / "super_alerts" / alert_type
    
    def get_superduper_alerts_dir(self, date: str, alert_type: str = "bullish") -> Path:
        """Get superduper alerts directory for a specific date and type."""
        return self.get_historical_data_path(date) / "superduper_alerts" / alert_type
    
    def get_superduper_alerts_sent_dir(self, date: str, alert_type: str = "bullish", momentum_color: str = "green") -> Path:
        """Get superduper alerts sent directory for a specific date, type, and momentum color."""
        return self.get_historical_data_path(date) / "superduper_alerts_sent" / alert_type / momentum_color
    
    def get_trades_dir(self, date: str) -> Path:
        """Get trades directory for a specific date."""
        return self.get_historical_data_path(date)


@dataclass
class MomentumThresholds:
    """
    Momentum threshold configuration for alert color coding and filtering.
    
    Momentum is measured as percentage change per minute (e.g., 0.65 = 0.65%/min).
    """
    
    # Red momentum threshold (below this = red 🔴)
    red_threshold: float = 0.3
    
    # Green momentum threshold (at or above this = green 🟢)  
    green_threshold: float = 0.65
    
    # Yellow momentum is implicitly between red_threshold and green_threshold (🟡)
    
    def get_momentum_color_emoji(self, momentum: float) -> str:
        """
        Get color emoji for momentum value.
        
        Args:
            momentum: Momentum value to color code
            
        Returns:
            Color emoji string for the momentum value
        """
        if momentum < self.red_threshold:
            return "🔴"  # Red
        elif momentum < self.green_threshold:
            return "🟡"  # Yellow  
        else:
            return "🟢"  # Green
    
    def get_momentum_color_name(self, momentum: float) -> str:
        """
        Get color name for momentum value.
        
        Args:
            momentum: Momentum value to color code
            
        Returns:
            Color name string for the momentum value
        """
        if momentum < self.red_threshold:
            return "red"
        elif momentum < self.green_threshold:
            return "yellow"
        else:
            return "green"
    
    def is_red_momentum(self, momentum: float) -> bool:
        """Check if momentum is in red range."""
        return momentum < self.red_threshold
    
    def is_yellow_momentum(self, momentum: float) -> bool:
        """Check if momentum is in yellow range."""
        return self.red_threshold <= momentum < self.green_threshold
    
    def is_green_momentum(self, momentum: float) -> bool:
        """Check if momentum is in green range."""
        return momentum >= self.green_threshold
    
    def get_urgency_level(self, momentum: float) -> str:
        """
        Get urgency level for Telegram notifications.
        
        Args:
            momentum: Momentum value to evaluate
            
        Returns:
            'filtered' - Red and yellow momentum, don't send to Telegram
            'urgent' - Green momentum, urgent Telegram notification
        """
        if momentum < self.green_threshold:
            return 'filtered'
        else:
            return 'urgent'


@dataclass
class PriceMomentumConfig:
    """
    Price momentum configuration containing momentum thresholds and timeframe settings.
    """
    
    # Momentum thresholds for color coding and filtering
    momentum: MomentumThresholds = MomentumThresholds()
    
    # Trend analysis timeframe in minutes
    trend_analysis_timeframe_minutes: int = 20
    
    # Other momentum configuration can be added here in the future
    # e.g., penetration thresholds, strength thresholds, etc.


# Default configuration instances
DEFAULT_HISTORICAL_ROOT_DIR = HistoricalRootDir()
DEFAULT_PRICE_MOMENTUM_CONFIG = PriceMomentumConfig()


def get_historical_root_dir() -> HistoricalRootDir:
    """
    Get the current historical root directory configuration.
    
    Returns:
        HistoricalRootDir instance with current settings
    """
    return DEFAULT_HISTORICAL_ROOT_DIR


def get_price_momentum_config() -> PriceMomentumConfig:
    """
    Get the current price momentum configuration.
    
    Returns:
        PriceMomentumConfig instance with current settings
    """
    return DEFAULT_PRICE_MOMENTUM_CONFIG


def get_momentum_thresholds() -> MomentumThresholds:
    """
    Get the current momentum thresholds configuration.
    
    Returns:
        MomentumThresholds instance with current settings
    """
    return DEFAULT_PRICE_MOMENTUM_CONFIG.momentum