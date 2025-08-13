"""
Alert Configuration - Centralized momentum thresholds and alert settings

This module provides centralized configuration for all alert momentum thresholds
to ensure consistency across alert generation and notification systems.
"""

from dataclasses import dataclass


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


# Default configuration instance
DEFAULT_PRICE_MOMENTUM_CONFIG = PriceMomentumConfig()


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