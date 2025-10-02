"""
Momentum Alerts Configuration

This module provides centralized configuration for momentum alerts system,
including time periods and threshold constants for different momentum types.
"""

from dataclasses import dataclass


@dataclass
class MomentumAlertsConfig:
    """
    Configuration for momentum alerts system.

    Contains time period constants and momentum threshold values for
    different momentum calculation types (long, standard, and short).
    """

    # Time period constants for momentum calculations (in minutes)
    momentum_long_period: int = 60      # Future use - long-term momentum
    momentum_period: int = 20           # Standard 20-minute momentum
    momentum_short_period: int = 5      # Short-term 5-minute momentum

    # Momentum threshold constants (per minute percentage values)
    # Based on existing system green_threshold of 0.60
    momentum_long_threshold: float = 0.60    # Future use - long-term threshold
    momentum_threshold: float = 0.60         # Standard momentum threshold
    momentum_short_threshold: float = 0.60   # Short-term momentum threshold

    # Additional configuration options
    min_data_points_required: int = 2        # Minimum data points
    volatility_floor: float = 0.1           # Minimum volatility

    # Volume threshold constants for color coding
    volume_high_threshold: int = 80000       # Green volume threshold
    volume_low_threshold: int = 60000        # Red volume threshold


# Default configuration instance
DEFAULT_MOMENTUM_ALERTS_CONFIG = MomentumAlertsConfig()


def get_momentum_alerts_config() -> MomentumAlertsConfig:
    """
    Get the current momentum alerts configuration.

    Returns:
        MomentumAlertsConfig instance with current settings
    """
    return DEFAULT_MOMENTUM_ALERTS_CONFIG


def get_momentum_time_periods() -> tuple[int, int, int]:
    """
    Get momentum time periods: (long, standard, short).

    Returns:
        Tuple of (long_period, period, short_period)
    """
    config = get_momentum_alerts_config()
    return (config.momentum_long_period, config.momentum_period,
            config.momentum_short_period)


def get_momentum_thresholds_values() -> tuple[float, float, float]:
    """
    Get momentum threshold values in order: (long, standard, short).

    Returns:
        Tuple of (long_threshold, threshold, short_threshold)
    """
    config = get_momentum_alerts_config()
    return (config.momentum_long_threshold, config.momentum_threshold,
            config.momentum_short_threshold)


def get_volume_color_emoji(volume: int) -> str:
    """
    Get volume color emoji based on volume thresholds.

    Args:
        volume: Volume value to color code

    Returns:
        Color emoji string for the volume value
        🟢 for high volume (>= 80,000)
        🟡 for medium volume (60,000 - 79,999)
        🔴 for low volume (< 60,000)
    """
    config = get_momentum_alerts_config()

    if volume >= config.volume_high_threshold:
        return "🟢"  # Green - high volume
    elif volume >= config.volume_low_threshold:
        return "🟡"  # Yellow - medium volume
    else:
        return "🔴"  # Red - low volume
