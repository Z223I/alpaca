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
    momentum_period: int = 10           # Standard 10-minute momentum
    momentum_short_period: int = 5      # Short-term 5-minute momentum
    squeeze_duration: int = 10          # Squeeze detection period

    # Momentum threshold constants (per minute percentage values)
    # Based on existing system green_threshold of 0.60
    momentum_long_threshold: float = 0.60    # Future use - long-term threshold
    momentum_threshold: float = 0.60         # Standard momentum threshold
    momentum_short_threshold: float = 0.60   # Short-term momentum threshold

    # Momentum minimum threshold constants (per minute percentage values)
    # Minimum values for momentum detection/filtering
    momentum_long_minimum: float = 0.50      # Future use - long-term minimum
    momentum_minimum: float = 0.50           # Standard momentum minimum
    momentum_short_minimum: float = 0.50     # Short-term momentum minimum

    # Additional configuration options
    min_data_points_required: int = 2        # Minimum data points
    volatility_floor: float = 0.1           # Minimum volatility

    # Volume threshold constants for color coding
    volume_high_threshold: int = 250000       # Green volume threshold
    volume_low_threshold: int = 80000        # Red volume threshold


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


def get_momentum_minimums_values() -> tuple[float, float, float]:
    """
    Get momentum minimum threshold values in order: (long, standard, short).

    Returns:
        Tuple of (long_minimum, minimum, short_minimum)
    """
    config = get_momentum_alerts_config()
    return (config.momentum_long_minimum, config.momentum_minimum,
            config.momentum_short_minimum)


def get_momentum_color_emoji(momentum: float, threshold: float, minimum: float) -> str:
    """
    Get momentum color emoji based on threshold and minimum values.

    Args:
        momentum: Momentum value to color code
        threshold: High threshold for green light
        minimum: Minimum threshold for yellow light

    Returns:
        Color emoji string for the momentum value
        🟢 for high momentum (>= threshold)
        🟡 for medium momentum (>= minimum but < threshold)
        🔴 for low momentum (< minimum)
    """
    if momentum >= threshold:
        return "🟢"  # Green - high momentum
    elif momentum >= minimum:
        return "🟡"  # Yellow - medium momentum
    else:
        return "🔴"  # Red - low momentum


def get_momentum_standard_color_emoji(momentum: float) -> str:
    """
    Get standard momentum color emoji using standard thresholds.

    Args:
        momentum: Momentum value to color code

    Returns:
        Color emoji string for the momentum value
    """
    config = get_momentum_alerts_config()
    return get_momentum_color_emoji(momentum, config.momentum_threshold, config.momentum_minimum)


def get_momentum_long_color_emoji(momentum: float) -> str:
    """
    Get long-term momentum color emoji using long-term thresholds.

    Args:
        momentum: Momentum value to color code

    Returns:
        Color emoji string for the momentum value
    """
    config = get_momentum_alerts_config()
    return get_momentum_color_emoji(momentum, config.momentum_long_threshold, config.momentum_long_minimum)


def get_momentum_short_color_emoji(momentum: float) -> str:
    """
    Get short-term momentum color emoji using short-term thresholds.

    Args:
        momentum: Momentum value to color code

    Returns:
        Color emoji string for the momentum value
    """
    config = get_momentum_alerts_config()
    return get_momentum_color_emoji(momentum, config.momentum_short_threshold, config.momentum_short_minimum)


def get_urgency_level_dual(momentum: float, momentum_short: float) -> str:
    """
    Get urgency level for Telegram notifications requiring BOTH momentum indicators to be green.

    Args:
        momentum: Regular momentum value to evaluate
        momentum_short: Short-term momentum value to evaluate

    Returns:
        'filtered' - If either momentum is red/yellow, don't send to Telegram
        'urgent' - Only if BOTH momentums are green (>= threshold), urgent Telegram notification
    """
    config = get_momentum_alerts_config()

    # Both momentum values must meet their respective thresholds for urgent notification
    if (momentum < config.momentum_threshold or
        momentum_short < config.momentum_short_threshold):
        return 'filtered'
    else:
        return 'urgent'


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


def get_squeeze_emoji(momentum_per_minute: float) -> str:
    """
    Get squeeze indicator emoji based on momentum per minute.

    Args:
        momentum_per_minute: Momentum value (percent per minute)

    Returns:
        Emoji string for squeeze indicator
        🟢 for squeezing (>= 1.0% per minute)
        🔴 for not squeezing (< 1.0% per minute)
    """
    if momentum_per_minute >= 1.0:
        return "🟢"  # Squeezing - high momentum
    else:
        return "🔴"  # Not squeezing
