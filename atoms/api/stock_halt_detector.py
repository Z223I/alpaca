"""
Stock Halt Detector

This module provides functionality to detect if a stock is halted by analyzing
gaps in minute-by-minute trading data. A stock is considered halted if there's
a gap larger than 1 minute between consecutive timestamps.

This mirrors the halt detection logic from code/orb_alerts.py but provides
it as a reusable API component.
"""

import logging
import pandas as pd
from typing import Optional


def is_stock_halted(symbol_data: pd.DataFrame, symbol: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if a stock is halted by analyzing gaps in minute-by-minute data.

    A stock is considered halted if there's a gap larger than 1 minute between
    consecutive timestamps in the recent data, indicating missing minute bars.

    Args:
        symbol_data: DataFrame containing market data with timestamp column
        symbol: Stock symbol for logging purposes
        logger: Optional logger instance for debug/info messages

    Returns:
        True if stock appears to be halted, False if trading normally
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        if symbol_data is None or len(symbol_data) < 2:
            logger.warning(f"Insufficient market data for halt detection: {symbol}")
            return False  # Assume not halted if insufficient data

        # Check all data points for gaps indicating a halt
        data_to_check = symbol_data.copy()

        if len(data_to_check) < 2:
            return False  # Not enough data to detect halt

        # Convert timestamp column to datetime if it's not already
        if 'timestamp' in data_to_check.columns:
            data_to_check['timestamp'] = pd.to_datetime(data_to_check['timestamp'])

            # Sort by timestamp to ensure proper ordering
            data_to_check = data_to_check.sort_values('timestamp')

            # Check for gaps between consecutive timestamps
            timestamps = data_to_check['timestamp'].tolist()

            for i in range(1, len(timestamps)):
                time_diff = timestamps[i] - timestamps[i - 1]

                # If gap is more than 61 seconds (allowing for slight timing variations),
                # consider it a halt
                if time_diff.total_seconds() > 61:
                    logger.info(
                        f"Trading halt detected for {symbol}: "
                        f"gap from {timestamps[i-1].strftime('%H:%M:%S')} to "
                        f"{timestamps[i].strftime('%H:%M:%S')} "
                        f"({time_diff.total_seconds():.0f} seconds)"
                    )
                    return True  # Stock appears to be halted

            logger.debug(
                f"Halt detection PASSED: {symbol} - no gaps detected in data window"
            )
            return False  # No halt detected
        else:
            logger.warning(
                f"No timestamp column found for halt detection: {symbol}"
            )
            return False  # Assume not halted if no timestamp data available

    except Exception as e:
        logger.error(f"Error in halt detection for {symbol}: {e}")
        return False  # Assume not halted if error occurs


def get_halt_status_emoji(is_halted: bool) -> str:
    """
    Get halt status emoji for display.

    Args:
        is_halted: Boolean indicating if stock is halted

    Returns:
        🔴 if halted, 🟢 if not halted
    """
    return "🔴" if is_halted else "🟢"


def get_halt_status_text(is_halted: bool) -> str:
    """
    Get halt status text for display.

    Args:
        is_halted: Boolean indicating if stock is halted

    Returns:
        "HALTED" if halted, "TRADING" if not halted
    """
    return "HALTED" if is_halted else "TRADING"