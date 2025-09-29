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


def is_stock_halted(symbol_data: pd.DataFrame, symbol: str,
                    logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if a stock is halted by analyzing gaps in minute-by-minute data.

    A stock is considered halted if:
    1. There's a gap larger than 2 minutes between consecutive timestamps, OR
    2. The most recent data is more than 3 minutes old (current halt)

    This uses the knowledge that halted stocks have missing data entirely.

    Args:
        symbol_data: DataFrame with timestamp index or timestamp column
        symbol: Stock symbol for logging purposes
        logger: Optional logger instance for debug/info messages

    Returns:
        True if stock appears to be halted, False if trading normally
    """
    from datetime import datetime, timedelta
    import pytz

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        if symbol_data is None or len(symbol_data) < 1:
            logger.warning(f"Insufficient market data for halt detection: "
                           f"{symbol}")
            return False  # Assume not halted if insufficient data

        data_to_check = symbol_data.copy()
        timestamps = None

        # Handle both timestamp index and timestamp column
        if hasattr(data_to_check.index, 'to_pydatetime'):
            # DataFrame has timestamp index (common in momentum alerts)
            timestamps = data_to_check.index.to_list()
            logger.debug(f"Using timestamp index for halt detection: "
                         f"{symbol}")
        elif 'timestamp' in data_to_check.columns:
            # DataFrame has timestamp column (legacy format)
            data_to_check['timestamp'] = pd.to_datetime(
                data_to_check['timestamp'])
            data_to_check = data_to_check.sort_values('timestamp')
            timestamps = data_to_check['timestamp'].tolist()
            logger.debug(f"Using timestamp column for halt detection: "
                         f"{symbol}")
        else:
            logger.warning(f"No timestamp data found for halt detection: "
                           f"{symbol}")
            return False

        if len(timestamps) < 1:
            return False

        # Check 1: Is the most recent data too old? (indicating current halt)
        current_time = datetime.now(pytz.timezone('US/Eastern'))
        most_recent_time = timestamps[-1]

        # Ensure timezone awareness for comparison
        et_tz = pytz.timezone('US/Eastern')
        if most_recent_time.tzinfo is None:
            most_recent_time = et_tz.localize(most_recent_time)
        elif most_recent_time.tzinfo != et_tz:
            most_recent_time = most_recent_time.astimezone(et_tz)

        time_since_last_data = ((current_time - most_recent_time)
                                 .total_seconds())

        # If most recent data is more than 3 minutes old, likely halted
        if time_since_last_data > 180:  # 3 minutes
            logger.info(
                f"Trading halt detected for {symbol}: "
                f"Most recent data is {time_since_last_data/60:.1f}min old "
                f"(last: {most_recent_time.strftime('%H:%M:%S ET')})"
            )
            return True

        # Check 2: Look for gaps between consecutive timestamps
        if len(timestamps) >= 2:
            for i in range(1, len(timestamps)):
                time_diff = timestamps[i] - timestamps[i - 1]

                # If gap is more than 2 minutes, consider it a halt
                # (allowing for some market irregularities)
                if time_diff.total_seconds() > 120:  # 2 minutes
                    logger.info(
                        f"Trading halt detected for {symbol}: "
                        f"gap from {timestamps[i-1].strftime('%H:%M:%S')} to "
                        f"{timestamps[i].strftime('%H:%M:%S')} "
                        f"({time_diff.total_seconds()/60:.1f} minutes)"
                    )
                    return True

        logger.debug(f"Halt detection PASSED: {symbol} - trading OK")
        return False

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