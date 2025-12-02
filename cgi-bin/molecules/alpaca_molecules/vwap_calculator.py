"""
VWAP Calculator Module

Provides reusable VWAP calculation methods that can be used across the application.
VWAP (Volume Weighted Average Price) resets daily and is calculated as:
    cumulative(typical_price × volume) / cumulative(volume)

Where typical_price = (high + low + close) / 3
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def calculate_typical_price(df: pd.DataFrame) -> pd.Series:
    """
    Calculate typical price for each bar.

    Typical price = (high + low + close) / 3

    Args:
        df: DataFrame with 'high', 'low', 'close' columns

    Returns:
        Series of typical prices
    """
    return (df['high'] + df['low'] + df['close']) / 3


def calculate_vwap_for_single_day(df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP for a single day's worth of data.

    This assumes the DataFrame contains data for a single trading day only.
    VWAP is calculated as cumulative(typical_price × volume) / cumulative(volume)

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
            for a single trading day

    Returns:
        Series of VWAP values (cumulative within the day)
    """
    if df.empty or 'volume' not in df.columns:
        return pd.Series(dtype=float)

    typical_price = calculate_typical_price(df)

    # Calculate cumulative VWAP
    cumulative_tp_volume = (typical_price * df['volume']).cumsum()
    cumulative_volume = df['volume'].cumsum()

    # Avoid division by zero
    vwap = cumulative_tp_volume / cumulative_volume.replace(0, np.nan)

    return vwap


def calculate_vwap_by_day(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> List[Dict[str, any]]:
    """
    Calculate VWAP with daily resets.

    This function groups bars by trading day and calculates VWAP separately
    for each day, ensuring VWAP resets at the start of each trading day.

    Args:
        df: DataFrame with columns: timestamp, high, low, close, volume
        timestamp_col: Name of the timestamp column (default: 'timestamp')

    Returns:
        List of dictionaries with 'time' and 'value' keys, suitable for
        charting libraries. Filters out invalid values (NaN, inf, zero volume).
    """
    if df.empty or 'volume' not in df.columns:
        return []

    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure proper datetime conversion
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df['datetime'] = pd.to_datetime(df[timestamp_col], utc=True)
    else:
        df['datetime'] = df[timestamp_col]

    # Extract date for grouping
    df['date'] = df['datetime'].dt.date

    result = []

    # Calculate VWAP separately for each trading day
    for date, day_df in df.groupby('date'):
        vwap_series = calculate_vwap_for_single_day(day_df)

        # Build result list with original timestamps
        for timestamp, vwap_value, volume in zip(
            day_df[timestamp_col], vwap_series, day_df['volume']
        ):
            # Only include valid values with actual volume
            if (not pd.isna(vwap_value) and
                volume > 0 and
                not np.isinf(vwap_value)):
                result.append({
                    'time': timestamp,
                    'value': float(vwap_value)
                })

    return result


def calculate_vwap_from_bars(bars: List[Dict]) -> List[Dict[str, any]]:
    """
    Calculate VWAP from a list of bar dictionaries.

    This is a convenience wrapper around calculate_vwap_by_day that accepts
    a list of bar dictionaries (as returned by APIs) instead of a DataFrame.

    Args:
        bars: List of bar dictionaries with keys: timestamp, high, low, close, volume

    Returns:
        List of dictionaries with 'time' and 'value' keys
    """
    if not bars:
        return []

    df = pd.DataFrame(bars)
    return calculate_vwap_by_day(df)


def get_latest_vwap(bars: List[Dict]) -> float:
    """
    Get the most recent VWAP value from a list of bars.

    Args:
        bars: List of bar dictionaries with keys: timestamp, high, low, close, volume

    Returns:
        Most recent VWAP value, or None if calculation fails
    """
    vwap_data = calculate_vwap_from_bars(bars)

    if vwap_data:
        return vwap_data[-1]['value']

    return None
