#!/usr/bin/env python3
"""
Find Last Market Close Data

A utility function to identify the actual last market close bar (4:00 PM ET) from
collected bar data. This function is used to establish a baseline closing price
for calculating gains in premarket and post-market scenarios.

Key Features:
- Finds the most recent completed trading day's market close
- Uses only bars at or before 4:00 PM ET (excludes after-hours data)
- Returns the last bar at or before 4:00 PM for consistent baseline pricing
- Handles weekends and determines proper target close date
"""

from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Optional
import pandas as pd
import pytz


def find_last_market_close_data(
    bars_data: Dict[str, pd.DataFrame],
    et_tz: pytz.timezone,
    market_close: dt_time,
    verbose: bool = False,
    tracked_symbols: Optional[list] = None
) -> Optional[Dict]:
    """
    Find the actual last market close bar (4:00 PM ET) from the collected data.

    Args:
        bars_data: Dictionary of symbol -> DataFrame of 5-minute bars
        et_tz: Eastern timezone object (pytz.timezone('US/Eastern'))
        market_close: Market close time (dt_time object, typically 16:00)
        verbose: Enable verbose logging (default: False)
        tracked_symbols: List of symbols to provide detailed debugging for

    Returns:
        Dictionary with close_timestamp and close_bars_dict, or None if not found
        Structure:
        {
            'close_timestamp': datetime,  # The market close timestamp (4:00 PM ET)
            'close_bars_dict': {
                'SYMBOL': {
                    'close_price': float,
                    'close_time': datetime,
                    'bar_data': pandas.Series
                }
            },
            'target_date': date  # The date of the market close
        }
    """
    if not bars_data:
        return None

    if tracked_symbols is None:
        tracked_symbols = []

    # Get a representative symbol's data to find market close times
    sample_symbol = next(iter(bars_data.keys()))
    sample_bars = bars_data[sample_symbol]

    if sample_bars.empty:
        return None

    current_et = datetime.now(et_tz)
    current_time = current_et.time()
    current_date = current_et.date()

    # Find the most recent completed trading day
    target_date = current_date

    # If market hasn't closed yet today, use yesterday
    if current_et.weekday() < 5 and current_time < market_close:
        target_date = current_date - timedelta(days=1)

    # Skip back to most recent weekday
    while target_date.weekday() >= 5:  # Skip weekends
        target_date -= timedelta(days=1)

    if verbose:
        print(f"Looking for market close on: {target_date}")

    # Look for the 4:00 PM ET bar on the target date across all symbols
    target_close_time = et_tz.localize(
        datetime.combine(target_date, market_close)
    )

    # Find bars closest to 4:00 PM ET (within 10 minutes)
    close_bars_dict = {}
    found_any_close = False

    for symbol, bars_df in bars_data.items():
        if bars_df.empty:
            continue

        # Filter to target date
        target_date_bars = bars_df[bars_df.index.date == target_date]

        if target_date_bars.empty:
            continue

        # Find the bar closest to 4:00 PM ET (market close)
        # Only use bars at or before 4:00 PM to avoid after-hours prices
        market_close_time = et_tz.localize(
            datetime.combine(target_date, market_close)  # 4:00 PM ET
        )

        # Filter to bars at or before 4:00 PM (regular market hours)
        regular_hours_bars = target_date_bars[target_date_bars.index <= market_close_time]

        if regular_hours_bars.empty:
            # Skip symbols without bars at or before 4:00 PM
            continue

        # Use the LAST bar at or before 4:00 PM
        last_bar = regular_hours_bars.iloc[-1]
        last_time = regular_hours_bars.index[-1]

        close_bars_dict[symbol] = {
            'close_price': float(last_bar['close']),
            'close_time': last_time,
            'bar_data': last_bar
        }
        found_any_close = True

        # Debug logging for tracked symbols
        is_tracked = symbol in tracked_symbols
        if (verbose and symbol == sample_symbol) or is_tracked:
            time_diff_minutes = abs((last_time - target_close_time).total_seconds()) / 60
            print(f"Found market close bar for {symbol}: {last_time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                  f"(${last_bar['close']:.2f}, {time_diff_minutes:.0f} min from 4PM)")
            if is_tracked:
                print(f"!!! TRACKED SYMBOL {symbol} - Previous close bar details:")
                print(f"    All bars on {target_date}: {len(target_date_bars)} bars")
                print(f"    First bar: {target_date_bars.index[0]} @ ${target_date_bars.iloc[0]['close']:.4f}")
                print(f"    Last bar: {target_date_bars.index[-1]} @ ${target_date_bars.iloc[-1]['close']:.4f}")

    if found_any_close:
        return {
            'close_timestamp': target_close_time,
            'close_bars_dict': close_bars_dict,
            'target_date': target_date
        }

    if verbose:
        print("Could not find market close bars in data")
    return None
