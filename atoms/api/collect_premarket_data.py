#!/usr/bin/env python3
"""
Collect Premarket Data

A utility function to collect 5-minute bar data for symbols over a specified lookback period,
then filter the data to include only bars after the last market close (4:00 PM ET).

Key Features:
- Fetches 5-minute bars for a configurable lookback period (default: 7 days)
- Processes symbols in batches to avoid API limits
- Includes retry logic for failed symbol fetches
- Identifies the last market close and filters bars after that time
- Returns premarket data with previous close prices for gain calculations
"""

import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any
import pandas as pd
import pytz
import alpaca_trade_api as tradeapi
from atoms.api.find_last_market_close_data import find_last_market_close_data


def collect_premarket_data(
    client: Any,
    symbols: List[str],
    criteria: Any,
    et_tz: pytz.timezone,
    market_close: dt_time,
    verbose: bool = False,
    tracked_symbols: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Collect 5-minute bars for the last N days, then filter for premarket data.

    Args:
        client: Alpaca API client instance
        symbols: List of stock symbols to collect data for
        criteria: PremarketCriteria object containing:
            - feed: Data feed ('iex' or 'sip')
            - lookback_days: Number of days to look back
        et_tz: Eastern timezone object (pytz.timezone('US/Eastern'))
        market_close: Market close time (dt_time object, typically 16:00)
        verbose: Enable verbose logging (default: False)
        tracked_symbols: List of symbols to provide detailed debugging for

    Returns:
        Dictionary mapping symbols to their premarket data:
        {
            'SYMBOL': {
                'premarket_bars': pd.DataFrame,  # Bars after market close
                'previous_close': float,         # Close price at 4:00 PM ET
                'previous_close_time': datetime, # Timestamp of close
                'last_market_close_ref': datetime # Reference market close time
            }
        }
    """
    if tracked_symbols is None:
        tracked_symbols = []

    if verbose:
        print(f"Collecting premarket data for {len(symbols)} symbols...")
        print(f"Using {criteria.feed.upper()} feed with {criteria.lookback_days}-day lookback")

    # Calculate date range (lookback_days back to capture last market close)
    end_time = datetime.now(et_tz)
    start_time = end_time - timedelta(days=criteria.lookback_days)

    if verbose:
        print(f"Data range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} ET")

    all_bars_data = {}
    failed_symbols = []  # Track symbols that failed to fetch
    batch_size = 50  # Process symbols in batches to avoid API limits

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]

        if verbose:
            print(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")

        try:
            # Get 5-minute bars for each symbol individually
            for symbol in batch_symbols:
                try:
                    bars = client.get_bars(
                        symbol,  # Individual symbol request
                        tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),  # 5-minute bars
                        start=start_time.strftime('%Y-%m-%d'),
                        end=end_time.strftime('%Y-%m-%d'),
                        limit=5000,  # Large limit to capture all data
                        feed=criteria.feed
                    )

                    if bars and len(bars) > 0:
                        # Convert to DataFrame format
                        bar_data = []
                        for bar in bars:
                            bar_dict = {
                                'open': float(bar.o),
                                'high': float(bar.h),
                                'low': float(bar.l),
                                'close': float(bar.c),
                                'volume': int(bar.v),
                                'timestamp': bar.t
                            }
                            bar_data.append(bar_dict)

                        if bar_data:
                            df = pd.DataFrame(bar_data)
                            df.set_index('timestamp', inplace=True)
                            # Ensure timezone-aware index
                            if df.index.tz is None:
                                df.index = df.index.tz_localize('UTC')
                            df.index = df.index.tz_convert(et_tz)

                            all_bars_data[symbol] = df
                    else:
                        # No bars returned - might be inactive symbol
                        if symbol not in failed_symbols:
                            failed_symbols.append(symbol)

                except Exception as symbol_error:
                    error_message = str(symbol_error)
                    # Suppress specific error messages
                    should_suppress = (
                        "Period 'max' is invalid" in error_message or
                        "possibly delisted" in error_message or
                        "Period" in error_message
                    )

                    if verbose and not should_suppress:
                        print(f"Error fetching data for {symbol}: {symbol_error}")

                    # Track failed symbols for retry
                    if symbol not in failed_symbols:
                        failed_symbols.append(symbol)

                    # Log specific symbols we're tracking
                    if symbol in tracked_symbols:
                        print(f"!!! TRACKED SYMBOL {symbol} - Error during data fetch: {symbol_error}")
                    continue

        except Exception as e:
            if verbose:
                print(f"Error collecting data for batch: {e}")
            continue

    if verbose:
        print(f"Successfully collected raw data for {len(all_bars_data)} symbols")
        if failed_symbols:
            print(f"Failed to collect data for {len(failed_symbols)} symbols")

    # Retry failed symbols (up to 2 retries)
    if failed_symbols:
        max_retries = 2
        for retry_attempt in range(max_retries):
            if not failed_symbols:
                break

            if verbose:
                print(f"\nRetry attempt {retry_attempt + 1}/{max_retries} for {len(failed_symbols)} failed symbols...")

            retry_failed = []
            for symbol in failed_symbols:
                try:
                    bars = client.get_bars(
                        symbol,
                        tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
                        start=start_time.strftime('%Y-%m-%d'),
                        end=end_time.strftime('%Y-%m-%d'),
                        limit=5000,
                        feed=criteria.feed
                    )

                    if bars and len(bars) > 0:
                        bar_data = []
                        for bar in bars:
                            bar_dict = {
                                'open': float(bar.o),
                                'high': float(bar.h),
                                'low': float(bar.l),
                                'close': float(bar.c),
                                'volume': int(bar.v),
                                'timestamp': bar.t
                            }
                            bar_data.append(bar_dict)

                        if bar_data:
                            df = pd.DataFrame(bar_data)
                            df.set_index('timestamp', inplace=True)
                            if df.index.tz is None:
                                df.index = df.index.tz_localize('UTC')
                            df.index = df.index.tz_convert(et_tz)

                            all_bars_data[symbol] = df
                            if verbose:
                                print(f"✓ Retry success for {symbol}")
                    else:
                        # Still no bars - symbol likely inactive
                        retry_failed.append(symbol)

                except Exception as retry_error:
                    error_message = str(retry_error)
                    # Suppress specific error messages
                    should_suppress = (
                        "Period 'max' is invalid" in error_message or
                        "possibly delisted" in error_message or
                        "Period" in error_message
                    )

                    if verbose and not should_suppress:
                        print(f"Retry failed for {symbol}: {retry_error}")
                    retry_failed.append(symbol)

                    if symbol in tracked_symbols:
                        print(f"!!! TRACKED SYMBOL {symbol} - Retry {retry_attempt + 1} failed: {retry_error}")
                    continue

            failed_symbols = retry_failed

            if failed_symbols and retry_attempt < max_retries - 1:
                if verbose:
                    print(f"Waiting 10 seconds before next retry...")
                time.sleep(10)

        if verbose and failed_symbols:
            print(f"\nFinal: {len(failed_symbols)} symbols still failed after retries")
            if tracked_symbols:
                tracked_failed = [s for s in failed_symbols if s in tracked_symbols]
                if tracked_failed:
                    print(f"!!! TRACKED SYMBOLS that failed: {', '.join(tracked_failed)}")

    if verbose:
        print(f"Final total: Successfully collected raw data for {len(all_bars_data)} symbols")

    # Check if tracked symbols made it through data collection
    for tracked in tracked_symbols:
        if tracked in all_bars_data:
            print(f"!!! TRACKED SYMBOL {tracked} - Successfully collected {len(all_bars_data[tracked])} bars")
        else:
            print(f"!!! TRACKED SYMBOL {tracked} - NOT in all_bars_data (not collected or error)")

    # Find the actual last market close data
    market_close_data = find_last_market_close_data(
        bars_data=all_bars_data,
        et_tz=et_tz,
        market_close=market_close,
        verbose=verbose,
        tracked_symbols=tracked_symbols
    )

    if not market_close_data:
        if verbose:
            print("Could not determine last market close. Using fallback method.")
        # Fallback: find the most recent completed trading day at 4 PM
        current_et = datetime.now(et_tz)
        target_date = current_et.date()

        # If market hasn't closed yet today, use yesterday
        if current_et.weekday() < 5 and current_et.time() < market_close:
            target_date = current_et.date() - timedelta(days=1)

        # Skip backwards to most recent weekday
        while target_date.weekday() >= 5:
            target_date -= timedelta(days=1)

        fallback_close_time = et_tz.localize(
            datetime.combine(target_date, market_close)
        )

        if verbose:
            print(f"Using fallback market close: {fallback_close_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Create fallback close data using last available bars before fallback time
        market_close_data = {
            'close_timestamp': fallback_close_time,
            'close_bars_dict': {},
            'target_date': target_date
        }

        for symbol, bars_df in all_bars_data.items():
            # Filter to the target date only
            target_date_bars = bars_df[bars_df.index.date == target_date]
            if target_date_bars.empty:
                continue

            # Only use bars at or before 4:00 PM to avoid after-hours prices
            regular_hours_bars = target_date_bars[target_date_bars.index <= fallback_close_time]

            if regular_hours_bars.empty:
                # Skip symbols without bars at or before 4:00 PM
                continue

            last_bar = regular_hours_bars.iloc[-1]
            market_close_data['close_bars_dict'][symbol] = {
                'close_price': float(last_bar['close']),
                'close_time': regular_hours_bars.index[-1],
                'bar_data': last_bar
            }

    close_timestamp = market_close_data['close_timestamp']
    close_bars_dict = market_close_data['close_bars_dict']

    # Filter data to only include bars since market close
    premarket_data = {}
    for symbol, bars_df in all_bars_data.items():
        # Track specific symbols
        is_tracked = symbol in tracked_symbols

        # Skip symbols without close data
        if symbol not in close_bars_dict:
            if is_tracked:
                print(f"!!! TRACKED SYMBOL {symbol} - NOT in close_bars_dict (no previous close price found)")
            continue

        # Filter for bars after market close timestamp
        premarket_bars = bars_df[bars_df.index > close_timestamp]

        if not premarket_bars.empty:
            close_info = close_bars_dict[symbol]
            premarket_data[symbol] = {
                'premarket_bars': premarket_bars,
                'previous_close': close_info['close_price'],
                'previous_close_time': close_info['close_time'],
                'last_market_close_ref': close_timestamp
            }
            if is_tracked:
                print(f"!!! TRACKED SYMBOL {symbol} - Added to premarket_data: {len(premarket_bars)} bars after {close_timestamp}")
        else:
            if is_tracked:
                print(f"!!! TRACKED SYMBOL {symbol} - No bars after close_timestamp {close_timestamp}")

    if verbose:
        print(f"Filtered to {len(premarket_data)} symbols with premarket activity")
        if close_timestamp:
            print(f"Reference market close: {close_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    return premarket_data
