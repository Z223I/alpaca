"""
Atom for calculating Open Range Breakout (ORB) levels from stock market data.
"""

import pandas as pd
from typing import Optional, Tuple


def calculate_orb_levels(symbol_data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Open Range Breakout (ORB) high and low levels from the first 15 candlesticks.

    Args:
        symbol_data: DataFrame with timestamp, open, high, low, close, volume columns

    Returns:
        Tuple of (orb_high, orb_low) or (None, None) if insufficient data
    """
    isDebugging = False  # Set to True for debugging output

    if isDebugging:
        print("=== ORB Calculation Debug ===")
        print(f"Input data shape: {symbol_data.shape}")
        print(f"Input data columns: {list(symbol_data.columns)}")

    if symbol_data.empty:
        if isDebugging:
            print("DEBUG: No data available for ORB calculation.")
        return None, None

    try:
        # Sort by timestamp to ensure chronological order
        symbol_data = symbol_data.copy().sort_values('timestamp')

        if isDebugging:
            print(f"DEBUG: Total candlesticks available: {len(symbol_data)}")
            print(f"DEBUG: First timestamp: {symbol_data['timestamp'].iloc[0]}")
            print(f"DEBUG: Last timestamp: {symbol_data['timestamp'].iloc[-1]}")

        # Use first 15 candlesticks for ORB calculation
        orb_candlesticks = 15
        if len(symbol_data) < orb_candlesticks:
            if isDebugging:
                print(f"DEBUG: Insufficient data - only {len(symbol_data)} candlesticks available, need {orb_candlesticks}")
            # Use all available data if less than 15 candlesticks
            orb_data = symbol_data
            orb_candlesticks = len(symbol_data)
        else:
            orb_data = symbol_data.head(orb_candlesticks)

        if isDebugging:
            print(f"DEBUG: Using first {orb_candlesticks} candlesticks for ORB")
            print(f"DEBUG: ORB period timestamps: {orb_data['timestamp'].iloc[0]} to {orb_data['timestamp'].iloc[-1]}")
            print(f"DEBUG: ORB data high values: {orb_data['high'].tolist()}")
            print(f"DEBUG: ORB data low values: {orb_data['low'].tolist()}")

        # Calculate ORB high and low from the first N candlesticks
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()

        if isDebugging:
            print(f"DEBUG: Calculated orb_high: {orb_high}, orb_low: {orb_low}")

        if orb_high is None or orb_low is None:
            if isDebugging:
                print("DEBUG: ORB levels could not be calculated - no valid high/low data.")
            return None, None

        if isDebugging:
            print("=== ORB Calculation Complete ===")

        return orb_high, orb_low

    except Exception as e:
        if isDebugging:
            print(f"DEBUG: Exception in ORB calculation: {e}")
            import traceback
            traceback.print_exc()
        return None, None