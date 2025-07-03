"""
Atom for calculating Volume Weighted Average Price (VWAP) from data.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_vwap(df: pd.DataFrame,
                   price_column: str = 'close') -> Tuple[bool, pd.Series]:
    """
    Calculate Volume Weighted Average Price (VWAP) for candlestick data.

    Args:
        df: DataFrame containing candlestick data
        price_column: Column name to use for price data (default: 'close')

    Returns:
        tuple: (success: bool, vwap: pandas Series)
    """
    # Initialize VWAP series with zeros
    vwap = pd.Series([0.0] * len(df), index=df.index)
    
    if len(df) == 0:
        return False, vwap

    if price_column not in df.columns:
        return False, vwap

    if 'volume' not in df.columns:
        return False, vwap

    # Calculate VWAP using cumulative price*volume / cumulative volume
    price_volume = df[price_column] * df['volume']
    cumulative_pv = price_volume.cumsum()
    cumulative_volume = df['volume'].cumsum()

    # Avoid division by zero
    vwap_values = cumulative_pv / cumulative_volume.replace(0, np.nan)
    vwap_values = vwap_values.fillna(0.0)

    return True, vwap_values


def calculate_vwap_typical(df: pd.DataFrame) -> Tuple[bool, pd.Series]:
    """
    Calculate VWAP using typical price (HLC/3) for candlestick data.

    Args:
        df: DataFrame containing candlestick data
           (must have high, low, close, volume)

    Returns:
        tuple: (success: bool, vwap: pandas Series)
    """
    # Initialize VWAP series with zeros
    vwap = pd.Series([0.0] * len(df), index=df.index)
    
    if len(df) == 0:
        return False, vwap

    required_columns = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        return False, vwap

    # Calculate typical price (HLC/3)
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Calculate VWAP using typical price
    price_volume = typical_price * df['volume']
    cumulative_pv = price_volume.cumsum()
    cumulative_volume = df['volume'].cumsum()

    # Avoid division by zero
    vwap_values = cumulative_pv / cumulative_volume.replace(0, np.nan)
    vwap_values = vwap_values.fillna(0.0)

    return True, vwap_values


def calculate_vwap_hlc(df: pd.DataFrame) -> Tuple[bool, pd.Series]:
    """
    Calculate VWAP using HLC average price for candlestick data.
    Same as typical price method - kept for compatibility.

    Args:
        df: DataFrame containing candlestick data
           (must have high, low, close, volume)

    Returns:
        tuple: (success: bool, vwap: pandas Series)
    """
    return calculate_vwap_typical(df)


def calculate_vwap_ohlc(df: pd.DataFrame) -> Tuple[bool, pd.Series]:
    """
    Calculate VWAP using OHLC average price for candlestick data.

    Args:
        df: DataFrame containing candlestick data
           (must have open, high, low, close, volume)

    Returns:
        tuple: (success: bool, vwap: pandas Series)
    """
    # Initialize VWAP series with zeros
    vwap = pd.Series([0.0] * len(df), index=df.index)
    
    if len(df) == 0:
        return False, vwap

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        return False, vwap

    # Calculate OHLC average price
    ohlc_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Calculate VWAP using OHLC price
    price_volume = ohlc_price * df['volume']
    cumulative_pv = price_volume.cumsum()
    cumulative_volume = df['volume'].cumsum()

    # Avoid division by zero
    vwap_values = cumulative_pv / cumulative_volume.replace(0, np.nan)
    vwap_values = vwap_values.fillna(0.0)

    return True, vwap_values


# Example usage
if __name__ == "__main__":
    # Create sample candlestick data
    candlestick_data = {
        'open': [100, 102, 101, 103, 104, 102, 105, 107, 106, 108,
                 110, 109, 111, 112],
        'high': [102, 104, 103, 105, 106, 104, 107, 109, 108, 110,
                 112, 111, 113, 114],
        'low': [99, 101, 100, 102, 103, 101, 104, 106, 105, 107,
                109, 108, 110, 111],
        'close': [101, 103, 102, 104, 105, 103, 106, 108, 107, 109,
                  111, 110, 112, 113],
        'volume': [1000, 1200, 800, 1500, 1100, 900, 1300, 1600, 1000,
                   1400, 1800, 1200, 1500, 1700]
    }
    
    # Create DataFrame
    df = pd.DataFrame(candlestick_data)
    df.index = pd.date_range('2024-01-01', periods=len(df), freq='D')
    
    print("Sample Candlestick Data:")
    print(df.head())
    print()

    # Calculate VWAP using close prices (default)
    success, vwap_result = calculate_vwap(df, price_column='close')

    if success:
        df['vwap_close'] = vwap_result
        print("VWAP using Close prices:")
        for i, (idx, row) in enumerate(df.iterrows()):
            date_str = str(idx)[:10]
            print(f"Day {i+1} ({date_str}): Close=${row['close']:.2f}, "
                  f"Volume={row['volume']}, VWAP=${row['vwap_close']:.2f}")
    else:
        print("Failed to calculate VWAP using Close prices")

    print("\n" + "="*80 + "\n")

    # Calculate VWAP using typical price (HLC/3)
    success, vwap_typical_result = calculate_vwap_typical(df)

    if success:
        df['vwap_typical'] = vwap_typical_result
        print("VWAP using Typical Price (HLC/3):")
        for i, (idx, row) in enumerate(df.iterrows()):
            typical = (row['high'] + row['low'] + row['close']) / 3
            date_str = str(idx)[:10]
            print(f"Day {i+1} ({date_str}): Typical=${typical:.2f}, "
                  f"Volume={row['volume']}, VWAP=${row['vwap_typical']:.2f}")
    else:
        print("Failed to calculate VWAP using Typical Price")

    print("\n" + "="*80 + "\n")

    # Calculate VWAP using OHLC average price
    success, vwap_ohlc_result = calculate_vwap_ohlc(df)

    if success:
        df['vwap_ohlc'] = vwap_ohlc_result
        print("VWAP using OHLC Average Price:")
        for i, (idx, row) in enumerate(df.iterrows()):
            ohlc = (row['open'] + row['high'] + row['low'] + row['close']) / 4
            date_str = str(idx)[:10]
            print(f"Day {i+1} ({date_str}): OHLC=${ohlc:.2f}, "
                  f"Volume={row['volume']}, VWAP=${row['vwap_ohlc']:.2f}")
    else:
        print("Failed to calculate VWAP using OHLC Average Price")

    print("\n" + "="*80 + "\n")

    # Example of failure case - missing volume column
    df_no_volume = df.drop('volume', axis=1)
    success, vwap_fail = calculate_vwap(df_no_volume, price_column='close')

    print("Test with missing volume column:")
    if success:
        print("VWAP calculation succeeded (unexpected)")
    else:
        print("VWAP calculation failed as expected - returning zeros")
        print(f"VWAP values (first 5): {vwap_fail.head().tolist()}")

    print("\n" + "="*80 + "\n")

    # Example of failure case - invalid price column
    success, vwap_fail2 = calculate_vwap(df, price_column='invalid_column')

    print("Test with invalid price column:")
    if success:
        print("VWAP calculation succeeded (unexpected)")
    else:
        print("VWAP calculation failed as expected - returning zeros")
        print(f"VWAP values (first 5): {vwap_fail2.head().tolist()}")

    print("\nVWAP Formula:")
    print("VWAP = Σ(Price × Volume) / Σ(Volume)")
    print("where the sum is cumulative from the start of the period")
    print("\nVWAP Methods:")
    print("- calculate_vwap(): Uses specified price column (default: 'close')")
    print("- calculate_vwap_typical(): Uses typical price (HLC/3)")
    print("- calculate_vwap_ohlc(): Uses OHLC average (OHLC/4)")
    print("Returns: (success: bool, vwap: pandas Series) - VWAP initialized "
          "to zeros on failure")
