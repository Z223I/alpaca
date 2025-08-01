"""
Atom for calculating Exponential Moving Average (EMA) from candlestick data.
"""

import pandas as pd
from typing import Tuple


def calculate_ema(df: pd.DataFrame, period: int = 9) -> Tuple[bool, pd.Series]:
    """
    Calculate Exponential Moving Average (EMA) for candlestick data using typical price (HLC/3).

    Args:
        df: DataFrame containing candlestick data with 'high', 'low', 'close' columns
        period: Number of periods for EMA calculation (default: 9)

    Returns:
        tuple: (success: bool, ema: pandas Series)
    """
    # Initialize EMA series with zeros
    ema = pd.Series([0.0] * len(df), index=df.index)

    if len(df) < period:
        return False, ema

    # Check required columns
    required_columns = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        return False, ema

    # Calculate typical price (HLC/3)
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Calculate EMA using pandas built-in function
    ema_values = typical_price.ewm(span=period, adjust=False).mean()

    return True, ema_values


def calculate_ema_manual(df: pd.DataFrame, period: int = 9) -> Tuple[bool, pd.Series]:
    """
    Manual calculation of EMA using the traditional formula and typical price (HLC/3).

    Args:
        df: DataFrame containing candlestick data with 'high', 'low', 'close' columns
        period: Number of periods for EMA calculation (default: 9)

    Returns:
        tuple: (success: bool, ema: pandas Series)
    """
    # Initialize EMA series with zeros
    ema_values = [0.0] * len(df)

    if len(df) < period:
        return False, pd.Series(ema_values, index=df.index)

    # Check required columns
    required_columns = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        return False, pd.Series(ema_values, index=df.index)

    # Calculate typical price (HLC/3)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    prices = typical_price.values

    # Calculate the smoothing factor (alpha)
    alpha = 2 / (period + 1)

    # Initialize with Simple Moving Average for the first EMA value
    sma = sum(prices[:period]) / period
    ema_values[period - 1] = sma

    # Calculate subsequent EMA values
    for i in range(period, len(prices)):
        ema = alpha * prices[i] + (1 - alpha) * ema_values[i - 1]
        ema_values[i] = ema

    return True, pd.Series(ema_values, index=df.index)


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

    # Calculate EMA using pandas (typical price HLC/3, 9-period)
    success, ema_result = calculate_ema(df, period=9)

    if success:
        df['ema_9'] = ema_result
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        print("EMA (9-period) using pandas - Typical Price (HLC/3):")
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['ema_9'] > 0:
                date_str = str(idx)[:10]
                print(f"Day {i+1} ({date_str}): TypicalPrice=${row['typical_price']:.2f}, "
                      f"EMA=${row['ema_9']:.2f}")
            else:
                date_str = str(idx)[:10]
                print(f"Day {i+1} ({date_str}): TypicalPrice=${row['typical_price']:.2f}, "
                      f"EMA=0.00 (insufficient data)")
    else:
        print("Failed to calculate EMA using pandas")

    print("\n" + "="*70 + "\n")

    # Calculate EMA manually
    success, ema_manual_result = calculate_ema_manual(df, period=9)

    if success:
        df['ema_9_manual'] = ema_manual_result
        print("EMA (9-period) manual calculation - Typical Price (HLC/3):")
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['ema_9_manual'] > 0:
                date_str = str(idx)[:10]
                print(f"Day {i+1} ({date_str}): TypicalPrice=${row['typical_price']:.2f}, "
                      f"EMA=${row['ema_9_manual']:.2f}")
            else:
                date_str = str(idx)[:10]
                print(f"Day {i+1} ({date_str}): TypicalPrice=${row['typical_price']:.2f}, "
                      f"EMA=0.00 (insufficient data)")
    else:
        print("Failed to calculate EMA manually")

    print("\n" + "="*70 + "\n")

    # Example with different period
    success, ema_20_result = calculate_ema(df, period=20)

    if success:
        df['ema_20'] = ema_20_result
        print("EMA (20-period) using Typical Price (HLC/3):")
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['ema_20'] > 0:
                date_str = str(idx)[:10]
                print(f"Day {i+1} ({date_str}): TypicalPrice=${row['typical_price']:.2f}, "
                      f"EMA=${row['ema_20']:.2f}")
            else:
                date_str = str(idx)[:10]
                print(f"Day {i+1} ({date_str}): TypicalPrice=${row['typical_price']:.2f}, "
                      f"EMA=0.00 (insufficient data)")
    else:
        print("Failed to calculate EMA (20-period) - insufficient data")

    print("\n" + "="*70 + "\n")

    # Example of failure case - insufficient data
    small_df = df.head(5)  # Only 5 rows, less than period of 9
    success, ema_fail = calculate_ema(small_df, period=9)

    print("Test with insufficient data (5 rows, period=9):")
    if success:
        print("EMA calculation succeeded (unexpected)")
    else:
        print("EMA calculation failed as expected - returning zeros")
        print(f"EMA values: {ema_fail.tolist()}")

    print("\n" + "="*70 + "\n")

    # Example of failure case - missing required columns
    incomplete_df = df[['open', 'volume']].copy()  # Missing high, low, close
    success, ema_fail2 = calculate_ema(incomplete_df, period=9)

    print("Test with missing required columns (high, low, close):")
    if success:
        print("EMA calculation succeeded (unexpected)")
    else:
        print("EMA calculation failed as expected - returning zeros")
        print(f"EMA values (first 5): {ema_fail2.head().tolist()}")

    print("\nEMA Formula:")
    print("EMA = α × Typical_Price + (1 - α) × Previous_EMA")
    print("where Typical_Price = (High + Low + Close) / 3")
    print("and α (smoothing factor) = 2 / (period + 1) = 2 / 10 = 0.2 for 9-period")
    print("\nRequired columns: high, low, close")
    print("Returns: (success: bool, ema: pandas Series) - EMA initialized "
          "to zeros on failure")
