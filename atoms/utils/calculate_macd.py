"""
Atom for calculating MACD (Moving Average Convergence Divergence) from candlestick data.
"""

import pandas as pd
from typing import Tuple, Dict, Any


def calculate_macd(df: pd.DataFrame, fast_length: int = 12, slow_length: int = 26, 
                   signal_length: int = 9, source: str = 'close') -> Tuple[bool, Dict[str, pd.Series]]:
    """
    Calculate MACD (Moving Average Convergence Divergence) for candlestick data.

    Args:
        df: DataFrame containing candlestick data with 'high', 'low', 'close' columns
        fast_length: Fast EMA period (default: 12)
        slow_length: Slow EMA period (default: 26)
        signal_length: Signal line EMA period (default: 9)
        source: Source price column ('close', 'open', 'high', 'low', or 'typical' for HLC/3)

    Returns:
        tuple: (success: bool, macd_dict: Dict containing 'macd', 'signal', 'histogram' Series)
    """
    # Initialize empty result
    empty_result = {
        'macd': pd.Series([0.0] * len(df), index=df.index),
        'signal': pd.Series([0.0] * len(df), index=df.index),
        'histogram': pd.Series([0.0] * len(df), index=df.index)
    }

    if len(df) < slow_length:
        return False, empty_result

    # Check required columns based on source
    if source == 'typical':
        required_columns = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return False, empty_result
        source_prices = (df['high'] + df['low'] + df['close']) / 3
    else:
        if source not in df.columns:
            return False, empty_result
        source_prices = df[source]

    # Calculate fast and slow EMAs
    fast_ema = source_prices.ewm(span=fast_length, adjust=False).mean()
    slow_ema = source_prices.ewm(span=slow_length, adjust=False).mean()

    # Calculate MACD line (fast EMA - slow EMA)
    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD line)
    signal_line = macd_line.ewm(span=signal_length, adjust=False).mean()

    # Calculate histogram (MACD - Signal)
    histogram = macd_line - signal_line

    result = {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

    return True, result


def calculate_macd_manual(df: pd.DataFrame, fast_length: int = 12, slow_length: int = 26, 
                          signal_length: int = 9, source: str = 'close') -> Tuple[bool, Dict[str, pd.Series]]:
    """
    Manual calculation of MACD using traditional EMA formula.

    Args:
        df: DataFrame containing candlestick data
        fast_length: Fast EMA period (default: 12)
        slow_length: Slow EMA period (default: 26)
        signal_length: Signal line EMA period (default: 9)
        source: Source price column ('close', 'open', 'high', 'low', or 'typical' for HLC/3)

    Returns:
        tuple: (success: bool, macd_dict: Dict containing 'macd', 'signal', 'histogram' Series)
    """
    # Initialize empty result
    empty_result = {
        'macd': pd.Series([0.0] * len(df), index=df.index),
        'signal': pd.Series([0.0] * len(df), index=df.index),
        'histogram': pd.Series([0.0] * len(df), index=df.index)
    }

    if len(df) < slow_length:
        return False, empty_result

    # Check required columns based on source
    if source == 'typical':
        required_columns = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return False, empty_result
        source_prices = (df['high'] + df['low'] + df['close']) / 3
    else:
        if source not in df.columns:
            return False, empty_result
        source_prices = df[source]

    prices = source_prices.values
    n_periods = len(prices)

    # Calculate smoothing factors (alpha)
    fast_alpha = 2 / (fast_length + 1)
    slow_alpha = 2 / (slow_length + 1)
    signal_alpha = 2 / (signal_length + 1)

    # Initialize arrays
    fast_ema = [0.0] * n_periods
    slow_ema = [0.0] * n_periods
    macd_values = [0.0] * n_periods
    signal_values = [0.0] * n_periods
    histogram_values = [0.0] * n_periods

    # Initialize fast EMA with SMA
    fast_sma = sum(prices[:fast_length]) / fast_length
    fast_ema[fast_length - 1] = fast_sma

    # Calculate subsequent fast EMA values
    for i in range(fast_length, n_periods):
        fast_ema[i] = fast_alpha * prices[i] + (1 - fast_alpha) * fast_ema[i - 1]

    # Initialize slow EMA with SMA
    slow_sma = sum(prices[:slow_length]) / slow_length
    slow_ema[slow_length - 1] = slow_sma

    # Calculate subsequent slow EMA values
    for i in range(slow_length, n_periods):
        slow_ema[i] = slow_alpha * prices[i] + (1 - slow_alpha) * slow_ema[i - 1]

    # Calculate MACD line starting from slow_length
    for i in range(slow_length - 1, n_periods):
        macd_values[i] = fast_ema[i] - slow_ema[i]

    # Calculate signal line (EMA of MACD)
    # Start signal calculation after we have enough MACD values
    signal_start_idx = slow_length - 1 + signal_length - 1
    if signal_start_idx < n_periods:
        # Initialize signal EMA with SMA of MACD values
        signal_sma = sum(macd_values[slow_length - 1:signal_start_idx + 1]) / signal_length
        signal_values[signal_start_idx] = signal_sma

        # Calculate subsequent signal EMA values
        for i in range(signal_start_idx + 1, n_periods):
            signal_values[i] = signal_alpha * macd_values[i] + (1 - signal_alpha) * signal_values[i - 1]

    # Calculate histogram
    for i in range(n_periods):
        histogram_values[i] = macd_values[i] - signal_values[i]

    result = {
        'macd': pd.Series(macd_values, index=df.index),
        'signal': pd.Series(signal_values, index=df.index),
        'histogram': pd.Series(histogram_values, index=df.index)
    }

    return True, result


# Example usage
if __name__ == "__main__":
    # Create sample candlestick data
    candlestick_data = {
        'open': [100, 102, 101, 103, 104, 102, 105, 107, 106, 108,
                 110, 109, 111, 112, 114, 113, 115, 117, 116, 118,
                 120, 119, 121, 122, 124, 123, 125, 127, 126, 128,
                 130, 129, 131, 132, 134, 133, 135, 137, 136, 138],
        'high': [102, 104, 103, 105, 106, 104, 107, 109, 108, 110,
                 112, 111, 113, 114, 116, 115, 117, 119, 118, 120,
                 122, 121, 123, 124, 126, 125, 127, 129, 128, 130,
                 132, 131, 133, 134, 136, 135, 137, 139, 138, 140],
        'low': [99, 101, 100, 102, 103, 101, 104, 106, 105, 107,
                109, 108, 110, 111, 113, 112, 114, 116, 115, 117,
                119, 118, 120, 121, 123, 122, 124, 126, 125, 127,
                129, 128, 130, 131, 133, 132, 134, 136, 135, 137],
        'close': [101, 103, 102, 104, 105, 103, 106, 108, 107, 109,
                  111, 110, 112, 113, 115, 114, 116, 118, 117, 119,
                  121, 120, 122, 123, 125, 124, 126, 128, 127, 129,
                  131, 130, 132, 133, 135, 134, 136, 138, 137, 139]
    }

    # Create DataFrame
    df = pd.DataFrame(candlestick_data)
    df.index = pd.date_range('2024-01-01', periods=len(df), freq='D')

    print("Sample Candlestick Data (first 10 rows):")
    print(df.head(10))
    print()

    # Calculate MACD using pandas (FastLength=12, SlowLength=26, Source=close, SignalLength=9)
    success, macd_result = calculate_macd(df, fast_length=12, slow_length=26, 
                                         signal_length=9, source='close')

    if success:
        print("MACD (12,26,9) using pandas - Close Price:")
        print("=" * 80)
        for i, (idx, row) in enumerate(df.iterrows()):
            macd_val = macd_result['macd'].iloc[i]
            signal_val = macd_result['signal'].iloc[i]
            hist_val = macd_result['histogram'].iloc[i]
            
            date_str = str(idx)[:10]
            if i >= 25:  # Start showing from index where we have meaningful values
                print(f"Day {i+1:2d} ({date_str}): Close=${row['close']:6.2f}, "
                      f"MACD={macd_val:7.3f}, Signal={signal_val:7.3f}, Histogram={hist_val:7.3f}")
    else:
        print("Failed to calculate MACD using pandas")

    print("\n" + "="*80 + "\n")

    # Calculate MACD manually
    success, macd_manual_result = calculate_macd_manual(df, fast_length=12, slow_length=26, 
                                                       signal_length=9, source='close')

    if success:
        print("MACD (12,26,9) manual calculation - Close Price:")
        print("=" * 80)
        for i, (idx, row) in enumerate(df.iterrows()):
            macd_val = macd_manual_result['macd'].iloc[i]
            signal_val = macd_manual_result['signal'].iloc[i]
            hist_val = macd_manual_result['histogram'].iloc[i]
            
            date_str = str(idx)[:10]
            if i >= 25:  # Start showing from index where we have meaningful values
                print(f"Day {i+1:2d} ({date_str}): Close=${row['close']:6.2f}, "
                      f"MACD={macd_val:7.3f}, Signal={signal_val:7.3f}, Histogram={hist_val:7.3f}")
    else:
        print("Failed to calculate MACD manually")

    print("\n" + "="*80 + "\n")

    # Example with typical price source
    success, macd_typical_result = calculate_macd(df, fast_length=12, slow_length=26, 
                                                 signal_length=9, source='typical')

    if success:
        print("MACD (12,26,9) using Typical Price (HLC/3):")
        print("=" * 80)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        for i, (idx, row) in enumerate(df.iterrows()):
            macd_val = macd_typical_result['macd'].iloc[i]
            signal_val = macd_typical_result['signal'].iloc[i]
            hist_val = macd_typical_result['histogram'].iloc[i]
            
            date_str = str(idx)[:10]
            if i >= 25:  # Start showing from index where we have meaningful values
                print(f"Day {i+1:2d} ({date_str}): Typical=${typical_price.iloc[i]:6.2f}, "
                      f"MACD={macd_val:7.3f}, Signal={signal_val:7.3f}, Histogram={hist_val:7.3f}")
    else:
        print("Failed to calculate MACD using typical price")

    print("\n" + "="*80 + "\n")

    # Test failure case - insufficient data
    small_df = df.head(20)  # Only 20 rows, less than slow_length of 26
    success, macd_fail = calculate_macd(small_df, fast_length=12, slow_length=26, 
                                       signal_length=9, source='close')

    print("Test with insufficient data (20 rows, slow_length=26):")
    if success:
        print("MACD calculation succeeded (unexpected)")
    else:
        print("MACD calculation failed as expected - returning zeros")
        print(f"MACD values (first 5): {macd_fail['macd'].head().tolist()}")

    print("\n" + "="*80 + "\n")

    print("MACD Formula:")
    print("Fast EMA = EMA(source_price, fast_length)")
    print("Slow EMA = EMA(source_price, slow_length)")
    print("MACD Line = Fast EMA - Slow EMA")
    print("Signal Line = EMA(MACD Line, signal_length)")
    print("Histogram = MACD Line - Signal Line")
    print(f"\nDefault parameters: FastLength=12, SlowLength=26, SignalLength=9, Source=close")
    print("Required columns: Depends on source ('close', 'open', 'high', 'low', or 'high'/'low'/'close' for typical)")
    print("Returns: (success: bool, dict with 'macd'/'signal'/'histogram' Series) - initialized to zeros on failure")