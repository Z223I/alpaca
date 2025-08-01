"""
Atom for calculating vector angle by fitting a line through N candlesticks.
"""

import math
import pandas as pd
import numpy as np


def calculate_vector_angle(df: pd.DataFrame, price_column: str = 'close', num_candles: int = 15) -> float:
    """
    Calculate the vector angle by fitting a line through N candlesticks.

    Args:
        df: DataFrame containing stock price data
        price_column: Column name for price data (default: 'close')
        num_candles: Number of candlesticks to analyze (default: 15)

    Returns:
        Angle in degrees (positive = upward trend, negative = downward trend)

    Raises:
        ValueError: If DataFrame has insufficient data or invalid column
    """
    try:
        # Validate inputs
        if df.empty:
            raise ValueError("DataFrame is empty")

        if price_column not in df.columns:
            raise ValueError(f"Column '{price_column}' not found in DataFrame")

        # Check if we have enough data
        if len(df) < num_candles:
            raise ValueError(f"DataFrame has only {len(df)} rows, need at least {num_candles}")

        # Get the first num_candles rows
        subset = df.head(num_candles)

        # Create x values (time points: 0, 1, 2, ..., num_candles-1)
        x = np.arange(num_candles)

        # Check for non-numeric data and NaN values
        try:
            y_series = pd.to_numeric(subset[price_column], errors='raise')
            if y_series.isna().any():
                raise ValueError(f"Column '{price_column}' contains NaN values")
            y = np.array(y_series.values, dtype=float)
        except (ValueError, TypeError):
            raise ValueError(f"Column '{price_column}' contains non-numeric data")

        # Fit a line using linear regression (least squares)
        slope = np.polyfit(x, y, 1)[0]

        # Calculate angle from slope
        # slope = rise/run = tan(angle)
        angle_radians = math.atan(slope)
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    except Exception as e:
        print(f"Error calculating vector angle: {e}")
        raise