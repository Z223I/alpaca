"""
Atom for extracting and filtering symbol data from DataFrame.
"""

import pandas as pd
from typing import Optional


def extract_symbol_data(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """
    Extract and filter data for a specific symbol from a DataFrame.

    Args:
        df: DataFrame containing market data with 'symbol' column
        symbol: Stock symbol to filter for

    Returns:
        DataFrame with filtered and sorted data for the symbol, or None if no data found
    """
    try:
        # Filter data for the specific symbol
        symbol_data = df[df['symbol'] == symbol].copy()

        if symbol_data.empty:
            return None

        # Sort by timestamp
        symbol_data = symbol_data.sort_values('timestamp')

        return symbol_data

    except Exception as e:
        print(f"Error extracting symbol data for {symbol}: {e}")
        return None