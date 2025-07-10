"""
Timestamp alignment utilities for matching alerts with market data.

This atom provides functions to align alert timestamps with corresponding market data
for accurate performance analysis.
"""

import pandas as pd
from typing import Tuple, Optional, Dict, Any
import numpy as np


def align_alerts_to_market_data(
    alerts_df: pd.DataFrame,
    market_data_df: pd.DataFrame,
    alert_timestamp_col: str = "timestamp",
    market_timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    tolerance_seconds: int = 60
) -> pd.DataFrame:
    """
    Align alerts with corresponding market data timestamps.
    
    Args:
        alerts_df: DataFrame containing alerts
        market_data_df: DataFrame containing market data
        alert_timestamp_col: Column name for alert timestamps
        market_timestamp_col: Column name for market data timestamps  
        symbol_col: Column name for symbol identification
        tolerance_seconds: Maximum time difference for matching (default: 60 seconds)
        
    Returns:
        DataFrame with alerts matched to market data
        
    Raises:
        KeyError: If required columns not found
        ValueError: If DataFrames are empty or invalid
    """
    if alerts_df.empty or market_data_df.empty:
        return pd.DataFrame()
    
    # Validate required columns
    required_alert_cols = [alert_timestamp_col, symbol_col]
    required_market_cols = [market_timestamp_col, symbol_col]
    
    for col in required_alert_cols:
        if col not in alerts_df.columns:
            raise KeyError(f"Alert column '{col}' not found")
            
    for col in required_market_cols:
        if col not in market_data_df.columns:
            raise KeyError(f"Market data column '{col}' not found")
    
    # Convert timestamps
    alerts_df = alerts_df.copy()
    market_data_df = market_data_df.copy()
    
    alerts_df[alert_timestamp_col] = pd.to_datetime(alerts_df[alert_timestamp_col])
    market_data_df[market_timestamp_col] = pd.to_datetime(market_data_df[market_timestamp_col])
    
    aligned_data = []
    
    for _, alert in alerts_df.iterrows():
        symbol = alert[symbol_col]
        alert_time = alert[alert_timestamp_col]
        
        # Filter market data for the same symbol
        symbol_market_data = market_data_df[market_data_df[symbol_col] == symbol]
        
        if symbol_market_data.empty:
            continue
            
        # Find closest market data timestamp
        time_diffs = np.abs(
            (symbol_market_data[market_timestamp_col] - alert_time).dt.total_seconds()
        )
        
        min_diff_idx = time_diffs.idxmin()
        min_diff = time_diffs.iloc[time_diffs.argmin()]
        
        # Check if within tolerance
        if min_diff <= tolerance_seconds:
            market_row = symbol_market_data.loc[min_diff_idx]
            
            # Combine alert and market data
            aligned_row = alert.to_dict()
            aligned_row['market_timestamp'] = market_row[market_timestamp_col]
            aligned_row['time_diff_seconds'] = min_diff
            
            # Add market data fields with prefix
            for col, value in market_row.items():
                if col not in [market_timestamp_col, symbol_col]:
                    aligned_row[f'market_{col}'] = value
                    
            aligned_data.append(aligned_row)
    
    return pd.DataFrame(aligned_data)


def find_market_data_at_time(
    market_data_df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    symbol: str,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    tolerance_seconds: int = 60
) -> Optional[Dict[str, Any]]:
    """
    Find market data row closest to target timestamp for specific symbol.
    
    Args:
        market_data_df: DataFrame containing market data
        target_timestamp: Target timestamp to find data for
        symbol: Symbol to filter by
        timestamp_col: Column name for timestamps
        symbol_col: Column name for symbol
        tolerance_seconds: Maximum time difference for matching
        
    Returns:
        Dictionary with market data or None if no match found
    """
    if market_data_df.empty:
        return None
        
    # Filter by symbol
    symbol_data = market_data_df[market_data_df[symbol_col] == symbol]
    
    if symbol_data.empty:
        return None
        
    # Convert timestamps
    symbol_data = symbol_data.copy()
    symbol_data[timestamp_col] = pd.to_datetime(symbol_data[timestamp_col])
    target_timestamp = pd.to_datetime(target_timestamp)
    
    # Calculate time differences
    time_diffs = np.abs(
        (symbol_data[timestamp_col] - target_timestamp).dt.total_seconds()
    )
    
    min_diff_idx = time_diffs.idxmin()
    min_diff = time_diffs.iloc[time_diffs.argmin()]
    
    # Check tolerance
    if min_diff <= tolerance_seconds:
        market_row = symbol_data.loc[min_diff_idx]
        result = market_row.to_dict()
        result['time_diff_seconds'] = min_diff
        return result
        
    return None


def validate_timestamp_alignment(
    alerts_df: pd.DataFrame,
    market_data_df: pd.DataFrame,
    alert_timestamp_col: str = "timestamp",
    market_timestamp_col: str = "timestamp"
) -> Dict[str, Any]:
    """
    Validate timestamp ranges and alignment between alerts and market data.
    
    Args:
        alerts_df: DataFrame containing alerts
        market_data_df: DataFrame containing market data
        alert_timestamp_col: Column name for alert timestamps
        market_timestamp_col: Column name for market data timestamps
        
    Returns:
        Dictionary with validation statistics
    """
    if alerts_df.empty or market_data_df.empty:
        return {
            'valid': False,
            'error': 'One or both DataFrames are empty'
        }
    
    try:
        alert_times = pd.to_datetime(alerts_df[alert_timestamp_col])
        market_times = pd.to_datetime(market_data_df[market_timestamp_col])
        
        alert_start = alert_times.min()
        alert_end = alert_times.max()
        market_start = market_times.min()
        market_end = market_times.max()
        
        # Check overlap
        overlap_start = max(alert_start, market_start)
        overlap_end = min(alert_end, market_end)
        has_overlap = overlap_start <= overlap_end
        
        return {
            'valid': True,
            'alert_time_range': (alert_start, alert_end),
            'market_time_range': (market_start, market_end),
            'overlap_range': (overlap_start, overlap_end) if has_overlap else None,
            'has_overlap': has_overlap,
            'alert_count': len(alerts_df),
            'market_data_count': len(market_data_df)
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }