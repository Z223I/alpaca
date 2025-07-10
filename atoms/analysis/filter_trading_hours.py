"""
Trading hours filtering utility for market data analysis.

This atom filters DataFrames to include only regular trading hours (9:30 AM - 4:00 PM ET).
Designed for reusability across different analysis components.
"""

import pandas as pd
from typing import Optional


def filter_trading_hours(
    df: pd.DataFrame, 
    start_time: str = "09:30", 
    end_time: str = "16:00", 
    timezone: str = "US/Eastern",
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Filter DataFrame to include only regular trading hours (9:30 AM - 4:00 PM ET).
    
    Args:
        df: DataFrame with timestamp column
        start_time: Trading day start time (default: "09:30")
        end_time: Trading day end time (default: "16:00") 
        timezone: Target timezone (default: "US/Eastern")
        timestamp_col: Name of timestamp column (default: "timestamp")
    
    Returns:
        Filtered DataFrame containing only regular trading hours data
        
    Raises:
        KeyError: If timestamp column not found in DataFrame
        ValueError: If timestamp conversion fails
    """
    if df.empty:
        return df
    
    if timestamp_col not in df.columns:
        raise KeyError(f"Column '{timestamp_col}' not found in DataFrame")
    
    # Create a copy to avoid modifying original
    filtered_df = df.copy()
    
    try:
        # Convert timestamps to Eastern Time
        filtered_df[timestamp_col] = pd.to_datetime(filtered_df[timestamp_col])
        
        # Handle timezone conversion
        if filtered_df[timestamp_col].dt.tz is None:
            # Assume UTC if no timezone info
            filtered_df[timestamp_col] = filtered_df[timestamp_col].dt.tz_localize('UTC')
        
        filtered_df[timestamp_col] = filtered_df[timestamp_col].dt.tz_convert(timezone)
        
        # Create time filters
        start_time_obj = pd.to_datetime(start_time).time()
        end_time_obj = pd.to_datetime(end_time).time()
        
        # Filter to trading hours
        time_mask = (
            (filtered_df[timestamp_col].dt.time >= start_time_obj) & 
            (filtered_df[timestamp_col].dt.time <= end_time_obj)
        )
        
        # Exclude weekends (Monday=0, Sunday=6)
        weekday_mask = filtered_df[timestamp_col].dt.dayofweek < 5
        
        # Combine filters
        final_mask = time_mask & weekday_mask
        
        return filtered_df[final_mask].reset_index(drop=True)
        
    except Exception as e:
        raise ValueError(f"Error filtering trading hours: {str(e)}")


def is_trading_hours(
    timestamp: pd.Timestamp,
    start_time: str = "09:30",
    end_time: str = "16:00", 
    timezone: str = "US/Eastern"
) -> bool:
    """
    Check if a single timestamp is within trading hours.
    
    Args:
        timestamp: Timestamp to check
        start_time: Trading day start time (default: "09:30")
        end_time: Trading day end time (default: "16:00")
        timezone: Target timezone (default: "US/Eastern")
        
    Returns:
        True if timestamp is within trading hours, False otherwise
    """
    try:
        # Convert to Eastern Time
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        timestamp = timestamp.tz_convert(timezone)
        
        # Check if weekday (Monday=0, Sunday=6)
        if timestamp.dayofweek >= 5:
            return False
            
        # Check time range
        start_time_obj = pd.to_datetime(start_time).time()
        end_time_obj = pd.to_datetime(end_time).time()
        
        return start_time_obj <= timestamp.time() <= end_time_obj
        
    except Exception:
        return False