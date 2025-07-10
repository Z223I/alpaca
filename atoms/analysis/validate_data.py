"""
Data quality validation utilities for alert analysis.

This atom provides functions to validate data quality and identify issues
in market data and alert data.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np


def validate_market_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    timestamp_col: str = "timestamp"
) -> Dict[str, Any]:
    """
    Validate market data quality and completeness.
    
    Args:
        df: DataFrame containing market data
        required_columns: List of required columns to check
        timestamp_col: Name of timestamp column
        
    Returns:
        Dictionary with validation results
    """
    if required_columns is None:
        required_columns = [timestamp_col, "symbol", "open", "high", "low", "close", "volume"]
    
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_result['valid'] = False
        validation_result['errors'].append('DataFrame is empty')
        return validation_result
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_result['valid'] = False
        validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Validate timestamps
    if timestamp_col in df.columns:
        try:
            timestamps = pd.to_datetime(df[timestamp_col])
            
            # Check for null timestamps
            null_timestamps = timestamps.isna().sum()
            if null_timestamps > 0:
                validation_result['warnings'].append(f'{null_timestamps} null timestamps found')
            
            # Check for duplicate timestamps per symbol
            if 'symbol' in df.columns:
                duplicates = df.groupby('symbol')[timestamp_col].apply(
                    lambda x: x.duplicated().sum()
                ).sum()
                if duplicates > 0:
                    validation_result['warnings'].append(f'{duplicates} duplicate timestamps found')
            
            # Check chronological order
            if not timestamps.is_monotonic_increasing:
                validation_result['warnings'].append('Timestamps are not in chronological order')
                
            validation_result['statistics']['timestamp_range'] = (timestamps.min(), timestamps.max())
            validation_result['statistics']['total_records'] = len(df)
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Timestamp validation error: {str(e)}')
    
    # Validate price data
    price_columns = ['open', 'high', 'low', 'close']
    available_price_cols = [col for col in price_columns if col in df.columns]
    
    if available_price_cols:
        for col in available_price_cols:
            # Check for negative prices
            negative_prices = (df[col] < 0).sum()
            if negative_prices > 0:
                validation_result['warnings'].append(f'{negative_prices} negative prices in {col}')
            
            # Check for zero prices
            zero_prices = (df[col] == 0).sum()
            if zero_prices > 0:
                validation_result['warnings'].append(f'{zero_prices} zero prices in {col}')
            
            # Check for null prices
            null_prices = df[col].isna().sum()
            if null_prices > 0:
                validation_result['warnings'].append(f'{null_prices} null prices in {col}')
    
    # Validate OHLC relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= Open, Low, Close
        invalid_high = ((df['high'] < df['open']) | 
                       (df['high'] < df['low']) | 
                       (df['high'] < df['close'])).sum()
        if invalid_high > 0:
            validation_result['warnings'].append(f'{invalid_high} records with invalid high prices')
        
        # Low should be <= Open, High, Close
        invalid_low = ((df['low'] > df['open']) | 
                      (df['low'] > df['high']) | 
                      (df['low'] > df['close'])).sum()
        if invalid_low > 0:
            validation_result['warnings'].append(f'{invalid_low} records with invalid low prices')
    
    # Validate volume
    if 'volume' in df.columns:
        negative_volume = (df['volume'] < 0).sum()
        if negative_volume > 0:
            validation_result['warnings'].append(f'{negative_volume} negative volume values')
        
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            validation_result['statistics']['zero_volume_count'] = zero_volume
    
    return validation_result


def validate_alert_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    timestamp_col: str = "timestamp"
) -> Dict[str, Any]:
    """
    Validate alert data quality and completeness.
    
    Args:
        df: DataFrame containing alert data
        required_columns: List of required columns to check
        timestamp_col: Name of timestamp column
        
    Returns:
        Dictionary with validation results
    """
    if required_columns is None:
        required_columns = [
            timestamp_col, "symbol", "current_price", "breakout_type", 
            "priority", "confidence_score", "recommended_stop_loss", "recommended_take_profit"
        ]
    
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_result['valid'] = False
        validation_result['errors'].append('DataFrame is empty')
        return validation_result
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_result['valid'] = False
        validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Validate timestamps
    if timestamp_col in df.columns:
        try:
            timestamps = pd.to_datetime(df[timestamp_col])
            
            null_timestamps = timestamps.isna().sum()
            if null_timestamps > 0:
                validation_result['warnings'].append(f'{null_timestamps} null timestamps found')
                
            validation_result['statistics']['timestamp_range'] = (timestamps.min(), timestamps.max())
            validation_result['statistics']['total_alerts'] = len(df)
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Timestamp validation error: {str(e)}')
    
    # Validate price fields
    price_fields = ['current_price', 'recommended_stop_loss', 'recommended_take_profit']
    for field in price_fields:
        if field in df.columns:
            # Check for negative or zero prices
            invalid_prices = (df[field] <= 0).sum()
            if invalid_prices > 0:
                validation_result['warnings'].append(f'{invalid_prices} invalid prices in {field}')
            
            # Check for null prices
            null_prices = df[field].isna().sum()
            if null_prices > 0:
                validation_result['warnings'].append(f'{null_prices} null prices in {field}')
    
    # Validate confidence scores
    if 'confidence_score' in df.columns:
        invalid_confidence = ((df['confidence_score'] < 0) | (df['confidence_score'] > 1)).sum()
        if invalid_confidence > 0:
            validation_result['warnings'].append(f'{invalid_confidence} invalid confidence scores (should be 0-1)')
    
    # Validate breakout types
    if 'breakout_type' in df.columns:
        valid_breakout_types = ['bullish_breakout', 'bearish_breakdown', 'bullish', 'bearish']
        invalid_breakout = ~df['breakout_type'].isin(valid_breakout_types)
        if invalid_breakout.sum() > 0:
            validation_result['warnings'].append(f'{invalid_breakout.sum()} invalid breakout types')
    
    # Validate priority levels
    if 'priority' in df.columns:
        valid_priorities = ['HIGH', 'MEDIUM', 'LOW', 'High', 'Medium', 'Low']
        invalid_priority = ~df['priority'].isin(valid_priorities)
        if invalid_priority.sum() > 0:
            validation_result['warnings'].append(f'{invalid_priority.sum()} invalid priority levels')
    
    # Statistics
    if 'symbol' in df.columns:
        validation_result['statistics']['unique_symbols'] = df['symbol'].nunique()
        validation_result['statistics']['symbol_counts'] = df['symbol'].value_counts().to_dict()
    
    return validation_result


def identify_data_gaps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_frequency: str = "1min",
    symbol_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identify gaps in time series data.
    
    Args:
        df: DataFrame with time series data
        timestamp_col: Name of timestamp column
        expected_frequency: Expected data frequency (e.g., '1min', '5min')
        symbol_col: Name of symbol column (optional)
        
    Returns:
        Dictionary with gap analysis results
    """
    if df.empty:
        return {'gaps_found': False, 'message': 'DataFrame is empty'}
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    gap_results = {
        'gaps_found': False,
        'gaps': [],
        'statistics': {}
    }
    
    if symbol_col and symbol_col in df.columns:
        # Analyze gaps per symbol
        for symbol in df[symbol_col].unique():
            symbol_df = df[df[symbol_col] == symbol].sort_values(timestamp_col)
            symbol_gaps = _find_time_gaps(symbol_df[timestamp_col], expected_frequency)
            
            if symbol_gaps:
                gap_results['gaps_found'] = True
                gap_results['gaps'].extend([
                    {'symbol': symbol, **gap} for gap in symbol_gaps
                ])
    else:
        # Analyze gaps for entire dataset
        df_sorted = df.sort_values(timestamp_col)
        gaps = _find_time_gaps(df_sorted[timestamp_col], expected_frequency)
        
        if gaps:
            gap_results['gaps_found'] = True
            gap_results['gaps'] = gaps
    
    gap_results['statistics']['total_gaps'] = len(gap_results['gaps'])
    
    return gap_results


def _find_time_gaps(timestamps: pd.Series, expected_frequency: str) -> List[Dict[str, Any]]:
    """
    Helper function to find gaps in a timestamp series.
    
    Args:
        timestamps: Series of timestamps
        expected_frequency: Expected frequency
        
    Returns:
        List of gap dictionaries
    """
    if len(timestamps) < 2:
        return []
    
    gaps = []
    freq_delta = pd.Timedelta(expected_frequency)
    
    for i in range(1, len(timestamps)):
        time_diff = timestamps.iloc[i] - timestamps.iloc[i-1]
        
        if time_diff > freq_delta * 1.5:  # Allow some tolerance
            gaps.append({
                'start_time': timestamps.iloc[i-1],
                'end_time': timestamps.iloc[i],
                'duration': time_diff,
                'missing_periods': int(time_diff / freq_delta) - 1
            })
    
    return gaps