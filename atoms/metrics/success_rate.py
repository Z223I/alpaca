"""
Success rate calculation utilities for alert performance analysis.

This atom provides functions to calculate success rates and related metrics
for trading alerts and strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def calculate_success_rate(
    trades_df: pd.DataFrame,
    status_col: str = "status",
    success_values: Optional[List[str]] = None
) -> float:
    """
    Calculate success rate from trades DataFrame.
    
    Args:
        trades_df: DataFrame containing trade results
        status_col: Column name containing trade status
        success_values: List of values considered successful (default: ['SUCCESS', 'PROFIT'])
        
    Returns:
        Success rate as a percentage (0-100)
    """
    if trades_df.empty:
        return 0.0
    
    if status_col not in trades_df.columns:
        raise KeyError(f"Status column '{status_col}' not found in DataFrame")
    
    if success_values is None:
        success_values = ['SUCCESS', 'PROFIT', 'WIN']
    
    total_trades = len(trades_df)
    successful_trades = trades_df[status_col].isin(success_values).sum()
    
    return (successful_trades / total_trades) * 100


def calculate_success_rate_by_group(
    trades_df: pd.DataFrame,
    group_col: str,
    status_col: str = "status",
    success_values: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate success rates grouped by a specific column.
    
    Args:
        trades_df: DataFrame containing trade results
        group_col: Column to group by (e.g., 'symbol', 'priority')
        status_col: Column name containing trade status
        success_values: List of values considered successful
        
    Returns:
        DataFrame with success rates by group
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    if group_col not in trades_df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame")
    
    if success_values is None:
        success_values = ['SUCCESS', 'PROFIT', 'WIN']
    
    def group_success_rate(group):
        total = len(group)
        successful = group[status_col].isin(success_values).sum()
        return pd.Series({
            'total_trades': total,
            'successful_trades': successful,
            'success_rate': (successful / total) * 100 if total > 0 else 0.0
        })
    
    result = trades_df.groupby(group_col).apply(group_success_rate).reset_index()
    return result.sort_values('success_rate', ascending=False)


def calculate_win_loss_ratio(
    trades_df: pd.DataFrame,
    return_col: str = "return_pct",
    count_based: bool = False
) -> Dict[str, float]:
    """
    Calculate win/loss ratio for trades.
    
    Args:
        trades_df: DataFrame containing trade results
        return_col: Column name containing returns
        count_based: If True, calculate ratio based on trade counts; 
                    if False, based on average returns
        
    Returns:
        Dictionary with win/loss metrics
    """
    if trades_df.empty or return_col not in trades_df.columns:
        return {
            'win_loss_ratio': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'average_win': 0.0,
            'average_loss': 0.0
        }
    
    # Separate wins and losses
    wins = trades_df[trades_df[return_col] > 0]
    losses = trades_df[trades_df[return_col] < 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    
    average_win = wins[return_col].mean() if not wins.empty else 0.0
    average_loss = abs(losses[return_col].mean()) if not losses.empty else 0.0
    
    if count_based:
        win_loss_ratio = win_count / loss_count if loss_count > 0 else float('inf')
    else:
        win_loss_ratio = average_win / average_loss if average_loss > 0 else float('inf')
    
    return {
        'win_loss_ratio': win_loss_ratio,
        'win_count': win_count,
        'loss_count': loss_count,
        'average_win': average_win,
        'average_loss': average_loss
    }


def calculate_profit_factor(
    trades_df: pd.DataFrame,
    return_col: str = "return_pct"
) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades_df: DataFrame containing trade results
        return_col: Column name containing returns
        
    Returns:
        Profit factor
    """
    if trades_df.empty or return_col not in trades_df.columns:
        return 0.0
    
    gross_profit = trades_df[trades_df[return_col] > 0][return_col].sum()
    gross_loss = abs(trades_df[trades_df[return_col] < 0][return_col].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_hit_rate_by_timeframe(
    trades_df: pd.DataFrame,
    duration_col: str = "duration_minutes",
    status_col: str = "status",
    timeframes: Optional[List[int]] = None,
    success_values: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate hit rates by different timeframes.
    
    Args:
        trades_df: DataFrame containing trade results
        duration_col: Column name containing trade duration in minutes
        status_col: Column name containing trade status
        timeframes: List of timeframe thresholds in minutes
        success_values: List of values considered successful
        
    Returns:
        DataFrame with hit rates by timeframe
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    if timeframes is None:
        timeframes = [5, 15, 30, 60, 120, 240]  # 5min to 4 hours
    
    if success_values is None:
        success_values = ['SUCCESS', 'PROFIT', 'WIN']
    
    results = []
    
    for timeframe in timeframes:
        timeframe_trades = trades_df[trades_df[duration_col] <= timeframe]
        
        if not timeframe_trades.empty:
            total = len(timeframe_trades)
            successful = timeframe_trades[status_col].isin(success_values).sum()
            hit_rate = (successful / total) * 100
            
            results.append({
                'timeframe_minutes': timeframe,
                'total_trades': total,
                'successful_trades': successful,
                'hit_rate': hit_rate
            })
    
    return pd.DataFrame(results)


def calculate_alert_performance_summary(
    trades_df: pd.DataFrame,
    return_col: str = "return_pct",
    status_col: str = "status",
    confidence_col: str = "confidence_score",
    priority_col: str = "priority"
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance summary for alerts.
    
    Args:
        trades_df: DataFrame containing trade results
        return_col: Column name containing returns
        status_col: Column name containing trade status
        confidence_col: Column name containing confidence scores
        priority_col: Column name containing priority levels
        
    Returns:
        Dictionary with comprehensive performance metrics
    """
    if trades_df.empty:
        return {
            'total_alerts': 0,
            'success_rate': 0.0,
            'average_return': 0.0,
            'profit_factor': 0.0,
            'win_loss_ratio': 0.0
        }
    
    summary = {
        'total_alerts': len(trades_df),
        'success_rate': calculate_success_rate(trades_df, status_col),
        'profit_factor': calculate_profit_factor(trades_df, return_col)
    }
    
    # Average return
    if return_col in trades_df.columns:
        summary['average_return'] = trades_df[return_col].mean()
        summary['median_return'] = trades_df[return_col].median()
        summary['return_std'] = trades_df[return_col].std()
    
    # Win/Loss metrics
    win_loss_metrics = calculate_win_loss_ratio(trades_df, return_col)
    summary.update(win_loss_metrics)
    
    # Performance by priority
    if priority_col in trades_df.columns:
        priority_performance = calculate_success_rate_by_group(trades_df, priority_col, status_col)
        summary['priority_performance'] = priority_performance.to_dict('records')
    
    # Confidence analysis
    if confidence_col in trades_df.columns:
        summary['average_confidence'] = trades_df[confidence_col].mean()
        summary['confidence_correlation'] = trades_df[confidence_col].corr(
            trades_df[return_col]
        ) if return_col in trades_df.columns else None
    
    return summary