"""
Return calculation utilities for trading performance analysis.

This atom provides functions to calculate various types of returns
from trading data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


def calculate_simple_return(
    entry_price: Union[float, pd.Series],
    exit_price: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Calculate simple return percentage.
    
    Args:
        entry_price: Entry price(s)
        exit_price: Exit price(s)
        
    Returns:
        Simple return percentage(s)
    """
    return ((exit_price - entry_price) / entry_price) * 100


def calculate_log_return(
    entry_price: Union[float, pd.Series],
    exit_price: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Calculate logarithmic return.
    
    Args:
        entry_price: Entry price(s)
        exit_price: Exit price(s)
        
    Returns:
        Log return(s)
    """
    return np.log(exit_price / entry_price) * 100


def calculate_trade_returns(
    trades_df: pd.DataFrame,
    entry_col: str = "entry_price",
    exit_col: str = "exit_price",
    return_type: str = "simple"
) -> pd.Series:
    """
    Calculate returns for a DataFrame of trades.
    
    Args:
        trades_df: DataFrame containing trade data
        entry_col: Column name for entry prices
        exit_col: Column name for exit prices
        return_type: Type of return calculation ('simple' or 'log')
        
    Returns:
        Series of calculated returns
    """
    if trades_df.empty:
        return pd.Series(dtype=float)
    
    if entry_col not in trades_df.columns or exit_col not in trades_df.columns:
        raise KeyError(f"Required columns '{entry_col}' or '{exit_col}' not found")
    
    entry_prices = trades_df[entry_col]
    exit_prices = trades_df[exit_col]
    
    if return_type == "simple":
        return calculate_simple_return(entry_prices, exit_prices)
    elif return_type == "log":
        return calculate_log_return(entry_prices, exit_prices)
    else:
        raise ValueError(f"Invalid return_type '{return_type}'. Use 'simple' or 'log'")


def calculate_cumulative_returns(
    returns: pd.Series,
    compound: bool = True
) -> pd.Series:
    """
    Calculate cumulative returns from a series of returns.
    
    Args:
        returns: Series of period returns (in percentage)
        compound: If True, use compounding; if False, use simple addition
        
    Returns:
        Series of cumulative returns
    """
    if returns.empty:
        return pd.Series(dtype=float)
    
    if compound:
        # Convert percentage to decimal, add 1, calculate cumulative product, subtract 1
        return ((1 + returns / 100).cumprod() - 1) * 100
    else:
        # Simple cumulative sum
        return returns.cumsum()


def calculate_annualized_return(
    total_return: float,
    days: int
) -> float:
    """
    Calculate annualized return from total return and time period.
    
    Args:
        total_return: Total return percentage
        days: Number of days in the period
        
    Returns:
        Annualized return percentage
    """
    if days <= 0:
        return 0.0
    
    years = days / 365.25
    return ((1 + total_return / 100) ** (1 / years) - 1) * 100


def calculate_maximum_favorable_excursion(
    trades_df: pd.DataFrame,
    entry_col: str = "entry_price",
    high_col: str = "max_price_reached",
    trade_type_col: str = "trade_type"
) -> pd.Series:
    """
    Calculate Maximum Favorable Excursion (MFE) for trades.
    
    Args:
        trades_df: DataFrame containing trade data
        entry_col: Column name for entry prices
        high_col: Column name for maximum price reached during trade
        trade_type_col: Column indicating trade direction ('long' or 'short')
        
    Returns:
        Series of MFE values (percentage)
    """
    if trades_df.empty:
        return pd.Series(dtype=float)
    
    entry_prices = trades_df[entry_col]
    max_prices = trades_df[high_col]
    
    if trade_type_col in trades_df.columns:
        # Handle long and short trades differently
        long_mask = trades_df[trade_type_col].str.lower() == 'long'
        short_mask = trades_df[trade_type_col].str.lower() == 'short'
        
        mfe = pd.Series(index=trades_df.index, dtype=float)
        
        # For long trades: MFE = (max_price - entry_price) / entry_price
        mfe[long_mask] = calculate_simple_return(
            entry_prices[long_mask], 
            max_prices[long_mask]
        )
        
        # For short trades: MFE = (entry_price - min_price) / entry_price
        # Assuming high_col actually contains min_price for short trades
        mfe[short_mask] = calculate_simple_return(
            max_prices[short_mask],
            entry_prices[short_mask] 
        )
        
        return mfe
    else:
        # Assume all long trades
        return calculate_simple_return(entry_prices, max_prices)


def calculate_maximum_adverse_excursion(
    trades_df: pd.DataFrame,
    entry_col: str = "entry_price", 
    low_col: str = "min_price_reached",
    trade_type_col: str = "trade_type"
) -> pd.Series:
    """
    Calculate Maximum Adverse Excursion (MAE) for trades.
    
    Args:
        trades_df: DataFrame containing trade data
        entry_col: Column name for entry prices
        low_col: Column name for minimum price reached during trade
        trade_type_col: Column indicating trade direction ('long' or 'short')
        
    Returns:
        Series of MAE values (percentage, negative values indicate adverse movement)
    """
    if trades_df.empty:
        return pd.Series(dtype=float)
    
    entry_prices = trades_df[entry_col]
    min_prices = trades_df[low_col]
    
    if trade_type_col in trades_df.columns:
        # Handle long and short trades differently
        long_mask = trades_df[trade_type_col].str.lower() == 'long'
        short_mask = trades_df[trade_type_col].str.lower() == 'short'
        
        mae = pd.Series(index=trades_df.index, dtype=float)
        
        # For long trades: MAE = (min_price - entry_price) / entry_price
        mae[long_mask] = calculate_simple_return(
            entry_prices[long_mask],
            min_prices[long_mask]
        )
        
        # For short trades: MAE = (entry_price - max_price) / entry_price  
        # Assuming low_col actually contains max_price for short trades
        mae[short_mask] = calculate_simple_return(
            min_prices[short_mask],
            entry_prices[short_mask]
        )
        
        return mae
    else:
        # Assume all long trades
        return calculate_simple_return(entry_prices, min_prices)


def calculate_risk_adjusted_return(
    returns: pd.Series,
    benchmark_return: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate risk-adjusted return metrics.
    
    Args:
        returns: Series of period returns (percentage)
        benchmark_return: Risk-free rate or benchmark return (annual percentage)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Dictionary with risk-adjusted metrics
    """
    if returns.empty:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }
    
    # Convert to decimal for calculations
    returns_decimal = returns / 100
    benchmark_decimal = benchmark_return / 100
    
    # Calculate excess returns
    excess_returns = returns_decimal - (benchmark_decimal / periods_per_year)
    
    # Sharpe Ratio
    if returns_decimal.std() != 0:
        sharpe_ratio = (excess_returns.mean() * periods_per_year) / (returns_decimal.std() * np.sqrt(periods_per_year))
    else:
        sharpe_ratio = 0.0
    
    # Sortino Ratio (using downside deviation)
    downside_returns = returns_decimal[returns_decimal < benchmark_decimal / periods_per_year]
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino_ratio = (excess_returns.mean() * periods_per_year) / (downside_returns.std() * np.sqrt(periods_per_year))
    else:
        sortino_ratio = 0.0
    
    # Calmar Ratio (annual return / max drawdown)
    cumulative_returns = calculate_cumulative_returns(returns, compound=True)
    max_drawdown = calculate_maximum_drawdown(cumulative_returns)
    
    annual_return = (excess_returns.mean() * periods_per_year) * 100
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio, 
        'calmar_ratio': calmar_ratio
    }


def calculate_maximum_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from cumulative returns.
    
    Args:
        cumulative_returns: Series of cumulative returns (percentage)
        
    Returns:
        Maximum drawdown percentage (negative value)
    """
    if cumulative_returns.empty:
        return 0.0
    
    # Convert to wealth index (assuming starting value of 100)
    wealth_index = 100 * (1 + cumulative_returns / 100)
    
    # Calculate running maximum
    running_max = wealth_index.expanding().max()
    
    # Calculate drawdown
    drawdown = (wealth_index - running_max) / running_max * 100
    
    return drawdown.min()  # Most negative value