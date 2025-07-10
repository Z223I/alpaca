"""
Trade execution simulation utilities.

This atom provides functions to simulate trade execution based on
alert parameters and market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


def simulate_trade(
    alert: Dict[str, Any],
    market_data: pd.DataFrame,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    max_duration_hours: int = 24,
    timestamp_col: str = "timestamp"
) -> Dict[str, Any]:
    """
    Simulate a single trade execution based on alert and market data.
    
    Args:
        alert: Dictionary containing alert information
        market_data: DataFrame with market data after alert time
        stop_loss: Stop loss price (overrides alert recommendation)
        take_profit: Take profit price (overrides alert recommendation)
        max_duration_hours: Maximum trade duration before forced exit
        timestamp_col: Name of timestamp column
        
    Returns:
        Dictionary with trade simulation results
    """
    # Extract alert information
    entry_price = alert.get('current_price', 0)
    alert_time = pd.to_datetime(alert.get('timestamp'))
    breakout_type = alert.get('breakout_type', '')
    symbol = alert.get('symbol', '')
    
    # Use provided levels or fall back to alert recommendations
    stop_loss_price = stop_loss or alert.get('recommended_stop_loss', 0)
    take_profit_price = take_profit or alert.get('recommended_take_profit', 0)
    
    # Determine trade direction
    is_long = 'bullish' in breakout_type.lower()
    
    # Initialize trade result
    trade_result = {
        'symbol': symbol,
        'alert_time': alert_time,
        'entry_price': entry_price,
        'stop_loss': stop_loss_price,
        'take_profit': take_profit_price,
        'trade_direction': 'long' if is_long else 'short',
        'breakout_type': breakout_type,
        'exit_price': entry_price,
        'exit_time': alert_time,
        'exit_reason': 'NO_DATA',
        'duration_minutes': 0,
        'return_pct': 0.0,
        'status': 'FAILED',
        'max_favorable_price': entry_price,
        'max_adverse_price': entry_price
    }
    
    # Validate inputs
    if entry_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
        trade_result['exit_reason'] = 'INVALID_PRICES'
        return trade_result
    
    if market_data.empty:
        return trade_result
    
    # Filter market data to after alert time
    market_data = market_data.copy()
    market_data[timestamp_col] = pd.to_datetime(market_data[timestamp_col])
    future_data = market_data[market_data[timestamp_col] > alert_time]
    
    if future_data.empty:
        return trade_result
    
    # Sort by timestamp
    future_data = future_data.sort_values(timestamp_col)
    
    # Calculate expiration time
    expiration_time = alert_time + timedelta(hours=max_duration_hours)
    
    # Track price extremes
    max_favorable = entry_price
    max_adverse = entry_price
    
    # Simulate trade execution
    for _, row in future_data.iterrows():
        current_time = row[timestamp_col]
        
        # Check if trade expired
        if current_time > expiration_time:
            trade_result.update({
                'exit_price': row.get('close', entry_price),
                'exit_time': current_time,
                'exit_reason': 'EXPIRED',
                'duration_minutes': (current_time - alert_time).total_seconds() / 60
            })
            break
        
        # Get OHLC data
        open_price = row.get('open', entry_price)
        high_price = row.get('high', entry_price)
        low_price = row.get('low', entry_price)
        close_price = row.get('close', entry_price)
        
        # Update price extremes
        if is_long:
            max_favorable = max(max_favorable, high_price)
            max_adverse = min(max_adverse, low_price)
        else:
            max_favorable = min(max_favorable, low_price)
            max_adverse = max(max_adverse, high_price)
        
        # Check for stop loss hit
        if is_long and low_price <= stop_loss_price:
            trade_result.update({
                'exit_price': stop_loss_price,
                'exit_time': current_time,
                'exit_reason': 'STOP_LOSS',
                'duration_minutes': (current_time - alert_time).total_seconds() / 60,
                'status': 'STOPPED_OUT'
            })
            break
        elif not is_long and high_price >= stop_loss_price:
            trade_result.update({
                'exit_price': stop_loss_price,
                'exit_time': current_time,
                'exit_reason': 'STOP_LOSS',
                'duration_minutes': (current_time - alert_time).total_seconds() / 60,
                'status': 'STOPPED_OUT'
            })
            break
        
        # Check for take profit hit
        if is_long and high_price >= take_profit_price:
            trade_result.update({
                'exit_price': take_profit_price,
                'exit_time': current_time,
                'exit_reason': 'TAKE_PROFIT',
                'duration_minutes': (current_time - alert_time).total_seconds() / 60,
                'status': 'SUCCESS'
            })
            break
        elif not is_long and low_price <= take_profit_price:
            trade_result.update({
                'exit_price': take_profit_price,
                'exit_time': current_time,
                'exit_reason': 'TAKE_PROFIT',
                'duration_minutes': (current_time - alert_time).total_seconds() / 60,
                'status': 'SUCCESS'
            })
            break
    else:
        # Trade still open at end of data
        last_row = future_data.iloc[-1]
        trade_result.update({
            'exit_price': last_row.get('close', entry_price),
            'exit_time': last_row[timestamp_col],
            'exit_reason': 'END_OF_DATA',
            'duration_minutes': (last_row[timestamp_col] - alert_time).total_seconds() / 60
        })
    
    # Calculate final metrics
    trade_result['max_favorable_price'] = max_favorable
    trade_result['max_adverse_price'] = max_adverse
    
    # Calculate return
    if is_long:
        trade_result['return_pct'] = ((trade_result['exit_price'] - entry_price) / entry_price) * 100
    else:
        trade_result['return_pct'] = ((entry_price - trade_result['exit_price']) / entry_price) * 100
    
    # Update status if not already set
    if trade_result['status'] not in ['SUCCESS', 'STOPPED_OUT']:
        if trade_result['return_pct'] > 0:
            trade_result['status'] = 'PROFIT'
        elif trade_result['return_pct'] < 0:
            trade_result['status'] = 'LOSS'
        else:
            trade_result['status'] = 'BREAKEVEN'
    
    return trade_result


def simulate_multiple_trades(
    alerts_df: pd.DataFrame,
    market_data_df: pd.DataFrame,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Simulate multiple trades from alerts DataFrame.
    
    Args:
        alerts_df: DataFrame containing alerts
        market_data_df: DataFrame containing market data
        symbol_col: Column name for symbol
        timestamp_col: Column name for timestamp
        
    Returns:
        DataFrame with simulation results for all trades
    """
    if alerts_df.empty or market_data_df.empty:
        return pd.DataFrame()
    
    results = []
    
    for _, alert in alerts_df.iterrows():
        symbol = alert[symbol_col]
        
        # Filter market data for this symbol
        symbol_market_data = market_data_df[market_data_df[symbol_col] == symbol]
        
        if symbol_market_data.empty:
            continue
        
        # Simulate trade
        trade_result = simulate_trade(alert.to_dict(), symbol_market_data, timestamp_col=timestamp_col)
        results.append(trade_result)
    
    return pd.DataFrame(results)


def calculate_trade_metrics(trade_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate additional metrics for a single trade result.
    
    Args:
        trade_result: Dictionary containing trade simulation results
        
    Returns:
        Dictionary with additional calculated metrics
    """
    entry_price = trade_result.get('entry_price', 0)
    exit_price = trade_result.get('exit_price', 0)
    max_favorable = trade_result.get('max_favorable_price', entry_price)
    max_adverse = trade_result.get('max_adverse_price', entry_price)
    is_long = trade_result.get('trade_direction', 'long') == 'long'
    
    metrics = trade_result.copy()
    
    # Maximum Favorable Excursion (MFE)
    if is_long:
        metrics['mfe_pct'] = ((max_favorable - entry_price) / entry_price) * 100
        metrics['mae_pct'] = ((max_adverse - entry_price) / entry_price) * 100
    else:
        metrics['mfe_pct'] = ((entry_price - max_favorable) / entry_price) * 100
        metrics['mae_pct'] = ((entry_price - max_adverse) / entry_price) * 100
    
    # Risk/Reward ratio
    stop_loss = trade_result.get('stop_loss', entry_price)
    take_profit = trade_result.get('take_profit', entry_price)
    
    if is_long:
        potential_loss = abs(entry_price - stop_loss)
        potential_gain = abs(take_profit - entry_price)
    else:
        potential_loss = abs(stop_loss - entry_price)
        potential_gain = abs(entry_price - take_profit)
    
    if potential_loss > 0:
        metrics['risk_reward_ratio'] = potential_gain / potential_loss
    else:
        metrics['risk_reward_ratio'] = float('inf')
    
    # Trade efficiency (actual return vs maximum favorable)
    if metrics['mfe_pct'] != 0:
        metrics['trade_efficiency'] = metrics['return_pct'] / metrics['mfe_pct']
    else:
        metrics['trade_efficiency'] = 0.0
    
    return metrics


def simulate_trade_with_slippage(
    alert: Dict[str, Any],
    market_data: pd.DataFrame,
    entry_slippage_pct: float = 0.05,
    exit_slippage_pct: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """
    Simulate trade with slippage applied to entry and exit prices.
    
    Args:
        alert: Dictionary containing alert information
        market_data: DataFrame with market data
        entry_slippage_pct: Slippage percentage on entry (negative for adverse)
        exit_slippage_pct: Slippage percentage on exit (negative for adverse)
        **kwargs: Additional arguments passed to simulate_trade
        
    Returns:
        Dictionary with trade simulation results including slippage
    """
    # Apply entry slippage
    original_price = alert.get('current_price', 0)
    breakout_type = alert.get('breakout_type', '')
    is_long = 'bullish' in breakout_type.lower()
    
    # Apply adverse slippage (worse fill)
    if is_long:
        adjusted_entry = original_price * (1 + entry_slippage_pct / 100)
    else:
        adjusted_entry = original_price * (1 - entry_slippage_pct / 100)
    
    # Update alert with slipped entry price
    slipped_alert = alert.copy()
    slipped_alert['current_price'] = adjusted_entry
    
    # Simulate trade
    result = simulate_trade(slipped_alert, market_data, **kwargs)
    
    # Apply exit slippage if trade was executed
    if result['status'] in ['SUCCESS', 'STOPPED_OUT', 'PROFIT', 'LOSS']:
        exit_price = result['exit_price']
        
        # Apply adverse slippage on exit
        if is_long:
            slipped_exit = exit_price * (1 - exit_slippage_pct / 100)
        else:
            slipped_exit = exit_price * (1 + exit_slippage_pct / 100)
        
        result['exit_price'] = slipped_exit
        result['original_entry_price'] = original_price
        result['entry_slippage_pct'] = entry_slippage_pct
        result['exit_slippage_pct'] = exit_slippage_pct
        
        # Recalculate return with slippage
        if is_long:
            result['return_pct'] = ((slipped_exit - adjusted_entry) / adjusted_entry) * 100
        else:
            result['return_pct'] = ((adjusted_entry - slipped_exit) / adjusted_entry) * 100
    
    return result