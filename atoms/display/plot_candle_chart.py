"""
Atom for plotting candlestick charts with volume for stock market data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Optional, Tuple
from datetime import time


def _calculate_orb_levels(symbol_data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate Open Range Breakout (ORB) high and low levels from the first 15 candlesticks.
    
    Args:
        symbol_data: DataFrame with timestamp, open, high, low, close, volume columns
        
    Returns:
        Tuple of (orb_high, orb_low) or (None, None) if insufficient data
    """
    isDebugging = False  # Set to True for debugging output
    
    if isDebugging:
        print("=== ORB Calculation Debug ===")
        print(f"Input data shape: {symbol_data.shape}")
        print(f"Input data columns: {list(symbol_data.columns)}")
    
    if symbol_data.empty:
        if isDebugging:
            print("DEBUG: No data available for ORB calculation.")
        return None, None
    
    try:
        # Sort by timestamp to ensure chronological order
        symbol_data = symbol_data.copy().sort_values('timestamp')
        
        if isDebugging:
            print(f"DEBUG: Total candlesticks available: {len(symbol_data)}")
            print(f"DEBUG: First timestamp: {symbol_data['timestamp'].iloc[0]}")
            print(f"DEBUG: Last timestamp: {symbol_data['timestamp'].iloc[-1]}")
        
        # Use first 15 candlesticks for ORB calculation
        orb_candlesticks = 15
        if len(symbol_data) < orb_candlesticks:
            if isDebugging:
                print(f"DEBUG: Insufficient data - only {len(symbol_data)} candlesticks available, need {orb_candlesticks}")
            # Use all available data if less than 15 candlesticks
            orb_data = symbol_data
            orb_candlesticks = len(symbol_data)
        else:
            orb_data = symbol_data.head(orb_candlesticks)
        
        if isDebugging:
            print(f"DEBUG: Using first {orb_candlesticks} candlesticks for ORB")
            print(f"DEBUG: ORB period timestamps: {orb_data['timestamp'].iloc[0]} to {orb_data['timestamp'].iloc[-1]}")
            print(f"DEBUG: ORB data high values: {orb_data['high'].tolist()}")
            print(f"DEBUG: ORB data low values: {orb_data['low'].tolist()}")
            
        # Calculate ORB high and low from the first N candlesticks
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()

        if isDebugging:
            print(f"DEBUG: Calculated orb_high: {orb_high}, orb_low: {orb_low}")

        if orb_high is None or orb_low is None:
            if isDebugging:
                print("DEBUG: ORB levels could not be calculated - no valid high/low data.")
            return None, None
        
        if isDebugging:
            print("=== ORB Calculation Complete ===")
        
        return orb_high, orb_low
        
    except Exception as e:
        if isDebugging:
            print(f"DEBUG: Exception in ORB calculation: {e}")
            import traceback
            traceback.print_exc()
        return None, None


def plot_candle_chart(df: pd.DataFrame, symbol: str, output_dir: str = 'plots') -> bool:
    """
    Create a candlestick chart with volume and ORB rectangle for a single stock symbol.
    
    Args:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        symbol: Stock symbol name
        output_dir: Directory to save the chart
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Filter data for the specific symbol
        symbol_data = df[df['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            print(f"No data found for symbol: {symbol}")
            return False
        
        # Sort by timestamp
        symbol_data = symbol_data.sort_values('timestamp')
        
        # Calculate ORB levels
        orb_high, orb_low = _calculate_orb_levels(symbol_data)
        
        # Create figure with subplots (price and volume)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
        
        # Plot candlesticks on top subplot
        for idx, row in symbol_data.iterrows():
            timestamp = row['timestamp']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Determine color (green for up, red for down)
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw the high-low line
            ax1.plot([timestamp, timestamp], [low_price, high_price], 
                    color='black', linewidth=1, zorder=1)
            
            # Draw the open-close rectangle
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            
            rect = Rectangle((mdates.date2num(timestamp) - 0.0003, bottom), 
                           0.0006, height, 
                           facecolor=color, edgecolor='none', alpha=0.8,
                           zorder=2)
            ax1.add_patch(rect)
        
        # Add ORB rectangle if ORB levels were calculated
        if orb_high is not None and orb_low is not None:
            # Get time range for ORB rectangle (first 15 candlesticks or all available data)
            x_min = mdates.date2num(symbol_data['timestamp'].min())
            
            # Calculate width to cover first 15 candlesticks (or all if less than 15)
            orb_candlesticks = min(15, len(symbol_data))
            if orb_candlesticks > 1:
                x_max = mdates.date2num(symbol_data['timestamp'].iloc[orb_candlesticks - 1])
            else:
                x_max = mdates.date2num(symbol_data['timestamp'].iloc[0])
            
            orb_width = x_max - x_min + 0.0012  # Add small padding to cover the last candlestick
            
            # Draw ORB rectangle (filled with thinner edge)
            orb_rect = Rectangle((x_min - 0.0006, orb_low), orb_width, orb_high - orb_low,
                               facecolor='yellow', edgecolor='red', alpha=0.6, linewidth=3)
            ax1.add_patch(orb_rect)
            
            # Add ORB level lines
            ax1.axhline(y=orb_high, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'ORB High: ${orb_high:.2f}')
            ax1.axhline(y=orb_low, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'ORB Low: ${orb_low:.2f}')
            
            # Add legend for ORB levels
            ax1.legend(loc='upper left', fontsize=10)
            
            print(f"ORB levels for {symbol}: High=${orb_high:.2f}, Low=${orb_low:.2f}")
        
        # Format price chart
        title = f'{symbol} - Candlestick Chart with Volume'
        if orb_high is not None and orb_low is not None:
            title += f' (ORB: ${orb_low:.2f}-${orb_high:.2f})'
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Plot volume on bottom subplot
        ax2.bar(symbol_data['timestamp'], symbol_data['volume'], 
               color='blue', alpha=0.6, width=0.0008)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the chart
        filename = f"{symbol}_candle_chart.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Close the plot to free memory
        plt.close(fig)
        
        print(f"Chart saved: {filepath}")
        return True
        
    except Exception as e:
        print(f"Error creating chart for {symbol}: {e}")
        return False