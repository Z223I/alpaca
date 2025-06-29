"""
Atom for plotting candlestick charts with volume for stock market data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Optional


def plot_candle_chart(df: pd.DataFrame, symbol: str, output_dir: str = 'plots') -> bool:
    """
    Create a candlestick chart with volume for a single stock symbol.
    
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
                    color='black', linewidth=1)
            
            # Draw the open-close rectangle
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            
            rect = Rectangle((mdates.date2num(timestamp) - 0.0003, bottom), 
                           0.0006, height, 
                           facecolor=color, edgecolor='black', alpha=0.8)
            ax1.add_patch(rect)
        
        # Format price chart
        ax1.set_title(f'{symbol} - Candlestick Chart with Volume', fontsize=14, fontweight='bold')
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