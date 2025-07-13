"""
Atom for plotting candlestick charts with volume for stock market data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from typing import Optional, Tuple, List
from datetime import time

from ..utils.calculate_orb_levels import calculate_orb_levels
from ..utils.extract_symbol_data import extract_symbol_data
from ..utils.calculate_ema import calculate_ema
from ..utils.calculate_vwap import calculate_vwap_typical


def plot_candle_chart(df: pd.DataFrame, symbol: str, output_dir: str = 'plots', alerts: Optional[List] = None) -> bool:
    """
    Create a candlestick chart with volume and ORB rectangle for a single stock symbol.
    
    Args:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        symbol: Stock symbol name
        output_dir: Directory to save the chart
        alerts: Optional list of alert dictionaries with timestamp_dt and alert_type
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract symbol data using the dedicated atom
        symbol_data = extract_symbol_data(df, symbol)
        
        if symbol_data is None:
            print(f"No data found for symbol: {symbol}")
            return False
        
        # Calculate ORB levels
        orb_high, orb_low = calculate_orb_levels(symbol_data)
        
        # Calculate EMA (9-period) for close prices
        ema_success, ema_values = calculate_ema(symbol_data, price_column='close', period=9)
        
        # Calculate EMA (20-period) for close prices
        ema20_success, ema20_values = calculate_ema(symbol_data, price_column='close', period=20)
        
        # Calculate VWAP using typical price (HLC/3)
        vwap_success, vwap_values = calculate_vwap_typical(symbol_data)
        
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
            
            rect = Rectangle((float(mdates.date2num(timestamp)) - 0.0003, bottom), 
                           0.0006, height, 
                           facecolor=color, edgecolor='none', alpha=0.8,
                           zorder=2)
            ax1.add_patch(rect)
        
        # Add ORB rectangle if ORB levels were calculated
        if orb_high is not None and orb_low is not None:
            # Get time range for ORB rectangle (first 15 candlesticks or all available data)
            x_min = float(mdates.date2num(symbol_data['timestamp'].min()))
            
            # Calculate width to cover first 15 candlesticks (or all if less than 15)
            orb_candlesticks = min(15, len(symbol_data))
            if orb_candlesticks > 1:
                x_max = float(mdates.date2num(symbol_data['timestamp'].iloc[orb_candlesticks - 1]))
            else:
                x_max = float(mdates.date2num(symbol_data['timestamp'].iloc[0]))
            
            orb_width = x_max - x_min + 0.0012  # Add small padding to cover the last candlestick
            
            # Draw ORB rectangle (no fill with thin edge)
            orb_rect = Rectangle((x_min - 0.0006, orb_low), orb_width, orb_high - orb_low,
                               facecolor='none', edgecolor='black', alpha=1.0, linewidth=1.5)
            ax1.add_patch(orb_rect)
            
            # Add ORB level lines
            ax1.axhline(y=orb_high, color='black', linestyle='--', linewidth=1, alpha=0.8, label=f'ORB High: ${orb_high:.2f}')
            ax1.axhline(y=orb_low, color='black', linestyle='--', linewidth=1, alpha=0.8, label=f'ORB Low: ${orb_low:.2f}')
            
            isDebugging = False  # Set to True for debugging output
            if isDebugging:
                # Add legend for ORB levels
                ax1.legend(loc='upper left', fontsize=10)
                
                print(f"ORB levels for {symbol}: High=${orb_high:.2f}, Low=${orb_low:.2f}")
        
        # Add EMA line if calculation was successful
        if ema_success:
            ax1.plot(symbol_data['timestamp'], ema_values, 
                    color='blue', linewidth=2, alpha=0.8, label='EMA(9)')
        
        # Add EMA20 line if calculation was successful
        if ema20_success:
            ax1.plot(symbol_data['timestamp'], ema20_values, 
                    color='orange', linewidth=2, alpha=0.8, label='EMA(20)')
        
        # Add VWAP line if calculation was successful
        if vwap_success:
            ax1.plot(symbol_data['timestamp'], vwap_values, 
                    color='purple', linewidth=2, alpha=0.8, label='VWAP')
        
        # Plot alerts as vertical bars if provided
        if alerts:
            # Convert symbol_data timestamps to timezone-naive for comparison
            chart_start_time = symbol_data['timestamp'].min()
            chart_end_time = symbol_data['timestamp'].max()
            
            # Make sure chart times are timezone-naive for comparison
            if hasattr(chart_start_time, 'tz_localize'):
                chart_start_time = chart_start_time.tz_localize(None) if chart_start_time.tz is None else chart_start_time.tz_convert(None)
                chart_end_time = chart_end_time.tz_localize(None) if chart_end_time.tz is None else chart_end_time.tz_convert(None)
            
            alert_count = 0
            for alert in alerts:
                alert_time = alert.get('timestamp_dt')
                alert_type = alert.get('alert_type', 'unknown')
                
                if not alert_time:
                    continue
                
                # Make alert time timezone-naive for comparison
                if hasattr(alert_time, 'tz'):
                    alert_time = alert_time.replace(tzinfo=None)
                
                # Check if alert time is within chart timeframe
                if chart_start_time <= alert_time <= chart_end_time:
                    # Set color based on alert type
                    color = 'green' if alert_type == 'bullish' else 'red'
                    alpha = 0.3
                    
                    # Draw vertical line spanning the price chart
                    y_min, y_max = ax1.get_ylim()
                    ax1.axvline(x=alert_time, color=color, alpha=alpha, linewidth=2, linestyle='-')
                    
                    # Add small label at top of chart
                    label_y = y_max - (y_max - y_min) * 0.05  # 5% from top
                    alert_symbol = '↑' if alert_type == 'bullish' else '↓'
                    ax1.text(alert_time, label_y, alert_symbol, 
                            color=color, fontsize=12, fontweight='bold', 
                            ha='center', va='top')
                    
                    alert_count += 1
            
            if alert_count > 0:
                print(f"Plotted {alert_count} alerts for {symbol}")

        # Add legend if any indicators were calculated
        if ema_success or ema20_success or vwap_success:
            # Determine legend position based on price trend
            # If closing price is higher than opening price, put legend at bottom right
            # Otherwise, put it at top right to avoid overlapping with rising prices
            first_close = symbol_data['close'].iloc[0]
            last_close = symbol_data['close'].iloc[-1]
            
            if last_close > first_close:
                legend_location = 'lower right'
            else:
                legend_location = 'upper right'
            
            ax1.legend(loc=legend_location, fontsize=10)
        
        # Format price chart
        title = symbol
        
        # Get date from first timestamp and format for US (MM/DD/YYYY)
        chart_date = symbol_data['timestamp'].iloc[0].strftime('%m/%d/%Y')
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.text(0.5, 0.95, chart_date, transform=ax1.transAxes, fontsize=12, 
                ha='center', va='top', style='italic')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Format Y-axis to show prices with 2 decimal places
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.2f}'))
        
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