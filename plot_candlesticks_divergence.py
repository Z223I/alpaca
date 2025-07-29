#!/usr/bin/env python3
"""
Script to plot candlesticks for periods with above/below mean EMA divergence.
Focuses on 12:00-14:00 ET timeframe with high resolution output.
"""

import json
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

def extract_candlestick_data(file_path):
    """Extract OHLC candlestick data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract timestamp
        timestamp = data.get('timestamp', '')
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('-0400', '-04:00'))
        else:
            return None
        
        # Extract data from the original alert
        original_alert = data.get('latest_super_alert', {}).get('original_alert', {})
        
        return {
            'timestamp': dt,
            'symbol': data.get('symbol', 'UNKNOWN'),
            'open': original_alert.get('open_price'),
            'high': original_alert.get('high_price'),
            'low': original_alert.get('low_price'),
            'close': original_alert.get('close_price'),
            'volume': original_alert.get('volume'),
            'ema_divergence': original_alert.get('ema_divergence'),
            'ema_9': original_alert.get('ema_9'),
            'ema_20': original_alert.get('ema_20'),
            'current_price': original_alert.get('current_price'),
            'confidence_score': original_alert.get('confidence_score'),
            'volume_ratio': original_alert.get('volume_ratio')
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def plot_candlesticks(df, title, filename, mean_divergence):
    """Plot candlestick chart with EMA lines."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Candlestick plot
    for i, row in df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        alpha = 0.8
        
        # Draw the body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        rect = Rectangle((mdates.date2num(row['timestamp']) - 0.0001, body_bottom),
                        0.0002, body_height, 
                        facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
        
        # Draw the wicks
        ax1.plot([mdates.date2num(row['timestamp']), mdates.date2num(row['timestamp'])],
                [row['low'], row['high']], color='black', linewidth=1, alpha=0.8)
    
    # Plot EMA lines
    ax1.plot(df['timestamp'], df['ema_9'], 'blue', linewidth=2, label='EMA 9', alpha=0.8)
    ax1.plot(df['timestamp'], df['ema_20'], 'orange', linewidth=2, label='EMA 20', alpha=0.8)
    
    # Formatting
    ax1.set_title(f'{title}\nVWAV Candlesticks (12:00-14:00 ET) - July 28, 2025', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=11)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics text box  
    stats_text = f"""Statistics:
Alerts: {len(df)}
Avg EMA Divergence: {df['ema_divergence'].mean()*100:.2f}%
Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}
Avg Volume Ratio: {df['volume_ratio'].mean():.2f}x
Avg Confidence: {df['confidence_score'].mean():.3f}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Volume plot
    colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(df['close'], df['open'])]
    ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7, width=0.0005)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time (ET)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis for volume plot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=1200, bbox_inches='tight', facecolor='white')
    print(f"High resolution plot saved as '{filename}'")
    plt.close()

def main():
    # Find all JSON files
    json_files = glob.glob('historical_data/2025-07-28/superduper_alerts/bullish/*.json')
    
    print(f"Found {len(json_files)} JSON files")
    
    # Extract data from all files
    data_points = []
    for file_path in json_files:
        data_point = extract_candlestick_data(file_path)
        if data_point and all(data_point[key] is not None for key in 
                            ['open', 'high', 'low', 'close', 'ema_divergence']):
            data_points.append(data_point)
    
    if not data_points:
        print("No valid data points found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    df = df.sort_values('timestamp')
    
    # Filter for 12:00-14:00 ET timeframe
    start_time = time(12, 0)  # 12:00 PM
    end_time = time(14, 0)    # 2:00 PM
    
    df_filtered = df[(df['timestamp'].dt.time >= start_time) & 
                     (df['timestamp'].dt.time <= end_time)].copy()
    
    if len(df_filtered) == 0:
        print("No data points found in 12:00-14:00 ET timeframe!")
        return
    
    print(f"Filtered to {len(df_filtered)} data points in 12:00-14:00 ET timeframe")
    
    # Calculate mean divergence for the filtered dataset
    mean_divergence = df_filtered['ema_divergence'].mean()
    print(f"Mean EMA divergence for 12:00-14:00 period: {mean_divergence*100:.3f}%")
    
    # Split data based on mean divergence
    above_mean = df_filtered[df_filtered['ema_divergence'] >= mean_divergence].copy()
    below_mean = df_filtered[df_filtered['ema_divergence'] < mean_divergence].copy()
    
    print(f"Above mean divergence: {len(above_mean)} alerts")
    print(f"Below mean divergence: {len(below_mean)} alerts")
    
    if len(above_mean) == 0 or len(below_mean) == 0:
        print("Warning: One of the groups is empty!")
        if len(above_mean) > 0:
            print("Only plotting above mean divergence data")
            plot_candlesticks(above_mean, 
                            f"Above Mean EMA Divergence (≥{mean_divergence*100:.2f}%)",
                            'candlesticks_above_mean_divergence_1200dpi.png',
                            mean_divergence)
        if len(below_mean) > 0:
            print("Only plotting below mean divergence data")
            plot_candlesticks(below_mean,
                            f"Below Mean EMA Divergence (<{mean_divergence*100:.2f}%)", 
                            'candlesticks_below_mean_divergence_1200dpi.png',
                            mean_divergence)
        return
    
    # Create plots for both groups
    plot_candlesticks(above_mean, 
                     f"Above Mean EMA Divergence (≥{mean_divergence*100:.2f}%)",
                     'candlesticks_above_mean_divergence_1200dpi.png',
                     mean_divergence)
    
    plot_candlesticks(below_mean,
                     f"Below Mean EMA Divergence (<{mean_divergence*100:.2f}%)", 
                     'candlesticks_below_mean_divergence_1200dpi.png',
                     mean_divergence)
    
    # Print detailed analysis
    print("\n=== DETAILED ANALYSIS (12:00-14:00 ET) ===")
    print(f"\nABOVE MEAN DIVERGENCE GROUP (≥{mean_divergence*100:.2f}%):")
    print(f"  Count: {len(above_mean)} alerts")
    print(f"  Time range: {above_mean['timestamp'].dt.time.min()} - {above_mean['timestamp'].dt.time.max()}")
    print(f"  Avg EMA divergence: {above_mean['ema_divergence'].mean()*100:.3f}%")
    print(f"  Price range: ${above_mean['low'].min():.2f} - ${above_mean['high'].max():.2f}")
    print(f"  Avg confidence: {above_mean['confidence_score'].mean():.3f}")
    print(f"  Avg volume ratio: {above_mean['volume_ratio'].mean():.2f}x")
    
    print(f"\nBELOW MEAN DIVERGENCE GROUP (<{mean_divergence*100:.2f}%):")
    print(f"  Count: {len(below_mean)} alerts")
    print(f"  Time range: {below_mean['timestamp'].dt.time.min()} - {below_mean['timestamp'].dt.time.max()}")
    print(f"  Avg EMA divergence: {below_mean['ema_divergence'].mean()*100:.3f}%")
    print(f"  Price range: ${below_mean['low'].min():.2f} - ${below_mean['high'].max():.2f}")
    print(f"  Avg confidence: {below_mean['confidence_score'].mean():.3f}")
    print(f"  Avg volume ratio: {below_mean['volume_ratio'].mean():.2f}x")
    
    # Compare the groups
    print(f"\nCOMPARISON:")
    price_diff = above_mean['current_price'].mean() - below_mean['current_price'].mean()
    conf_diff = above_mean['confidence_score'].mean() - below_mean['confidence_score'].mean()
    vol_diff = above_mean['volume_ratio'].mean() - below_mean['volume_ratio'].mean()
    
    print(f"  Price difference (above - below): ${price_diff:.3f}")
    print(f"  Confidence difference: {conf_diff:.3f}")
    print(f"  Volume ratio difference: {vol_diff:.2f}x")

if __name__ == "__main__":
    main()