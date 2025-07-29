#!/usr/bin/env python3
"""
Comprehensive candlestick chart with EMA lines and superduper alert overlays.
Creates two charts: one for alerts at/above mean EMA divergence, one for below.
Focuses on 12:00-14:00 ET timeframe.
"""

import json
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import pytz

def load_candlestick_data(symbol='VWAV', date='20250728'):
    """Load minute-by-minute candlestick data from CSV files."""
    csv_files = glob.glob(f'historical_data/2025-07-28/market_data/{symbol}_*.csv')
    
    if not csv_files:
        print(f"No CSV files found for {symbol}")
        return None
    
    # Combine all CSV files for the symbol
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    # Combine and sort by timestamp
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    
    # Localize to Eastern Time if naive
    if combined_df['timestamp'].dt.tz is None:
        combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize('US/Eastern')
    
    combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    return combined_df

def calculate_emas(df):
    """Calculate EMA 9 and EMA 20 for the price data."""
    df = df.copy()
    
    # Calculate EMA 9
    alpha_9 = 2 / (9 + 1)
    df['ema_9'] = df['close'].ewm(alpha=alpha_9, adjust=False).mean()
    
    # Calculate EMA 20
    alpha_20 = 2 / (20 + 1)
    df['ema_20'] = df['close'].ewm(alpha=alpha_20, adjust=False).mean()
    
    # Calculate divergence
    df['ema_divergence'] = (df['ema_9'] - df['ema_20']) / df['ema_20']
    
    return df

def load_superduper_alerts():
    """Load all superduper alerts from JSON files."""
    json_files = glob.glob('historical_data/2025-07-28/superduper_alerts/bullish/*.json')
    
    alerts = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp
            timestamp = data.get('timestamp', '')
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace('-0400', '-04:00'))
            else:
                continue
            
            # Extract data from the original alert
            original_alert = data.get('latest_super_alert', {}).get('original_alert', {})
            
            alerts.append({
                'timestamp': dt,
                'symbol': data.get('symbol', 'UNKNOWN'),
                'current_price': original_alert.get('current_price'),
                'ema_divergence': original_alert.get('ema_divergence'),
                'ema_9': original_alert.get('ema_9'),
                'ema_20': original_alert.get('ema_20'),
                'confidence_score': original_alert.get('confidence_score'),
                'volume_ratio': original_alert.get('volume_ratio')
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return pd.DataFrame(alerts)

def filter_timeframe(df, start_time=time(12, 0), end_time=time(14, 0)):
    """Filter data to specific timeframe (12:00-14:00 ET by default)."""
    return df[(df['timestamp'].dt.time >= start_time) & 
              (df['timestamp'].dt.time <= end_time)].copy()

def plot_comprehensive_chart(candlestick_df, alerts_df, title, filename, mean_divergence=None):
    """Plot comprehensive candlestick chart with EMA lines and alert overlays."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), 
                                   gridspec_kw={'height_ratios': [4, 1]})
    
    # Plot candlesticks
    for i, row in candlestick_df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        alpha = 0.7
        
        # Draw the body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        width = 0.0003  # Smaller width for minute bars
        rect = Rectangle((mdates.date2num(row['timestamp']) - width/2, body_bottom),
                        width, body_height, 
                        facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.3)
        ax1.add_patch(rect)
        
        # Draw the wicks
        ax1.plot([mdates.date2num(row['timestamp']), mdates.date2num(row['timestamp'])],
                [row['low'], row['high']], color='black', linewidth=0.5, alpha=0.8)
    
    # Plot EMA lines
    ax1.plot(candlestick_df['timestamp'], candlestick_df['ema_9'], 'blue', 
             linewidth=2, label='EMA 9', alpha=0.9)
    ax1.plot(candlestick_df['timestamp'], candlestick_df['ema_20'], 'orange', 
             linewidth=2, label='EMA 20', alpha=0.9)
    
    # Overlay superduper alerts as vertical bars
    if len(alerts_df) > 0:
        # Get the y-axis range for full-height bars
        y_min = min(candlestick_df['low'].min(), alerts_df['current_price'].min()) * 0.98
        y_max = max(candlestick_df['high'].max(), alerts_df['current_price'].max()) * 1.02
        
        for _, alert in alerts_df.iterrows():
            # Determine color based on the chart type and mean divergence
            if mean_divergence is not None:
                if title.startswith("Above Mean"):
                    color = 'green'  # All alerts in "above mean" chart are green
                elif title.startswith("Below Mean"):
                    color = 'red'    # All alerts in "below mean" chart are red
                else:
                    # Mixed chart - color by actual divergence
                    color = 'green' if alert['ema_divergence'] >= mean_divergence else 'red'
            else:
                color = 'blue'  # Default color if no mean specified
            
            # Draw vertical line from bottom to top of chart
            ax1.axvline(x=alert['timestamp'], color=color, alpha=0.7, linewidth=2, zorder=8)
            
            # Add divergence percentage at the top
            ax1.annotate(f'{alert["ema_divergence"]*100:.1f}%', 
                        (alert['timestamp'], y_max * 0.98),
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=8, alpha=0.9, ha='center', rotation=90,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add legend for alert colors
        if mean_divergence is not None:
            from matplotlib.lines import Line2D
            if title.startswith("Above Mean"):
                legend_elements = [
                    Line2D([0], [0], color='blue', linewidth=2, label='EMA 9'),
                    Line2D([0], [0], color='orange', linewidth=2, label='EMA 20'),
                    Line2D([0], [0], color='green', linewidth=2, alpha=0.7, 
                           label=f'Superduper Alerts ≥{mean_divergence*100:.1f}% ({len(alerts_df)})')
                ]
            elif title.startswith("Below Mean"):
                legend_elements = [
                    Line2D([0], [0], color='blue', linewidth=2, label='EMA 9'),
                    Line2D([0], [0], color='orange', linewidth=2, label='EMA 20'),
                    Line2D([0], [0], color='red', linewidth=2, alpha=0.7,
                           label=f'Superduper Alerts <{mean_divergence*100:.1f}% ({len(alerts_df)})')
                ]
            else:
                # Mixed chart
                legend_elements = [
                    Line2D([0], [0], color='blue', linewidth=2, label='EMA 9'),
                    Line2D([0], [0], color='orange', linewidth=2, label='EMA 20'),
                    Line2D([0], [0], color='green', linewidth=2, alpha=0.7, 
                           label=f'≥{mean_divergence*100:.1f}% divergence ({len(alerts_df[alerts_df["ema_divergence"] >= mean_divergence])})'),
                    Line2D([0], [0], color='red', linewidth=2, alpha=0.7,
                           label=f'<{mean_divergence*100:.1f}% divergence ({len(alerts_df[alerts_df["ema_divergence"] < mean_divergence])})')
                ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Formatting
    ax1.set_title(f'{title}\nVWAV Candlesticks with EMA Lines & Superduper Alert Overlays - July 28, 2025 (12:00-14:00 ET)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add EMA legend if no alerts (alert legend will override if present)
    if len(alerts_df) == 0 or mean_divergence is None:
        ax1.legend(loc='upper left', fontsize=11)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics text box  
    if len(alerts_df) > 0:
        stats_text = f"""Alert Statistics:
Superduper Alerts: {len(alerts_df)}
Avg EMA Divergence: {alerts_df['ema_divergence'].mean()*100:.2f}%
Alert Price Range: ${alerts_df['current_price'].min():.2f} - ${alerts_df['current_price'].max():.2f}
Avg Confidence: {alerts_df['confidence_score'].mean():.3f}
Avg Volume Ratio: {alerts_df['volume_ratio'].mean():.2f}x

Candlestick Data:
Total Bars: {len(candlestick_df)}
Price Range: ${candlestick_df['low'].min():.2f} - ${candlestick_df['high'].max():.2f}
EMA 9 Range: ${candlestick_df['ema_9'].min():.2f} - ${candlestick_df['ema_9'].max():.2f}
EMA 20 Range: ${candlestick_df['ema_20'].min():.2f} - ${candlestick_df['ema_20'].max():.2f}"""
    else:
        stats_text = f"""Candlestick Data:
Total Bars: {len(candlestick_df)}
Price Range: ${candlestick_df['low'].min():.2f} - ${candlestick_df['high'].max():.2f}
EMA 9 Range: ${candlestick_df['ema_9'].min():.2f} - ${candlestick_df['ema_9'].max():.2f}
EMA 20 Range: ${candlestick_df['ema_20'].min():.2f} - ${candlestick_df['ema_20'].max():.2f}
No Superduper Alerts in timeframe"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Volume plot
    colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(candlestick_df['close'], candlestick_df['open'])]
    ax2.bar(candlestick_df['timestamp'], candlestick_df['volume'], color=colors, alpha=0.6, width=0.0005)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time (ET)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Overlay alert volume markers as vertical bars
    if len(alerts_df) > 0:
        for _, alert in alerts_df.iterrows():
            # Determine color based on the chart type and mean divergence (same as main chart)
            if mean_divergence is not None:
                if title.startswith("Above Mean"):
                    color = 'green'  # All alerts in "above mean" chart are green
                elif title.startswith("Below Mean"):
                    color = 'red'    # All alerts in "below mean" chart are red
                else:
                    # Mixed chart - color by actual divergence
                    color = 'green' if alert['ema_divergence'] >= mean_divergence else 'red'
            else:
                color = 'blue'
            
            # Draw vertical line for volume chart
            ax2.axvline(x=alert['timestamp'], color=color, alpha=0.7, linewidth=2, zorder=10)
    
    # Format x-axis for volume plot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High resolution comprehensive plot saved as '{filename}'")
    plt.close()

def main():
    print("Loading candlestick data...")
    # Load minute-by-minute candlestick data
    candlestick_df = load_candlestick_data('VWAV')
    if candlestick_df is None:
        print("Failed to load candlestick data!")
        return
    
    print(f"Loaded {len(candlestick_df)} candlestick data points")
    
    # Calculate EMAs
    candlestick_df = calculate_emas(candlestick_df)
    
    print("Loading superduper alerts...")
    # Load superduper alerts
    all_alerts_df = load_superduper_alerts()
    print(f"Loaded {len(all_alerts_df)} superduper alerts")
    
    # Filter both datasets to 12:00-14:00 ET timeframe
    start_time = time(12, 0)  # 12:00 PM
    end_time = time(14, 0)    # 2:00 PM
    
    candlestick_filtered = filter_timeframe(candlestick_df, start_time, end_time)
    alerts_filtered = filter_timeframe(all_alerts_df, start_time, end_time)
    
    print(f"Filtered candlestick data: {len(candlestick_filtered)} bars")
    print(f"Filtered alerts: {len(alerts_filtered)} alerts")
    
    if len(candlestick_filtered) == 0:
        print("No candlestick data found in 12:00-14:00 ET timeframe!")
        return
    
    if len(alerts_filtered) == 0:
        print("No superduper alerts found in 12:00-14:00 ET timeframe!")
        # Still create chart with just candlesticks
        plot_comprehensive_chart(candlestick_filtered, pd.DataFrame(), 
                                "All Candlesticks (No Alerts in Timeframe)",
                                'comprehensive_candlesticks_no_alerts_1200dpi.png')
        return
    
    # Calculate mean EMA divergence for the filtered alerts
    mean_divergence = alerts_filtered['ema_divergence'].mean()
    print(f"Mean EMA divergence for alerts in 12:00-14:00 period: {mean_divergence*100:.3f}%")
    
    # Split alerts based on mean divergence
    above_mean_alerts = alerts_filtered[alerts_filtered['ema_divergence'] >= mean_divergence].copy()
    below_mean_alerts = alerts_filtered[alerts_filtered['ema_divergence'] < mean_divergence].copy()
    
    print(f"Above mean divergence: {len(above_mean_alerts)} alerts")
    print(f"Below mean divergence: {len(below_mean_alerts)} alerts")
    
    # Create comprehensive charts
    if len(above_mean_alerts) > 0:
        plot_comprehensive_chart(candlestick_filtered, above_mean_alerts,
                                f"Above Mean EMA Divergence (≥{mean_divergence*100:.2f}%)",
                                'comprehensive_above_mean_divergence.png',
                                mean_divergence)
    
    if len(below_mean_alerts) > 0:
        plot_comprehensive_chart(candlestick_filtered, below_mean_alerts,
                                f"Below Mean EMA Divergence (<{mean_divergence*100:.2f}%)",
                                'comprehensive_below_mean_divergence.png',
                                mean_divergence)
    
    # Print detailed analysis
    print("\n=== COMPREHENSIVE ANALYSIS (12:00-14:00 ET) ===")
    print(f"\nCANDLESTICK DATA:")
    print(f"  Total bars: {len(candlestick_filtered)}")
    print(f"  Time range: {candlestick_filtered['timestamp'].dt.time.min()} - {candlestick_filtered['timestamp'].dt.time.max()}")
    print(f"  Price range: ${candlestick_filtered['low'].min():.2f} - ${candlestick_filtered['high'].max():.2f}")
    print(f"  EMA 9 range: ${candlestick_filtered['ema_9'].min():.2f} - ${candlestick_filtered['ema_9'].max():.2f}")
    print(f"  EMA 20 range: ${candlestick_filtered['ema_20'].min():.2f} - ${candlestick_filtered['ema_20'].max():.2f}")
    print(f"  Average EMA divergence: {candlestick_filtered['ema_divergence'].mean()*100:.3f}%")
    
    if len(above_mean_alerts) > 0:
        print(f"\nABOVE MEAN DIVERGENCE ALERTS (≥{mean_divergence*100:.2f}%):")
        print(f"  Count: {len(above_mean_alerts)} alerts")
        print(f"  Time range: {above_mean_alerts['timestamp'].dt.time.min()} - {above_mean_alerts['timestamp'].dt.time.max()}")
        print(f"  Avg EMA divergence: {above_mean_alerts['ema_divergence'].mean()*100:.3f}%")
        print(f"  Alert price range: ${above_mean_alerts['current_price'].min():.2f} - ${above_mean_alerts['current_price'].max():.2f}")
        print(f"  Avg confidence: {above_mean_alerts['confidence_score'].mean():.3f}")
        print(f"  Avg volume ratio: {above_mean_alerts['volume_ratio'].mean():.2f}x")
    
    if len(below_mean_alerts) > 0:
        print(f"\nBELOW MEAN DIVERGENCE ALERTS (<{mean_divergence*100:.2f}%):")
        print(f"  Count: {len(below_mean_alerts)} alerts")
        print(f"  Time range: {below_mean_alerts['timestamp'].dt.time.min()} - {below_mean_alerts['timestamp'].dt.time.max()}")
        print(f"  Avg EMA divergence: {below_mean_alerts['ema_divergence'].mean()*100:.3f}%")
        print(f"  Alert price range: ${below_mean_alerts['current_price'].min():.2f} - ${below_mean_alerts['current_price'].max():.2f}")
        print(f"  Avg confidence: {below_mean_alerts['confidence_score'].mean():.3f}")
        print(f"  Avg volume ratio: {below_mean_alerts['volume_ratio'].mean():.2f}x")

if __name__ == "__main__":
    main()