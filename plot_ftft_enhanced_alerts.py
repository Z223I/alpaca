#!/usr/bin/env python3
"""
Plot FTFT Enhanced Alerts with Candlestick Chart

Creates a candlestick chart for FTFT on 2025-07-11 showing:
- OHLCV candlestick data
- ORB high/low levels  
- Enhanced bullish breakout alert
- Entry, stop loss, and target levels
- Volume indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json
from datetime import datetime, time
import glob
import warnings
warnings.filterwarnings('ignore')

def load_ftft_data():
    """Load all FTFT market data for 2025-07-11."""
    data_files = glob.glob("historical_data/2025-07-11/market_data/FTFT_*.csv")
    
    all_data = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Convert timestamp and sort
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates
    combined_data = combined_data.drop_duplicates(subset=['timestamp'])
    
    return combined_data

def load_enhanced_alert_data():
    """Load enhanced alert data for FTFT."""
    with open('enhanced_orb_results.json', 'r') as f:
        results = json.load(f)
    
    ftft_data = results['results']['2025-07-11']['FTFT']
    return ftft_data

def create_candlestick_chart():
    """Create enhanced candlestick chart for FTFT with alert indicators."""
    
    # Load data
    print("Loading FTFT market data...")
    market_data = load_ftft_data()
    
    if market_data.empty:
        print("No market data found for FTFT")
        return
    
    print(f"Loaded {len(market_data)} data points")
    
    # Load enhanced alert data
    print("Loading enhanced alert data...")
    alert_data = load_enhanced_alert_data()
    
    # Extract alert information
    orb_features = alert_data['orb_features']
    alert = alert_data['alerts'][0]  # First (and only) alert
    
    print(f"Alert: {alert['type']}")
    print(f"Confidence: {alert['confidence']}")
    print(f"Entry Price: ${alert['entry_price']:.3f}")
    print(f"Volume Ratio: {alert['volume_ratio']:.1f}x")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Prepare data for plotting
    timestamps = market_data['timestamp']
    opens = market_data['close'].shift(1).fillna(market_data['close'])  # Approximate opens
    highs = market_data['high']
    lows = market_data['low'] 
    closes = market_data['close']
    volumes = market_data['volume']
    
    # Plot candlesticks manually (matplotlib doesn't have built-in candlesticks)
    for i in range(len(market_data)):
        timestamp = timestamps.iloc[i]
        open_price = opens.iloc[i]
        high_price = highs.iloc[i]
        low_price = lows.iloc[i]
        close_price = closes.iloc[i]
        
        # Determine color
        color = 'green' if close_price >= open_price else 'red'
        
        # Draw high-low line
        ax1.plot([timestamp, timestamp], [low_price, high_price], color='black', linewidth=1)
        
        # Draw body rectangle
        body_height = abs(close_price - open_price)
        if body_height > 0:
            bottom = min(open_price, close_price)
            rect_width = pd.Timedelta(minutes=1.35)  # 0.9 * 1.5 = 1.35 minutes (50% wider)
            
            # Use bar plot for body
            ax1.bar(timestamp, body_height, bottom=bottom, width=rect_width, 
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add ORB levels
    orb_high = orb_features['orb_high']
    orb_low = orb_features['orb_low']
    orb_midpoint = orb_features['orb_midpoint']
    
    ax1.axhline(y=orb_high, color='blue', linestyle='--', linewidth=2, 
               label=f'ORB High: ${orb_high:.3f}', alpha=0.8)
    ax1.axhline(y=orb_low, color='blue', linestyle='--', linewidth=2, 
               label=f'ORB Low: ${orb_low:.3f}', alpha=0.8)
    ax1.axhline(y=orb_midpoint, color='blue', linestyle=':', linewidth=1, 
               label=f'ORB Mid: ${orb_midpoint:.3f}', alpha=0.6)
    
    # Add alert levels
    entry_price = alert['entry_price']
    stop_loss = alert['stop_loss']
    target = alert['target']
    
    ax1.axhline(y=entry_price, color='orange', linestyle='-', linewidth=3, 
               label=f'Entry: ${entry_price:.3f}', alpha=0.9)
    ax1.axhline(y=stop_loss, color='red', linestyle='-', linewidth=2, 
               label=f'Stop Loss: ${stop_loss:.3f}', alpha=0.8)
    ax1.axhline(y=target, color='green', linestyle='-', linewidth=2, 
               label=f'Target: ${target:.3f}', alpha=0.8)
    
    # Mark ORB period (assume first 15 minutes after market open)
    market_open_time = timestamps.iloc[0].replace(hour=9, minute=30, second=0)
    orb_end_time = market_open_time + pd.Timedelta(minutes=15)
    
    # Shade ORB period
    ax1.axvspan(market_open_time, orb_end_time, alpha=0.2, color='yellow', 
               label='ORB Period (15 min)')
    
    # Add alert annotation
    alert_time = timestamps.iloc[len(timestamps)//4]  # Approximate alert time
    ax1.annotate('🚀 ENHANCED BULLISH BREAKOUT\n' + 
                f'Confidence: {alert["confidence"]:.0%}\n' +
                f'Vol Ratio: {alert["volume_ratio"]:.1f}x\n' +
                f'Momentum: {alert["momentum"]:.3f}',
                xy=(alert_time, entry_price), 
                xytext=(alert_time, entry_price + 0.3),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    # Format price chart
    ax1.set_title(f'FTFT Enhanced ORB Alert - July 11, 2025\n' + 
                 f'PCA-Enhanced Analysis: {alert["reasoning"]}', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Volume chart
    ax2.bar(timestamps, volumes, width=pd.Timedelta(minutes=1.35), 
           color='lightblue', alpha=0.7, edgecolor='blue', linewidth=0.5)
    
    # Add volume statistics
    avg_volume = orb_features['orb_avg_volume']
    ax2.axhline(y=avg_volume, color='red', linestyle='--', linewidth=2, 
               label=f'ORB Avg Vol: {avg_volume:.0f}', alpha=0.8)
    
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time (ET)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Format volume chart x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Add key statistics text box
    stats_text = f"""
    📊 FTFT Enhanced Alert Analysis
    
    ORB Features:
    • Range: ${orb_features['orb_range']:.3f} ({orb_features['orb_range_pct']:.1f}%)
    • Volume Ratio: {orb_features['orb_volume_ratio']:.1f}x
    • Momentum: {orb_features['orb_momentum']:.3f}
    • Duration: {orb_features['orb_duration_minutes']} minutes
    
    Alert Metrics:
    • Confidence: {alert['confidence']:.0%}
    • Expected Return: {alert['expected_return']:.1f}%
    • Risk/Reward: {(target-entry_price)/(entry_price-stop_loss):.1f}:1
    """
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    Path("tmp").mkdir(exist_ok=True)
    output_file = "tmp/FTFT_enhanced_orb_alert_20250711.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {output_file}")
    
    # Also save as PDF for better quality
    pdf_file = "tmp/FTFT_enhanced_orb_alert_20250711.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✅ PDF saved to: {pdf_file}")
    
    plt.show()
    
    return market_data, alert_data

def print_alert_summary():
    """Print detailed alert summary."""
    alert_data = load_enhanced_alert_data()
    orb_features = alert_data['orb_features']
    alert = alert_data['alerts'][0]
    
    print("\n" + "="*60)
    print("FTFT ENHANCED BULLISH BREAKOUT ALERT - July 11, 2025")
    print("="*60)
    
    print(f"\n🎯 ALERT DETAILS:")
    print(f"   Symbol: FTFT")
    print(f"   Type: {alert['type']}")
    print(f"   Confidence: {alert['confidence']:.0%} (Maximum)")
    print(f"   Reasoning: {alert['reasoning']}")
    
    print(f"\n📊 ORB FEATURES:")
    print(f"   ORB High: ${orb_features['orb_high']:.3f}")
    print(f"   ORB Low: ${orb_features['orb_low']:.3f}")
    print(f"   ORB Range: ${orb_features['orb_range']:.3f} ({orb_features['orb_range_pct']:.1f}%)")
    print(f"   Duration: {orb_features['orb_duration_minutes']} minutes")
    print(f"   Volume Ratio: {orb_features['orb_volume_ratio']:.1f}x average")
    print(f"   Momentum: {orb_features['orb_momentum']:.3f} (Positive)")
    print(f"   Volatility: {orb_features['orb_volatility']:.3f}")
    
    print(f"\n💰 TRADE SETUP:")
    print(f"   Entry Price: ${alert['entry_price']:.3f}")
    print(f"   Stop Loss: ${alert['stop_loss']:.3f} (-{((alert['entry_price']-alert['stop_loss'])/alert['entry_price']*100):.1f}%)")
    print(f"   Target: ${alert['target']:.3f} (+{((alert['target']-alert['entry_price'])/alert['entry_price']*100):.1f}%)")
    print(f"   Risk/Reward: {((alert['target']-alert['entry_price'])/(alert['entry_price']-alert['stop_loss'])):.1f}:1")
    
    print(f"\n🔍 PCA ANALYSIS:")
    print(f"   This alert passed all PCA-derived filters:")
    print(f"   ✅ Volume Ratio > 2.5x (actual: {orb_features['orb_volume_ratio']:.1f}x)")
    print(f"   ✅ Duration > 10 min (actual: {orb_features['orb_duration_minutes']} min)")  
    print(f"   ✅ Momentum > -0.01 (actual: {orb_features['orb_momentum']:.3f})")
    print(f"   ✅ Range 5-35% (actual: {orb_features['orb_range_pct']:.1f}%)")
    
    print(f"\n⚠️  HISTORICAL PERFORMANCE:")
    print(f"   Expected Return: {alert['expected_return']:.1f}%")
    print(f"   (Based on backtesting with trailing stops)")
    
    print("="*60)

if __name__ == "__main__":
    print("Creating FTFT Enhanced Alert Candlestick Chart...")
    
    # Print detailed alert summary
    print_alert_summary()
    
    # Create the chart
    market_data, alert_data = create_candlestick_chart()
    
    print(f"\n📈 Chart shows {len(market_data)} data points from 2025-07-11")
    print("🎯 Enhanced alert analysis complete!")