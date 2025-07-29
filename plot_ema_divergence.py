#!/usr/bin/env python3
"""
Script to analyze and plot EMA divergence from superduper alert JSON files.
"""

import json
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np

def extract_ema_data(file_path):
    """Extract EMA divergence and related data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract timestamp
        timestamp = data.get('timestamp', '')
        if timestamp:
            # Parse timestamp (format: "2025-07-28T13:15:00-0400")
            dt = datetime.fromisoformat(timestamp.replace('-0400', '-04:00'))
        else:
            return None
        
        # Extract EMA data from the original alert
        original_alert = data.get('latest_super_alert', {}).get('original_alert', {})
        
        ema_divergence = original_alert.get('ema_divergence')
        ema_9 = original_alert.get('ema_9')
        ema_20 = original_alert.get('ema_20')
        current_price = original_alert.get('current_price')
        symbol = data.get('symbol', 'UNKNOWN')
        confidence_score = original_alert.get('confidence_score')
        volume_ratio = original_alert.get('volume_ratio')
        
        return {
            'timestamp': dt,
            'symbol': symbol,
            'ema_divergence': ema_divergence,
            'ema_9': ema_9,
            'ema_20': ema_20,
            'current_price': current_price,
            'confidence_score': confidence_score,
            'volume_ratio': volume_ratio,
            'file_path': file_path
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # Find all JSON files
    json_files = glob.glob('historical_data/2025-07-28/superduper_alerts/bullish/*.json')
    
    print(f"Found {len(json_files)} JSON files")
    
    # Extract data from all files
    data_points = []
    for file_path in json_files:
        data_point = extract_ema_data(file_path)
        if data_point and data_point['ema_divergence'] is not None:
            data_points.append(data_point)
    
    if not data_points:
        print("No valid data points found!")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data_points)
    df = df.sort_values('timestamp')
    
    print(f"Extracted {len(df)} valid data points")
    print(f"EMA Divergence range: {df['ema_divergence'].min():.4f} to {df['ema_divergence'].max():.4f}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: EMA Divergence over time
    ax1.plot(df['timestamp'], df['ema_divergence'] * 100, 'b-o', markersize=4, linewidth=1.5)
    ax1.set_title('EMA Divergence Over Time (VWAV - July 28, 2025)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (ET)')
    ax1.set_ylabel('EMA Divergence (%)')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add horizontal line at 0%
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='EMA 9 = EMA 20')
    ax1.legend()
    
    # Plot 2: EMA 9 vs EMA 20 over time
    ax2.plot(df['timestamp'], df['ema_9'], 'g-', label='EMA 9', linewidth=2)
    ax2.plot(df['timestamp'], df['ema_20'], 'r-', label='EMA 20', linewidth=2)
    ax2.plot(df['timestamp'], df['current_price'], 'b--', label='Current Price', alpha=0.7)
    ax2.set_title('EMA 9 vs EMA 20 vs Current Price', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (ET)')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: EMA Divergence vs Confidence Score
    scatter = ax3.scatter(df['ema_divergence'] * 100, df['confidence_score'], 
                         c=df.index, cmap='viridis', alpha=0.7, s=50)
    ax3.set_title('EMA Divergence vs Confidence Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('EMA Divergence (%)')
    ax3.set_ylabel('Confidence Score')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Time Sequence')
    
    # Plot 4: Distribution of EMA Divergence
    ax4.hist(df['ema_divergence'] * 100, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Distribution of EMA Divergence Values', fontsize=14, fontweight='bold')
    ax4.set_xlabel('EMA Divergence (%)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_div = df['ema_divergence'].mean() * 100
    std_div = df['ema_divergence'].std() * 100
    ax4.axvline(mean_div, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_div:.2f}%')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('ema_divergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'ema_divergence_analysis.png'")
    
    # Print summary statistics
    print("\n=== EMA DIVERGENCE ANALYSIS SUMMARY ===")
    print(f"Total alerts analyzed: {len(df)}")
    print(f"Symbol: {df['symbol'].iloc[0]}")
    print(f"Date: {df['timestamp'].dt.date.iloc[0]}")
    print(f"Time range: {df['timestamp'].dt.time.min()} - {df['timestamp'].dt.time.max()}")
    print(f"\nEMA Divergence Statistics:")
    print(f"  Mean: {mean_div:.3f}%")
    print(f"  Std Dev: {std_div:.3f}%")
    print(f"  Min: {df['ema_divergence'].min() * 100:.3f}%")
    print(f"  Max: {df['ema_divergence'].max() * 100:.3f}%")
    print(f"  Range: {(df['ema_divergence'].max() - df['ema_divergence'].min()) * 100:.3f}%")
    
    print(f"\nConfidence Score Statistics:")
    print(f"  Mean: {df['confidence_score'].mean():.3f}")
    print(f"  Min: {df['confidence_score'].min():.3f}")
    print(f"  Max: {df['confidence_score'].max():.3f}")
    
    print(f"\nVolume Ratio Statistics:")
    print(f"  Mean: {df['volume_ratio'].mean():.2f}x")
    print(f"  Min: {df['volume_ratio'].min():.2f}x")
    print(f"  Max: {df['volume_ratio'].max():.2f}x")
    
    # Show correlation between EMA divergence and other metrics
    correlation_conf = df['ema_divergence'].corr(df['confidence_score'])
    correlation_vol = df['ema_divergence'].corr(df['volume_ratio'])
    
    print(f"\nCorrelations:")
    print(f"  EMA Divergence vs Confidence Score: {correlation_conf:.3f}")
    print(f"  EMA Divergence vs Volume Ratio: {correlation_vol:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()