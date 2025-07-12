#!/usr/bin/env python3
"""Test PROK Enhanced Real-Time ORB Alert System for 2025-07-10"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import json
from pathlib import Path
from datetime import datetime
import pytz
import numpy as np

def test_prok_alerts():
    """Test PROK enhanced alerts for 2025-07-10"""
    
    symbol = "PROK"
    date = "2025-07-10"
    
    print(f"🧪 Testing Enhanced Real-Time ORB Alert System")
    print(f"Symbol: {symbol}, Date: {date}")
    print("="*50)
    
    # Load PROK data
    data_files = glob.glob(f'historical_data/{date}/market_data/{symbol}_*.csv')
    all_data = []
    
    for file in data_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        print(f"❌ No data found for {symbol} on {date}")
        return
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'])
    
    print(f"📊 Loaded {len(df)} data points for {symbol}")
    
    # Simulate ORB calculation (9:30-9:45 AM)
    et_tz = pytz.timezone('US/Eastern')
    market_open = df.iloc[0]['timestamp'].replace(hour=9, minute=30, second=0)
    orb_end = market_open + pd.Timedelta(minutes=15)
    
    # Filter ORB period data
    orb_data = df[(df['timestamp'] >= market_open) & (df['timestamp'] <= orb_end)]
    
    if len(orb_data) < 3:
        print(f"❌ Insufficient ORB data: {len(orb_data)} points")
        return
    
    # Calculate ORB features
    orb_high = orb_data['high'].max()
    orb_low = orb_data['low'].min()
    orb_range = orb_high - orb_low
    orb_midpoint = (orb_high + orb_low) / 2
    
    # Enhanced features
    orb_volume = orb_data['volume'].sum()
    orb_avg_volume = orb_data['volume'].mean()
    orb_volume_ratio = orb_volume / orb_avg_volume if orb_avg_volume > 0 else 1.0
    
    orb_open = orb_data.iloc[0]['close']
    orb_close = orb_data.iloc[-1]['close']
    orb_price_change = orb_close - orb_open
    orb_duration_minutes = len(orb_data)
    orb_momentum = orb_price_change / orb_duration_minutes if orb_duration_minutes > 0 else 0
    orb_range_pct = (orb_range / orb_open) * 100 if orb_open > 0 else 0
    
    print(f"\n📊 ORB Features for {symbol}:")
    print(f"   ORB High: ${orb_high:.3f}")
    print(f"   ORB Low: ${orb_low:.3f}")
    print(f"   ORB Range: ${orb_range:.3f} ({orb_range_pct:.1f}%)")
    print(f"   Volume Ratio: {orb_volume_ratio:.1f}x")
    print(f"   Momentum: {orb_momentum:.3f}")
    print(f"   Duration: {orb_duration_minutes} minutes")
    
    # Apply PCA filters
    pca_filters = {
        'volume_ratio_threshold': 2.5,
        'duration_threshold': 10,
        'momentum_threshold': -0.01,
        'range_pct_min': 5.0,
        'range_pct_max': 35.0
    }
    
    filters_passed = []
    filters_failed = []
    
    if orb_volume_ratio >= pca_filters['volume_ratio_threshold']:
        filters_passed.append(f"Volume Ratio: {orb_volume_ratio:.1f}x >= {pca_filters['volume_ratio_threshold']}x ✅")
    else:
        filters_failed.append(f"Volume Ratio: {orb_volume_ratio:.1f}x < {pca_filters['volume_ratio_threshold']}x ❌")
    
    if orb_duration_minutes >= pca_filters['duration_threshold']:
        filters_passed.append(f"Duration: {orb_duration_minutes}min >= {pca_filters['duration_threshold']}min ✅")
    else:
        filters_failed.append(f"Duration: {orb_duration_minutes}min < {pca_filters['duration_threshold']}min ❌")
    
    if orb_momentum >= pca_filters['momentum_threshold']:
        filters_passed.append(f"Momentum: {orb_momentum:.3f} >= {pca_filters['momentum_threshold']} ✅")
    else:
        filters_failed.append(f"Momentum: {orb_momentum:.3f} < {pca_filters['momentum_threshold']} ❌")
    
    if pca_filters['range_pct_min'] <= orb_range_pct <= pca_filters['range_pct_max']:
        filters_passed.append(f"Range: {orb_range_pct:.1f}% in [{pca_filters['range_pct_min']}-{pca_filters['range_pct_max']}%] ✅")
    else:
        filters_failed.append(f"Range: {orb_range_pct:.1f}% not in [{pca_filters['range_pct_min']}-{pca_filters['range_pct_max']}%] ❌")
    
    print(f"\n🔍 PCA Filter Results:")
    for filter_result in filters_passed:
        print(f"   {filter_result}")
    for filter_result in filters_failed:
        print(f"   {filter_result}")
    
    # Determine if alerts should be generated
    all_filters_passed = len(filters_failed) == 0
    
    if not all_filters_passed:
        print(f"\n❌ PCA filters failed - no alerts generated")
        return
    
    print(f"\n✅ All PCA filters passed - checking for breakouts...")
    
    # Check for breakouts after ORB period
    post_orb_data = df[df['timestamp'] > orb_end]
    
    alerts_generated = []
    
    # Check for bullish breakout
    bullish_breakout = post_orb_data[post_orb_data['high'] > orb_high]
    if not bullish_breakout.empty:
        first_breakout = bullish_breakout.iloc[0]
        alert_time = first_breakout['timestamp']
        
        # Calculate confidence
        confidence = 0.5
        if orb_volume_ratio > 5.0:
            confidence += 0.3
        elif orb_volume_ratio > 3.0:
            confidence += 0.2
        elif orb_volume_ratio > 2.5:
            confidence += 0.1
        
        if 15.0 <= orb_range_pct <= 25.0:
            confidence += 0.15
        elif 10.0 <= orb_range_pct <= 30.0:
            confidence += 0.1
        
        if orb_momentum > 0:
            confidence += 0.1
        
        if orb_duration_minutes >= 15:
            confidence += 0.05
        
        confidence = min(confidence, 1.0)
        
        # Create alert
        entry_price = orb_high * 1.002
        stop_loss = orb_low * 0.995
        target = entry_price + (orb_range * 1.5)
        
        alert = {
            "symbol": symbol,
            "date": date,
            "alert_time": alert_time.strftime('%H:%M:%S') + " ET",
            "alert_type": "ENHANCED_BULLISH_BREAKOUT",
            "confidence": confidence,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "volume_ratio": orb_volume_ratio,
            "momentum": orb_momentum,
            "range_pct": orb_range_pct,
            "reasoning": f"PCA-Enhanced: Vol Ratio {orb_volume_ratio:.1f}x, Momentum {orb_momentum:.3f}, Range {orb_range_pct:.1f}%",
            "orb_features": {
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_range": orb_range,
                "orb_midpoint": orb_midpoint
            }
        }
        
        alerts_generated.append(alert)
        print(f"🚀 ENHANCED BULLISH BREAKOUT ALERT:")
        print(f"   Time: {alert_time.strftime('%H:%M:%S')} ET")
        print(f"   Confidence: {confidence:.0%}")
        print(f"   Entry: ${entry_price:.3f}")
        print(f"   Stop: ${stop_loss:.3f}")
        print(f"   Target: ${target:.3f}")
    
    # Check for bearish breakdown
    bearish_breakdown = post_orb_data[post_orb_data['low'] < orb_low]
    if not bearish_breakdown.empty:
        first_breakdown = bearish_breakdown.iloc[0]
        alert_time = first_breakdown['timestamp']
        
        # Calculate confidence
        confidence = 0.5
        if orb_volume_ratio > 5.0:
            confidence += 0.3
        elif orb_volume_ratio > 3.0:
            confidence += 0.2
        elif orb_volume_ratio > 2.5:
            confidence += 0.1
        
        if 15.0 <= orb_range_pct <= 25.0:
            confidence += 0.15
        elif 10.0 <= orb_range_pct <= 30.0:
            confidence += 0.1
        
        if orb_momentum < 0:
            confidence += 0.1
        
        if orb_duration_minutes >= 15:
            confidence += 0.05
        
        confidence = min(confidence, 1.0)
        
        # Create alert
        entry_price = orb_low * 0.998
        stop_loss = orb_high * 1.005
        target = entry_price - (orb_range * 1.5)
        
        alert = {
            "symbol": symbol,
            "date": date,
            "alert_time": alert_time.strftime('%H:%M:%S') + " ET",
            "alert_type": "ENHANCED_BEARISH_BREAKDOWN",
            "confidence": confidence,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "volume_ratio": orb_volume_ratio,
            "momentum": orb_momentum,
            "range_pct": orb_range_pct,
            "reasoning": f"PCA-Enhanced: Vol Ratio {orb_volume_ratio:.1f}x, Momentum {orb_momentum:.3f}, Range {orb_range_pct:.1f}%",
            "orb_features": {
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_range": orb_range,
                "orb_midpoint": orb_midpoint
            }
        }
        
        alerts_generated.append(alert)
        print(f"🔻 ENHANCED BEARISH BREAKDOWN ALERT:")
        print(f"   Time: {alert_time.strftime('%H:%M:%S')} ET")
        print(f"   Confidence: {confidence:.0%}")
        print(f"   Entry: ${entry_price:.3f}")
        print(f"   Stop: ${stop_loss:.3f}")
        print(f"   Target: ${target:.3f}")
    
    if not alerts_generated:
        print(f"\n❌ No breakouts detected after ORB period")
        return
    
    # Create visualization
    create_prok_visualization(df, alerts_generated, orb_high, orb_low, orb_midpoint, symbol, date)
    
    # Save results
    save_prok_results(alerts_generated, symbol, date)

def create_prok_visualization(df, alerts, orb_high, orb_low, orb_midpoint, symbol, date):
    """Create candlestick chart with alert visualization"""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Plot candlesticks
    for i, row in df.iterrows():
        timestamp = row['timestamp']
        high = row['high']
        low = row['low']
        close = row['close']
        
        # Estimate open price
        if i > 0:
            open_price = df.iloc[i-1]['close']
        else:
            open_price = close
        
        # Color based on direction
        color = 'green' if close >= open_price else 'red'
        
        # Draw high-low line
        ax1.plot([timestamp, timestamp], [low, high], color='black', linewidth=1)
        
        # Draw body
        body_height = abs(close - open_price)
        if body_height > 0:
            bottom = min(open_price, close)
            ax1.bar(timestamp, body_height, bottom=bottom, 
                   width=pd.Timedelta(minutes=1.35), 
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add ORB levels
    ax1.axhline(y=orb_high, color='blue', linestyle='--', linewidth=2, 
               label=f'ORB High: ${orb_high:.3f}', alpha=0.8)
    ax1.axhline(y=orb_low, color='blue', linestyle='--', linewidth=2, 
               label=f'ORB Low: ${orb_low:.3f}', alpha=0.8)
    ax1.axhline(y=orb_midpoint, color='blue', linestyle=':', linewidth=1, 
               label=f'ORB Mid: ${orb_midpoint:.3f}', alpha=0.6)
    
    # Add alerts
    for i, alert in enumerate(alerts):
        alert_time_str = alert['alert_time'].replace(' ET', '')
        alert_time = datetime.strptime(f"{date} {alert_time_str}", "%Y-%m-%d %H:%M:%S")
        alert_time = pd.Timestamp(alert_time)
        
        color = 'green' if 'BULLISH' in alert['alert_type'] else 'red'
        alert_type = alert['alert_type'].replace('ENHANCED_', '')
        
        ax1.axvline(x=alert_time, color=color, linestyle='-', linewidth=4, 
                   alpha=0.9, label=f'{alert_type} Alert')
        
        # Add annotation
        direction = 'BULLISH BREAKOUT' if 'BULLISH' in alert['alert_type'] else 'BEARISH BREAKDOWN'
        emoji = '🚀' if 'BULLISH' in alert['alert_type'] else '🔻'
        
        ax1.annotate(f'{emoji} ENHANCED {direction}\\nConfidence: {alert["confidence"]:.0%}\\nEntry: ${alert["entry_price"]:.3f}\\nVol Ratio: {alert["volume_ratio"]:.1f}x',
                    xy=(alert_time, alert['entry_price']),
                    xytext=(alert_time, alert['entry_price'] + 0.3),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Add entry, stop, target lines
        ax1.axhline(y=alert['entry_price'], color='orange', linestyle='-', linewidth=2, 
                   label=f'Entry: ${alert["entry_price"]:.3f}', alpha=0.9)
        ax1.axhline(y=alert['stop_loss'], color='red', linestyle='-', linewidth=2, 
                   label=f'Stop Loss: ${alert["stop_loss"]:.3f}', alpha=0.8)
        ax1.axhline(y=alert['target'], color='green', linestyle='-', linewidth=2, 
                   label=f'Target: ${alert["target"]:.3f}', alpha=0.8)
    
    # Format price chart
    ax1.set_title(f'{symbol} Enhanced Real-Time ORB Alert Test - {date}\\n'
                 f'Alerts Generated: {len(alerts)}', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Volume chart
    timestamps = df['timestamp']
    volumes = df['volume']
    ax2.bar(timestamps, volumes, width=pd.Timedelta(minutes=1.35), 
           color='lightblue', alpha=0.7, edgecolor='blue', linewidth=0.5)
    
    # Add alert timing to volume chart
    for alert in alerts:
        alert_time_str = alert['alert_time'].replace(' ET', '')
        alert_time = datetime.strptime(f"{date} {alert_time_str}", "%Y-%m-%d %H:%M:%S")
        alert_time = pd.Timestamp(alert_time)
        
        color = 'green' if 'BULLISH' in alert['alert_type'] else 'red'
        ax2.axvline(x=alert_time, color=color, linestyle='-', linewidth=4, alpha=0.9)
    
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Time (ET)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    plt.tight_layout()
    
    # Save chart
    test_dir = Path(f"test_results/enhanced_alerts_{date}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    chart_file = test_dir / f"{symbol}_enhanced_realtime_alert_test.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    
    pdf_file = test_dir / f"{symbol}_enhanced_realtime_alert_test.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    
    print(f"\n📊 {symbol} enhanced alert visualization saved:")
    print(f"   PNG: {chart_file}")
    print(f"   PDF: {pdf_file}")
    
    plt.show()

def save_prok_results(alerts, symbol, date):
    """Save test results"""
    
    test_dir = Path(f"test_results/enhanced_alerts_{date}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save alerts
    alerts_file = test_dir / f"{symbol}_captured_alerts.json"
    with open(alerts_file, 'w') as f:
        json.dump(alerts, f, indent=2)
    
    # Save summary
    summary = {
        "symbol": symbol,
        "date": date,
        "total_alerts": len(alerts),
        "alert_types": [alert['alert_type'] for alert in alerts],
        "average_confidence": sum(alert['confidence'] for alert in alerts) / len(alerts) if alerts else 0,
        "test_results": {
            "pca_filters_passed": True,
            "alerts_generated": len(alerts) > 0,
            "real_time_simulation": "successful"
        }
    }
    
    summary_file = test_dir / f"{symbol}_test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 {symbol} test results saved:")
    print(f"   Alerts: {alerts_file}")
    print(f"   Summary: {summary_file}")

if __name__ == "__main__":
    test_prok_alerts()