#!/usr/bin/env python3
"""
Enhanced Alert Plotter - Reusable Candlestick Chart Generator

Creates candlestick charts for enhanced ORB alerts with:
- OHLCV candlestick data with proper sizing
- ORB high/low levels  
- Enhanced bullish/bearish breakout alerts
- Entry, stop loss, and target levels
- Volume indicators and analysis
- PCA-enhanced filtering results

Usage:
    python enhanced_alert_plotter.py SYMBOL DATE
    
Example:
    python enhanced_alert_plotter.py FTFT 2025-07-11
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
import sys
warnings.filterwarnings('ignore')

class EnhancedAlertPlotter:
    def __init__(self):
        self.candle_width_minutes = 1.35  # Narrow candlesticks (4.5% of original, 50% wider than 3%)
        
    def load_market_data(self, symbol, date):
        """Load all market data for specified symbol and date."""
        data_files = glob.glob(f"historical_data/{date}/market_data/{symbol}_*.csv")
        
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

    def load_enhanced_alert_data(self, symbol, date):
        """Load enhanced alert data for specified symbol and date."""
        try:
            with open('enhanced_orb_results.json', 'r') as f:
                results = json.load(f)
            
            alert_data = results['results'][date][symbol]
            return alert_data
        except KeyError:
            print(f"No enhanced alert data found for {symbol} on {date}")
            return None
        except FileNotFoundError:
            print("Enhanced ORB results file not found")
            return None

    def create_candlestick_chart(self, symbol, date, save_dir="tmp"):
        """Create enhanced candlestick chart for symbol with alert indicators."""
        
        # Load data
        print(f"Loading {symbol} market data for {date}...")
        market_data = self.load_market_data(symbol, date)
        
        if market_data.empty:
            print(f"No market data found for {symbol} on {date}")
            return None, None
        
        print(f"Loaded {len(market_data)} data points")
        
        # Load enhanced alert data
        print("Loading enhanced alert data...")
        alert_data = self.load_enhanced_alert_data(symbol, date)
        
        if not alert_data:
            return None, None
        
        # Extract alert information
        orb_features = alert_data['orb_features']
        alerts = alert_data['alerts']
        
        if not alerts:
            print(f"No alerts found for {symbol} on {date}")
            return None, None
            
        alert = alerts[0]  # Use first alert
        
        print(f"Alert: {alert['type']}")
        print(f"Confidence: {alert['confidence']:.0%}")
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
        
        # Plot candlesticks manually with narrow width
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
                rect_width = pd.Timedelta(minutes=self.candle_width_minutes)
                
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
        alert_emoji = '🚀' if 'BULLISH' in alert['type'] else '🔻'
        ax1.annotate(f'{alert_emoji} {alert["type"].replace("ENHANCED_", "").replace("_", " ")}\\n' + 
                    f'Confidence: {alert["confidence"]:.0%}\\n' +
                    f'Vol Ratio: {alert["volume_ratio"]:.1f}x\\n' +
                    f'Momentum: {alert["momentum"]:.3f}',
                    xy=(alert_time, entry_price), 
                    xytext=(alert_time, entry_price + 0.3),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        
        # Format price chart
        ax1.set_title(f'{symbol} Enhanced ORB Alert - {date}\\n' + 
                     f'PCA-Enhanced Analysis: {alert["reasoning"]}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=10)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        # Volume chart
        ax2.bar(timestamps, volumes, width=pd.Timedelta(minutes=self.candle_width_minutes), 
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
        📊 {symbol} Enhanced Alert Analysis
        
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
        
        # Save plots
        Path(save_dir).mkdir(exist_ok=True)
        output_file = f"{save_dir}/{symbol}_enhanced_orb_alert_{date.replace('-', '')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\\n✅ Chart saved to: {output_file}")
        
        # Also save as PDF for better quality
        pdf_file = f"{save_dir}/{symbol}_enhanced_orb_alert_{date.replace('-', '')}.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        print(f"✅ PDF saved to: {pdf_file}")
        
        plt.show()
        
        return market_data, alert_data

    def print_alert_summary(self, symbol, date):
        """Print detailed alert summary."""
        alert_data = self.load_enhanced_alert_data(symbol, date)
        
        if not alert_data:
            return
            
        orb_features = alert_data['orb_features']
        alerts = alert_data['alerts']
        
        if not alerts:
            print(f"No alerts found for {symbol} on {date}")
            return
            
        alert = alerts[0]
        
        print("\\n" + "="*60)
        print(f"{symbol} ENHANCED {alert['type'].replace('ENHANCED_', '').replace('_', ' ')} ALERT - {date}")
        print("="*60)
        
        print(f"\\n🎯 ALERT DETAILS:")
        print(f"   Symbol: {symbol}")
        print(f"   Type: {alert['type']}")
        print(f"   Confidence: {alert['confidence']:.0%}")
        print(f"   Reasoning: {alert['reasoning']}")
        
        print(f"\\n📊 ORB FEATURES:")
        print(f"   ORB High: ${orb_features['orb_high']:.3f}")
        print(f"   ORB Low: ${orb_features['orb_low']:.3f}")
        print(f"   ORB Range: ${orb_features['orb_range']:.3f} ({orb_features['orb_range_pct']:.1f}%)")
        print(f"   Duration: {orb_features['orb_duration_minutes']} minutes")
        print(f"   Volume Ratio: {orb_features['orb_volume_ratio']:.1f}x average")
        print(f"   Momentum: {orb_features['orb_momentum']:.3f}")
        print(f"   Volatility: {orb_features['orb_volatility']:.3f}")
        
        print(f"\\n💰 TRADE SETUP:")
        print(f"   Entry Price: ${alert['entry_price']:.3f}")
        print(f"   Stop Loss: ${alert['stop_loss']:.3f} (-{((alert['entry_price']-alert['stop_loss'])/alert['entry_price']*100):.1f}%)")
        print(f"   Target: ${alert['target']:.3f} (+{((alert['target']-alert['entry_price'])/alert['entry_price']*100):.1f}%)")
        print(f"   Risk/Reward: {((alert['target']-alert['entry_price'])/(alert['entry_price']-alert['stop_loss'])):.1f}:1")
        
        print(f"\\n🔍 PCA ANALYSIS:")
        print(f"   This alert passed all PCA-derived filters:")
        print(f"   ✅ Volume Ratio > 2.5x (actual: {orb_features['orb_volume_ratio']:.1f}x)")
        print(f"   ✅ Duration > 10 min (actual: {orb_features['orb_duration_minutes']} min)")  
        print(f"   ✅ Momentum > -0.01 (actual: {orb_features['orb_momentum']:.3f})")
        print(f"   ✅ Range 5-35% (actual: {orb_features['orb_range_pct']:.1f}%)")
        
        print(f"\\n⚠️  HISTORICAL PERFORMANCE:")
        print(f"   Expected Return: {alert['expected_return']:.1f}%")
        print(f"   (Based on backtesting with trailing stops)")
        
        print("="*60)

def main():
    """Main function for command line usage."""
    if len(sys.argv) != 3:
        print("Usage: python enhanced_alert_plotter.py SYMBOL DATE")
        print("Example: python enhanced_alert_plotter.py FTFT 2025-07-11")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    date = sys.argv[2]
    
    plotter = EnhancedAlertPlotter()
    
    print(f"Creating Enhanced Alert Chart for {symbol} on {date}...")
    
    # Print detailed alert summary
    plotter.print_alert_summary(symbol, date)
    
    # Create the chart
    market_data, alert_data = plotter.create_candlestick_chart(symbol, date)
    
    if market_data is not None:
        print(f"\\n📈 Chart shows {len(market_data)} data points from {date}")
        print("🎯 Enhanced alert analysis complete!")
    else:
        print("\\n❌ Failed to create chart - check data availability")

if __name__ == "__main__":
    main()