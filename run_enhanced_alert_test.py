#!/usr/bin/env python3
"""
Enhanced ORB Alert Test Runner

This script runs the enhanced real-time ORB alert system tests for any date and symbol(s).
It simulates the PCA-enhanced filtering and generates alerts with visualizations.

Usage:
    python run_enhanced_alert_test.py 2025-07-11                    # Test all symbols for date
    python run_enhanced_alert_test.py 2025-07-11 --symbol FTFT     # Test specific symbol
    python run_enhanced_alert_test.py 2025-07-10 --symbol PROK     # Test PROK on 2025-07-10
    python run_enhanced_alert_test.py 2025-07-11 --symbols FTFT,PROK,AAPL  # Test multiple symbols
    python run_enhanced_alert_test.py 2025-07-11 --all             # Test all available symbols
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import json
from pathlib import Path
from datetime import datetime
import pytz
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

class EnhancedORBTestRunner:
    """Enhanced ORB Alert Test Runner with PCA filtering"""
    
    def __init__(self, date: str):
        self.date = date
        self.test_dir = Path(f"test_results/enhanced_alerts_{date}")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # PCA-derived thresholds
        self.pca_filters = {
            'volume_ratio_threshold': 2.5,
            'duration_threshold': 10,
            'momentum_threshold': -0.01,
            'range_pct_min': 5.0,
            'range_pct_max': 35.0
        }
        
        print(f"🧪 Enhanced ORB Alert Test Runner")
        print(f"Date: {date}")
        print(f"PCA Filters: Vol>{self.pca_filters['volume_ratio_threshold']}x, "
              f"Duration>{self.pca_filters['duration_threshold']}min, "
              f"Momentum>{self.pca_filters['momentum_threshold']}, "
              f"Range:{self.pca_filters['range_pct_min']}-{self.pca_filters['range_pct_max']}%")
        print("="*70)
    
    def find_available_symbols(self) -> list:
        """Find all available symbols for the given date"""
        data_dir = Path(f"historical_data/{self.date}/market_data")
        if not data_dir.exists():
            print(f"❌ No data directory found for {self.date}")
            return []
        
        # Find all CSV files and extract symbols
        csv_files = list(data_dir.glob("*.csv"))
        symbols = []
        
        for file in csv_files:
            # Extract symbol from filename (assume format: SYMBOL_timestamp.csv)
            symbol = file.stem.split('_')[0]
            if symbol not in symbols:
                symbols.append(symbol)
        
        symbols.sort()
        return symbols
    
    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load all data for a specific symbol"""
        data_files = glob.glob(f'historical_data/{self.date}/market_data/{symbol}_*.csv')
        all_data = []
        
        for file in data_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"   Error loading {file}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine and clean data
        df = pd.concat(all_data, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'])
        
        return df
    
    def calculate_orb_features(self, symbol: str, df: pd.DataFrame) -> dict:
        """Calculate ORB features for a symbol"""
        if df.empty:
            return None
        
        # Define ORB period (9:30-9:45 AM)
        market_open = df.iloc[0]['timestamp'].replace(hour=9, minute=30, second=0)
        orb_end = market_open + pd.Timedelta(minutes=15)
        
        # Filter ORB period data
        orb_data = df[(df['timestamp'] >= market_open) & (df['timestamp'] <= orb_end)]
        
        if len(orb_data) < 3:
            return None
        
        # Basic ORB calculations
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
        
        return {
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_range': orb_range,
            'orb_range_pct': orb_range_pct,
            'orb_midpoint': orb_midpoint,
            'orb_volume': orb_volume,
            'orb_avg_volume': orb_avg_volume,
            'orb_volume_ratio': orb_volume_ratio,
            'orb_price_change': orb_price_change,
            'orb_duration_minutes': orb_duration_minutes,
            'orb_momentum': orb_momentum,
            'orb_open': orb_open,
            'orb_close': orb_close,
            'market_open': market_open,
            'orb_end': orb_end
        }
    
    def apply_pca_filters(self, orb_features: dict) -> tuple:
        """Apply PCA filters and return results"""
        filters_passed = []
        filters_failed = []
        
        # Volume ratio filter
        if orb_features['orb_volume_ratio'] >= self.pca_filters['volume_ratio_threshold']:
            filters_passed.append(f"Volume Ratio: {orb_features['orb_volume_ratio']:.1f}x >= {self.pca_filters['volume_ratio_threshold']}x ✅")
        else:
            filters_failed.append(f"Volume Ratio: {orb_features['orb_volume_ratio']:.1f}x < {self.pca_filters['volume_ratio_threshold']}x ❌")
        
        # Duration filter
        if orb_features['orb_duration_minutes'] >= self.pca_filters['duration_threshold']:
            filters_passed.append(f"Duration: {orb_features['orb_duration_minutes']}min >= {self.pca_filters['duration_threshold']}min ✅")
        else:
            filters_failed.append(f"Duration: {orb_features['orb_duration_minutes']}min < {self.pca_filters['duration_threshold']}min ❌")
        
        # Momentum filter
        if orb_features['orb_momentum'] >= self.pca_filters['momentum_threshold']:
            filters_passed.append(f"Momentum: {orb_features['orb_momentum']:.3f} >= {self.pca_filters['momentum_threshold']} ✅")
        else:
            filters_failed.append(f"Momentum: {orb_features['orb_momentum']:.3f} < {self.pca_filters['momentum_threshold']} ❌")
        
        # Range filter
        range_pct = orb_features['orb_range_pct']
        if self.pca_filters['range_pct_min'] <= range_pct <= self.pca_filters['range_pct_max']:
            filters_passed.append(f"Range: {range_pct:.1f}% in [{self.pca_filters['range_pct_min']}-{self.pca_filters['range_pct_max']}%] ✅")
        else:
            filters_failed.append(f"Range: {range_pct:.1f}% not in [{self.pca_filters['range_pct_min']}-{self.pca_filters['range_pct_max']}%] ❌")
        
        return filters_passed, filters_failed
    
    def calculate_confidence(self, orb_features: dict, direction: str) -> float:
        """Calculate confidence score based on PCA factors"""
        confidence = 0.5
        
        # Volume ratio contribution
        vol_ratio = orb_features['orb_volume_ratio']
        if vol_ratio > 5.0:
            confidence += 0.3
        elif vol_ratio > 3.0:
            confidence += 0.2
        elif vol_ratio > 2.5:
            confidence += 0.1
        
        # Range contribution
        range_pct = orb_features['orb_range_pct']
        if 15.0 <= range_pct <= 25.0:
            confidence += 0.15
        elif 10.0 <= range_pct <= 30.0:
            confidence += 0.1
        
        # Momentum contribution
        momentum = orb_features['orb_momentum']
        if direction == 'bullish' and momentum > 0:
            confidence += 0.1
        elif direction == 'bearish' and momentum < 0:
            confidence += 0.1
        
        # Duration contribution
        if orb_features['orb_duration_minutes'] >= 15:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def detect_breakouts(self, symbol: str, df: pd.DataFrame, orb_features: dict) -> list:
        """Detect breakouts and generate alerts"""
        alerts = []
        
        # Get post-ORB data
        post_orb_data = df[df['timestamp'] > orb_features['orb_end']]
        
        if post_orb_data.empty:
            return alerts
        
        orb_high = orb_features['orb_high']
        orb_low = orb_features['orb_low']
        orb_range = orb_features['orb_range']
        
        # Check for bullish breakout
        bullish_breakout = post_orb_data[post_orb_data['high'] > orb_high]
        if not bullish_breakout.empty:
            first_breakout = bullish_breakout.iloc[0]
            alert_time = first_breakout['timestamp']
            
            confidence = self.calculate_confidence(orb_features, 'bullish')
            entry_price = orb_high * 1.002
            stop_loss = orb_low * 0.995
            target = entry_price + (orb_range * 1.5)
            
            alert = {
                "symbol": symbol,
                "date": self.date,
                "alert_time": alert_time.strftime('%H:%M:%S') + " ET",
                "alert_timestamp": alert_time,
                "alert_type": "ENHANCED_BULLISH_BREAKOUT",
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "volume_ratio": orb_features['orb_volume_ratio'],
                "momentum": orb_features['orb_momentum'],
                "range_pct": orb_features['orb_range_pct'],
                "reasoning": f"PCA-Enhanced: Vol Ratio {orb_features['orb_volume_ratio']:.1f}x, Momentum {orb_features['orb_momentum']:.3f}, Range {orb_features['orb_range_pct']:.1f}%",
                "orb_features": {
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "orb_range": orb_range,
                    "orb_midpoint": orb_features['orb_midpoint']
                }
            }
            alerts.append(alert)
        
        # Check for bearish breakdown
        bearish_breakdown = post_orb_data[post_orb_data['low'] < orb_low]
        if not bearish_breakdown.empty:
            first_breakdown = bearish_breakdown.iloc[0]
            alert_time = first_breakdown['timestamp']
            
            confidence = self.calculate_confidence(orb_features, 'bearish')
            entry_price = orb_low * 0.998
            stop_loss = orb_high * 1.005
            target = entry_price - (orb_range * 1.5)
            
            alert = {
                "symbol": symbol,
                "date": self.date,
                "alert_time": alert_time.strftime('%H:%M:%S') + " ET",
                "alert_timestamp": alert_time,
                "alert_type": "ENHANCED_BEARISH_BREAKDOWN",
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "volume_ratio": orb_features['orb_volume_ratio'],
                "momentum": orb_features['orb_momentum'],
                "range_pct": orb_features['orb_range_pct'],
                "reasoning": f"PCA-Enhanced: Vol Ratio {orb_features['orb_volume_ratio']:.1f}x, Momentum {orb_features['orb_momentum']:.3f}, Range {orb_features['orb_range_pct']:.1f}%",
                "orb_features": {
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "orb_range": orb_range,
                    "orb_midpoint": orb_features['orb_midpoint']
                }
            }
            alerts.append(alert)
        
        return alerts
    
    def create_visualization(self, symbol: str, df: pd.DataFrame, alerts: list, orb_features: dict):
        """Create candlestick chart with alert visualization"""
        if df.empty:
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        # Plot candlesticks
        prev_close = None
        for i, row in df.iterrows():
            timestamp = row['timestamp']
            high = row['high']
            low = row['low']
            close = row['close']
            
            # Estimate open price
            if prev_close is not None:
                open_price = prev_close
            else:
                open_price = close
            
            prev_close = close
            
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
        orb_high = orb_features['orb_high']
        orb_low = orb_features['orb_low']
        orb_midpoint = orb_features['orb_midpoint']
        
        ax1.axhline(y=orb_high, color='blue', linestyle='--', linewidth=2, 
                   label=f'ORB High: ${orb_high:.3f}', alpha=0.8)
        ax1.axhline(y=orb_low, color='blue', linestyle='--', linewidth=2, 
                   label=f'ORB Low: ${orb_low:.3f}', alpha=0.8)
        ax1.axhline(y=orb_midpoint, color='blue', linestyle=':', linewidth=1, 
                   label=f'ORB Mid: ${orb_midpoint:.3f}', alpha=0.6)
        
        # Add alerts
        for i, alert in enumerate(alerts):
            alert_time = alert['alert_timestamp']
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
        ax1.set_title(f'{symbol} Enhanced Real-Time ORB Alert Test - {self.date}\\n'
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
            alert_time = alert['alert_timestamp']
            color = 'green' if 'BULLISH' in alert['alert_type'] else 'red'
            ax2.axvline(x=alert_time, color=color, linestyle='-', linewidth=4, alpha=0.9)
        
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time (ET)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        plt.tight_layout()
        
        # Save chart
        chart_file = self.test_dir / f"{symbol}_enhanced_realtime_alert_test.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        
        pdf_file = self.test_dir / f"{symbol}_enhanced_realtime_alert_test.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        
        print(f"   📊 Visualization saved: {chart_file}")
        plt.close()  # Close to prevent display in headless mode
    
    def test_symbol(self, symbol: str) -> dict:
        """Test a single symbol and return results"""
        print(f"\n📈 Testing {symbol}...")
        
        # Load data
        df = self.load_symbol_data(symbol)
        if df.empty:
            print(f"   ❌ No data found for {symbol}")
            return None
        
        print(f"   📊 Loaded {len(df)} data points")
        
        # Calculate ORB features
        orb_features = self.calculate_orb_features(symbol, df)
        if not orb_features:
            print(f"   ❌ Insufficient ORB data for {symbol}")
            return None
        
        print(f"   📊 ORB: High=${orb_features['orb_high']:.3f}, Low=${orb_features['orb_low']:.3f}, "
              f"Range={orb_features['orb_range_pct']:.1f}%, Vol={orb_features['orb_volume_ratio']:.1f}x")
        
        # Apply PCA filters
        filters_passed, filters_failed = self.apply_pca_filters(orb_features)
        
        print(f"   🔍 PCA Filters:")
        for filter_result in filters_passed:
            print(f"      {filter_result}")
        for filter_result in filters_failed:
            print(f"      {filter_result}")
        
        # Check if all filters passed
        if filters_failed:
            print(f"   ❌ PCA filters failed - no alerts generated")
            return {
                'symbol': symbol,
                'pca_passed': False,
                'alerts': [],
                'orb_features': orb_features
            }
        
        # Detect breakouts
        alerts = self.detect_breakouts(symbol, df, orb_features)
        
        if alerts:
            for alert in alerts:
                direction = "🚀 BULLISH" if "BULLISH" in alert['alert_type'] else "🔻 BEARISH"
                print(f"   {direction} ALERT: {alert['alert_time']} - Confidence: {alert['confidence']:.0%}")
            
            # Create visualization
            self.create_visualization(symbol, df, alerts, orb_features)
        else:
            print(f"   ⚠️  No breakouts detected after ORB period")
        
        # Save results
        result = {
            'symbol': symbol,
            'pca_passed': True,
            'alerts': alerts,
            'orb_features': orb_features
        }
        
        # Save individual symbol results
        symbol_file = self.test_dir / f"{symbol}_test_results.json"
        with open(symbol_file, 'w') as f:
            # Convert timestamps to strings for JSON serialization
            json_result = result.copy()
            for alert in json_result['alerts']:
                alert['alert_timestamp'] = alert['alert_timestamp'].isoformat()
            json.dump(json_result, f, indent=2, default=str)
        
        return result
    
    def run_tests(self, symbols: list) -> dict:
        """Run tests for multiple symbols"""
        all_results = {
            'date': self.date,
            'symbols_tested': len(symbols),
            'pca_filters': self.pca_filters,
            'results': {},
            'summary': {
                'total_alerts': 0,
                'symbols_with_alerts': 0,
                'symbols_passed_pca': 0
            }
        }
        
        for symbol in symbols:
            result = self.test_symbol(symbol)
            if result:
                all_results['results'][symbol] = result
                
                if result['pca_passed']:
                    all_results['summary']['symbols_passed_pca'] += 1
                
                if result['alerts']:
                    all_results['summary']['total_alerts'] += len(result['alerts'])
                    all_results['summary']['symbols_with_alerts'] += 1
        
        # Save comprehensive results
        summary_file = self.test_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📊 Test Summary:")
        print(f"   Date: {self.date}")
        print(f"   Symbols tested: {all_results['symbols_tested']}")
        print(f"   Symbols passed PCA: {all_results['summary']['symbols_passed_pca']}")
        print(f"   Symbols with alerts: {all_results['summary']['symbols_with_alerts']}")
        print(f"   Total alerts generated: {all_results['summary']['total_alerts']}")
        print(f"   Results saved to: {self.test_dir}")
        
        return all_results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Enhanced ORB Alert Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_enhanced_alert_test.py 2025-07-11                     # Test all symbols for date
  python run_enhanced_alert_test.py 2025-07-11 --symbol FTFT      # Test specific symbol
  python run_enhanced_alert_test.py 2025-07-10 --symbol PROK      # Test PROK on 2025-07-10
  python run_enhanced_alert_test.py 2025-07-11 --symbols FTFT,PROK,AAPL  # Multiple symbols
  python run_enhanced_alert_test.py 2025-07-11 --all              # Test all available symbols
        '''
    )
    
    parser.add_argument('date', help='Date to test (YYYY-MM-DD format)')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--symbol', help='Test specific symbol')
    group.add_argument('--symbols', help='Test multiple symbols (comma-separated)')
    group.add_argument('--all', action='store_true', help='Test all available symbols')
    
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation for faster processing')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    # Create test runner
    runner = EnhancedORBTestRunner(args.date)
    
    # Determine symbols to test
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.all:
        symbols = runner.find_available_symbols()
        if not symbols:
            print(f"❌ No symbols found for {args.date}")
            sys.exit(1)
        print(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
    else:
        # Default: find all available symbols
        symbols = runner.find_available_symbols()
        if not symbols:
            print(f"❌ No symbols found for {args.date}")
            sys.exit(1)
        print(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Run tests
    try:
        results = runner.run_tests(symbols)
        print(f"\n✅ Testing completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()