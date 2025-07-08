#!/usr/bin/env python3
"""
ORB Alert Analyzer - Flexible script to analyze any symbol's historical data for ORB patterns

Usage:
    python3 orb_analyzer.py SYMBOL [CSV_FILE_PATH]
    python3 orb_analyzer.py BSLK
    python3 orb_analyzer.py AAPL /path/to/data.csv

If CSV_FILE_PATH is not provided, it will search for the file in historical_data directories.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
import argparse
from pathlib import Path
import glob

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from atoms.config.alert_config import config
from molecules.orb_alert_engine import ORBAlertEngine
from atoms.websocket.alpaca_stream import MarketData
from atoms.alerts.alert_formatter import ORBAlert

class ORBAnalyzer:
    """Analyzes historical data for ORB patterns and alerts."""
    
    def __init__(self, symbol: str, csv_file: str = None):
        """
        Initialize ORB analyzer.
        
        Args:
            symbol: Trading symbol to analyze
            csv_file: Path to CSV file (optional, will auto-detect if not provided)
        """
        self.symbol = symbol.upper()
        self.csv_file = csv_file
        self.alerts_generated = []
        
        # Find CSV file if not provided
        if not self.csv_file:
            self.csv_file = self._find_csv_file()
        
        if not self.csv_file or not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found for symbol {self.symbol}")
    
    def _find_csv_file(self) -> str:
        """Auto-detect CSV file for the symbol."""
        search_patterns = [
            f"../historical_data/*/market_data/{self.symbol}_*.csv",
            f"../tmp/{self.symbol}_*.csv", 
            f"../data/{self.symbol}_*.csv",
            f"historical_data/*/market_data/{self.symbol}_*.csv",
            f"tmp/{self.symbol}_*.csv",
            f"data/{self.symbol}_*.csv",
            f"{self.symbol}_*.csv"
        ]
        
        for pattern in search_patterns:
            files = glob.glob(pattern)
            if files:
                # Return the most recent file
                return max(files, key=os.path.getmtime)
        
        return None
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate CSV data."""
        try:
            df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(df)} data points from {self.csv_file}")
            
            # Validate required columns
            required_cols = ['timestamp', 'symbol', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  Missing columns: {missing_cols}")
                # Try to add missing columns with defaults
                if 'symbol' not in df.columns:
                    df['symbol'] = self.symbol
                if 'trade_count' not in df.columns:
                    df['trade_count'] = 1
                if 'vwap' not in df.columns:
                    df['vwap'] = df['close']
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for the specific symbol if multiple symbols in file
            if 'symbol' in df.columns:
                df = df[df['symbol'] == self.symbol]
            
            if len(df) == 0:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def calculate_orb_metrics(self, df: pd.DataFrame, orb_minutes: int = 15) -> dict:
        """Calculate Opening Range Breakout metrics."""
        
        # Determine ORB period
        orb_start = df['timestamp'].iloc[0]
        orb_end = orb_start + timedelta(minutes=orb_minutes)
        
        # Split data
        orb_data = df[df['timestamp'] <= orb_end]
        post_orb_data = df[df['timestamp'] > orb_end]
        
        if len(orb_data) == 0:
            return {"error": "No data in ORB period"}
        
        # Calculate ORB metrics
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        orb_range = orb_high - orb_low
        orb_midpoint = (orb_high + orb_low) / 2
        orb_volume = orb_data['volume'].sum()
        orb_avg_volume = orb_data['volume'].mean()
        
        # Post-ORB metrics
        post_orb_high = post_orb_data['high'].max() if len(post_orb_data) > 0 else orb_high
        post_orb_low = post_orb_data['low'].min() if len(post_orb_data) > 0 else orb_low
        final_price = post_orb_data['close'].iloc[-1] if len(post_orb_data) > 0 else orb_data['close'].iloc[-1]
        
        return {
            'orb_start': orb_start,
            'orb_end': orb_end,
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_range': orb_range,
            'orb_range_pct': (orb_range / orb_midpoint) * 100,
            'orb_midpoint': orb_midpoint,
            'orb_volume': orb_volume,
            'orb_avg_volume': orb_avg_volume,
            'orb_data_points': len(orb_data),
            'post_orb_high': post_orb_high,
            'post_orb_low': post_orb_low,
            'final_price': final_price,
            'post_orb_data_points': len(post_orb_data),
            'max_high_breakout_pct': ((post_orb_high - orb_high) / orb_high * 100) if post_orb_high > orb_high else 0,
            'max_low_breakout_pct': ((orb_low - post_orb_low) / orb_low * 100) if post_orb_low < orb_low else 0,
        }
    
    def detect_breakouts(self, df: pd.DataFrame, metrics: dict) -> list:
        """Detect and analyze breakouts."""
        breakouts = []
        
        orb_high = metrics['orb_high']
        orb_low = metrics['orb_low']
        orb_end = metrics['orb_end']
        
        # Breakout thresholds
        high_threshold = orb_high * (1 + config.breakout_threshold)
        low_threshold = orb_low * (1 - config.breakout_threshold)
        
        post_orb_data = df[df['timestamp'] > orb_end]
        
        for _, row in post_orb_data.iterrows():
            current_price = row['close']
            current_high = row['high']
            current_low = row['low']
            
            # Check for high breakout
            if current_high > high_threshold:
                breakout_pct = ((current_high - orb_high) / orb_high * 100)
                breakouts.append({
                    'timestamp': row['timestamp'],
                    'type': 'HIGH',
                    'price': current_high,
                    'breakout_pct': breakout_pct,
                    'volume': row['volume']
                })
            
            # Check for low breakout
            if current_low < low_threshold:
                breakout_pct = ((orb_low - current_low) / orb_low * 100)
                breakouts.append({
                    'timestamp': row['timestamp'],
                    'type': 'LOW',
                    'price': current_low,
                    'breakout_pct': breakout_pct,
                    'volume': row['volume']
                })
        
        return breakouts
    
    def print_analysis(self, df: pd.DataFrame, metrics: dict, breakouts: list):
        """Print comprehensive analysis results."""
        
        print()
        print("=" * 80)
        print(f"📊 ORB ANALYSIS FOR {self.symbol}")
        print("=" * 80)
        print(f"Data Source: {self.csv_file}")
        print(f"Total Data Points: {len(df)}")
        print()
        
        # ORB Period Analysis
        print("=" * 80)
        print(f"📈 OPENING RANGE ANALYSIS (First {config.orb_period_minutes} minutes)")
        print("=" * 80)
        
        et_tz = pytz.timezone('US/Eastern')
        orb_start_et = metrics['orb_start'].astimezone(et_tz)
        orb_end_et = metrics['orb_end'].astimezone(et_tz)
        
        print(f"ORB Period: {orb_start_et.strftime('%H:%M')} - {orb_end_et.strftime('%H:%M')} ET")
        print(f"ORB Data Points: {metrics['orb_data_points']}")
        print(f"ORB High: ${metrics['orb_high']:.3f}")
        print(f"ORB Low: ${metrics['orb_low']:.3f}")
        print(f"ORB Range: ${metrics['orb_range']:.3f} ({metrics['orb_range_pct']:.2f}%)")
        print(f"ORB Midpoint: ${metrics['orb_midpoint']:.3f}")
        print(f"ORB Volume: {metrics['orb_volume']:,}")
        print(f"ORB Avg Volume/Min: {metrics['orb_avg_volume']:.0f}")
        print()
        
        # Breakout Analysis
        print("=" * 80)
        print("🎯 BREAKOUT ANALYSIS")
        print("=" * 80)
        
        high_threshold = metrics['orb_high'] * (1 + config.breakout_threshold)
        low_threshold = metrics['orb_low'] * (1 - config.breakout_threshold)
        
        print(f"High Breakout Threshold: ${high_threshold:.3f} (+{config.breakout_threshold*100:.1f}%)")
        print(f"Low Breakout Threshold: ${low_threshold:.3f} (-{config.breakout_threshold*100:.1f}%)")
        print()
        
        if breakouts:
            print(f"🚨 {len(breakouts)} BREAKOUT(S) DETECTED:")
            for breakout in breakouts:
                timestamp_et = breakout['timestamp'].astimezone(et_tz)
                print(f"  {timestamp_et.strftime('%H:%M:%S')} ET: {breakout['type']} BREAKOUT")
                print(f"    Price: ${breakout['price']:.3f} ({breakout['breakout_pct']:+.2f}%)")
                print(f"    Volume: {breakout['volume']:,}")
        else:
            print("⚪ No breakouts detected during the session")
        
        print()
        
        # Summary Statistics
        print("=" * 80)
        print("📋 SESSION SUMMARY")
        print("=" * 80)
        print(f"Post-ORB High: ${metrics['post_orb_high']:.3f}")
        print(f"Post-ORB Low: ${metrics['post_orb_low']:.3f}")
        print(f"Final Price: ${metrics['final_price']:.3f}")
        print(f"Max High Breakout: {metrics['max_high_breakout_pct']:.2f}% above ORB high")
        print(f"Max Low Breakout: {metrics['max_low_breakout_pct']:.2f}% below ORB low")
        
        # Trading Assessment
        print()
        print("=" * 80)
        print("💡 TRADING ASSESSMENT")
        print("=" * 80)
        
        # Assess ORB quality
        if metrics['orb_range_pct'] > 5:
            range_quality = "🟢 GOOD"
        elif metrics['orb_range_pct'] > 2:
            range_quality = "🟡 MODERATE"
        else:
            range_quality = "🔴 NARROW"
        
        print(f"ORB Range Quality: {range_quality} ({metrics['orb_range_pct']:.2f}%)")
        
        # Assess breakout potential
        if len(breakouts) > 0:
            max_breakout = max(breakouts, key=lambda x: x['breakout_pct'])
            if max_breakout['breakout_pct'] > 10:
                breakout_quality = "🟢 STRONG"
            elif max_breakout['breakout_pct'] > 5:
                breakout_quality = "🟡 MODERATE"
            else:
                breakout_quality = "🔴 WEAK"
            print(f"Breakout Strength: {breakout_quality} (Max: {max_breakout['breakout_pct']:.2f}%)")
        else:
            print("Breakout Strength: ⚪ NO BREAKOUTS")
        
        # Volume assessment
        if metrics['orb_avg_volume'] > 1000:
            volume_quality = "🟢 HIGH"
        elif metrics['orb_avg_volume'] > 500:
            volume_quality = "🟡 MODERATE"
        else:
            volume_quality = "🔴 LOW"
        
        print(f"Volume Quality: {volume_quality} (Avg: {metrics['orb_avg_volume']:.0f}/min)")
        
        # Overall assessment
        print()
        if len(breakouts) > 0 and metrics['orb_range_pct'] > 3 and metrics['orb_avg_volume'] > 500:
            print("🎯 OVERALL: EXCELLENT ORB TRADING OPPORTUNITY")
        elif len(breakouts) > 0 and metrics['orb_range_pct'] > 2:
            print("👍 OVERALL: GOOD ORB TRADING OPPORTUNITY")  
        elif metrics['orb_range_pct'] > 5:
            print("⚠️  OVERALL: POTENTIAL SETUP (Watch for breakouts)")
        else:
            print("❌ OVERALL: POOR ORB SETUP")
    
    def analyze(self) -> dict:
        """Run complete ORB analysis."""
        try:
            # Load data
            df = self.load_data()
            
            # Calculate metrics
            metrics = self.calculate_orb_metrics(df)
            if 'error' in metrics:
                print(f"❌ {metrics['error']}")
                return metrics
            
            # Detect breakouts
            breakouts = self.detect_breakouts(df, metrics)
            
            # Print analysis
            self.print_analysis(df, metrics, breakouts)
            
            return {
                'symbol': self.symbol,
                'metrics': metrics,
                'breakouts': breakouts,
                'data_points': len(df)
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {'error': str(e)}

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze historical data for ORB patterns')
    parser.add_argument('symbol', help='Trading symbol to analyze (e.g., BSLK, AAPL)')
    parser.add_argument('csv_file', nargs='?', help='Path to CSV file (optional, will auto-detect)')
    parser.add_argument('--orb-minutes', type=int, default=15, help='ORB period in minutes (default: 15)')
    
    args = parser.parse_args()
    
    print(f"🔍 ORB ALERT ANALYZER")
    print(f"Analyzing symbol: {args.symbol.upper()}")
    
    try:
        analyzer = ORBAnalyzer(args.symbol, args.csv_file)
        results = analyzer.analyze()
        
        if 'error' not in results:
            print(f"\n✅ Analysis completed successfully for {results['symbol']}")
        
    except Exception as e:
        print(f"❌ Failed to analyze {args.symbol}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()