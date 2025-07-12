#!/usr/bin/env python3
"""
Test Harness for Enhanced Real-Time ORB Alert System

This test harness simulates real-time data feed using historical FTFT data from 2025-07-11
to test the enhanced ORB alert system and capture alert timing for visualization.

Features:
- Simulates real-time market data feed from historical CSV files
- Drives the enhanced ORB alert system with realistic timing
- Captures alert generation with precise timestamps
- Creates candlestick chart with alert timing visualization
- Saves alerts to JSON for analysis
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json
import glob
from datetime import datetime, timedelta, time
import pytz
import sys
import logging
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

# Import the enhanced system components
from atoms.websocket.alpaca_stream import MarketData
from orb_alerts_enhanced_realtime import EnhancedORBAlertSystem, EnhancedORBAlert


class MarketDataSimulator:
    """Simulates real-time market data feed from historical CSV files."""
    
    def __init__(self, symbol: str, date: str):
        self.symbol = symbol
        self.date = date
        self.data = self._load_historical_data()
        self.callbacks = []
        
    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical data for the symbol and date."""
        data_files = glob.glob(f"historical_data/{self.date}/market_data/{self.symbol}_*.csv")
        
        all_data = []
        for file in data_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not all_data:
            raise ValueError(f"No historical data found for {self.symbol} on {self.date}")
        
        # Combine and sort data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
        combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
        combined_data = combined_data.drop_duplicates(subset=['timestamp'])
        
        print(f"📊 Loaded {len(combined_data)} data points for {self.symbol} on {self.date}")
        
        return combined_data
    
    def add_callback(self, callback):
        """Add callback for market data events."""
        self.callbacks.append(callback)
    
    async def simulate_realtime_feed(self, speed_multiplier: float = 1.0):
        """
        Simulate real-time data feed by replaying historical data.
        
        Args:
            speed_multiplier: Speed up factor (1.0 = real-time, 10.0 = 10x faster)
        """
        if self.data.empty:
            print(f"❌ No data to simulate for {self.symbol}")
            return
        
        print(f"🚀 Starting market data simulation for {self.symbol}")
        print(f"   📅 Date: {self.date}")
        print(f"   ⚡ Speed: {speed_multiplier}x real-time")
        print(f"   🕐 Duration: {len(self.data)} data points")
        
        start_time = self.data.iloc[0]['timestamp']
        
        prev_timestamp = None
        for i, row in self.data.iterrows():
            # Calculate time delay to simulate real-time
            if prev_timestamp is not None:
                current_time = row['timestamp']
                time_diff = (current_time - prev_timestamp).total_seconds()
                
                # Apply speed multiplier and wait
                wait_time = time_diff / speed_multiplier
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            prev_timestamp = row['timestamp']
            
            # Create MarketData object
            market_data = MarketData(
                symbol=self.symbol,
                timestamp=row['timestamp'],
                price=row['close'],
                volume=row['volume'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                trade_count=row.get('trade_count', 1),
                vwap=row.get('vwap', row['close'])
            )
            
            # Call all callbacks with market data
            for callback in self.callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    print(f"Error in callback: {e}")
            
            # Progress indicator
            if i % 50 == 0:
                progress = (i / len(self.data)) * 100
                current_time_str = row['timestamp'].strftime('%H:%M:%S')
                print(f"   📈 Progress: {progress:.1f}% - Time: {current_time_str}")
        
        print(f"✅ Market data simulation complete for {self.symbol}")


class EnhancedAlertTestHarness:
    """Test harness for the enhanced real-time ORB alert system."""
    
    def __init__(self, symbol: str, date: str):
        self.symbol = symbol
        self.date = date
        self.alerts_captured = []
        self.market_data_history = []
        self.test_results = {}
        
        # Setup test environment
        self._setup_test_environment()
        
        # Create market data simulator
        self.simulator = MarketDataSimulator(symbol, date)
        
        # Create mock enhanced ORB alert system for testing
        self.alert_system = self._create_mock_alert_system()
        
    def _setup_test_environment(self):
        """Setup test environment and directories."""
        # Create test results directory
        self.test_dir = Path("test_results") / f"enhanced_alerts_{self.date}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for test
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.test_dir / "test_log.txt"),
                logging.StreamHandler()
            ]
        )
        
        print(f"🧪 Test environment setup for {self.symbol} on {self.date}")
        print(f"   📁 Test results will be saved to: {self.test_dir}")
    
    def _create_mock_alert_system(self):
        """Create a simplified mock alert system for testing."""
        
        class MockEnhancedSystem:
            def __init__(self, test_harness):
                self.test_harness = test_harness
                self.orb_data_cache = {}
                self.alerts_generated = {}
                
                # PCA-derived thresholds
                self.volume_ratio_threshold = 2.5
                self.duration_threshold = 10
                self.momentum_threshold = -0.01
                self.range_pct_min = 5.0
                self.range_pct_max = 35.0
                self.min_orb_samples = 3
                self.orb_period_minutes = 15
                
                print(f"🎯 Mock Enhanced Alert System initialized")
                print(f"   📊 PCA Filters: Vol>{self.volume_ratio_threshold}x, Duration>{self.duration_threshold}min, "
                      f"Momentum>{self.momentum_threshold}, Range:{self.range_pct_min}-{self.range_pct_max}%")
            
            def process_market_data(self, market_data: MarketData):
                """Process market data and check for alerts."""
                # Store market data for later analysis
                self.test_harness.market_data_history.append({
                    'timestamp': market_data.timestamp,
                    'symbol': market_data.symbol,
                    'high': market_data.high,
                    'low': market_data.low,
                    'close': market_data.close,
                    'volume': market_data.volume,
                    'vwap': market_data.vwap
                })
                
                # Determine if we're in ORB period or post-ORB
                et_tz = pytz.timezone('US/Eastern')
                current_time = market_data.timestamp.replace(tzinfo=et_tz)
                
                # Market open for the test day
                market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
                orb_end_time = market_open + timedelta(minutes=self.orb_period_minutes)
                
                symbol = market_data.symbol
                
                # During ORB period: collect data
                if market_open <= current_time <= orb_end_time:
                    self._collect_orb_data(symbol, market_data)
                
                # After ORB period: check for breakouts
                elif current_time > orb_end_time:
                    self._check_enhanced_breakout(symbol, market_data, current_time)
            
            def _collect_orb_data(self, symbol: str, market_data: MarketData):
                """Collect data during ORB period."""
                if symbol not in self.orb_data_cache:
                    self.orb_data_cache[symbol] = []
                
                self.orb_data_cache[symbol].append({
                    'timestamp': market_data.timestamp,
                    'high': market_data.high,
                    'low': market_data.low,
                    'close': market_data.close,
                    'volume': market_data.volume,
                    'vwap': market_data.vwap
                })
            
            def _check_enhanced_breakout(self, symbol: str, market_data: MarketData, current_time: datetime):
                """Check for enhanced breakouts."""
                # Skip if already alerted
                if symbol in self.alerts_generated:
                    return
                
                # Skip if no ORB data
                if symbol not in self.orb_data_cache or len(self.orb_data_cache[symbol]) < self.min_orb_samples:
                    return
                
                # Calculate ORB features
                orb_features = self._calculate_orb_features(symbol)
                if not orb_features:
                    return
                
                # Apply PCA filters
                if not self._passes_pca_filters(orb_features):
                    return
                
                # Check for breakout
                current_price = market_data.close
                orb_high = orb_features['orb_high']
                orb_low = orb_features['orb_low']
                
                alert = None
                
                # Bullish breakout
                if current_price > orb_high:
                    alert = self._create_bullish_alert(symbol, current_time, current_price, orb_features)
                
                # Bearish breakdown
                elif current_price < orb_low:
                    alert = self._create_bearish_alert(symbol, current_time, current_price, orb_features)
                
                if alert:
                    self.test_harness._capture_alert(alert)
                    self.alerts_generated[symbol] = current_time
                    print(f"🚨 ALERT GENERATED: {alert.alert_message}")
            
            def _calculate_orb_features(self, symbol: str) -> Optional[Dict[str, Any]]:
                """Calculate ORB features."""
                orb_data = self.orb_data_cache[symbol]
                df = pd.DataFrame(orb_data)
                
                # Basic ORB calculations
                orb_high = df['high'].max()
                orb_low = df['low'].min()
                orb_range = orb_high - orb_low
                orb_midpoint = (orb_high + orb_low) / 2
                
                # Enhanced features
                orb_volume = df['volume'].sum()
                orb_avg_volume = df['volume'].mean()
                orb_volume_ratio = orb_volume / orb_avg_volume if orb_avg_volume > 0 else 1.0
                
                orb_open = df.iloc[0]['close']
                orb_close = df.iloc[-1]['close']
                orb_price_change = orb_close - orb_open
                orb_duration_minutes = len(df)
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
                    'orb_close': orb_close
                }
            
            def _passes_pca_filters(self, orb_features: Dict[str, Any]) -> bool:
                """Apply PCA filters."""
                if orb_features['orb_volume_ratio'] < self.volume_ratio_threshold:
                    return False
                if orb_features['orb_duration_minutes'] < self.duration_threshold:
                    return False
                if orb_features['orb_momentum'] < self.momentum_threshold:
                    return False
                range_pct = orb_features['orb_range_pct']
                if range_pct < self.range_pct_min or range_pct > self.range_pct_max:
                    return False
                return True
            
            def _create_bullish_alert(self, symbol: str, timestamp: datetime, current_price: float, orb_features: Dict[str, Any]) -> EnhancedORBAlert:
                """Create bullish alert."""
                entry_price = orb_features['orb_high'] * 1.002
                stop_loss = orb_features['orb_low'] * 0.995
                target = entry_price + (orb_features['orb_range'] * 1.5)
                
                confidence = self._calculate_confidence(orb_features, 'bullish')
                expected_return = ((target - entry_price) / entry_price) * 100
                reasoning = (f"PCA-Enhanced: Vol Ratio {orb_features['orb_volume_ratio']:.1f}x, "
                           f"Momentum {orb_features['orb_momentum']:.3f}, "
                           f"Range {orb_features['orb_range_pct']:.1f}%")
                
                return EnhancedORBAlert(
                    symbol=symbol,
                    timestamp=timestamp,
                    alert_type="ENHANCED_BULLISH_BREAKOUT",
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    volume_ratio=orb_features['orb_volume_ratio'],
                    momentum=orb_features['orb_momentum'],
                    range_pct=orb_features['orb_range_pct'],
                    expected_return=expected_return,
                    reasoning=reasoning,
                    orb_features=orb_features
                )
            
            def _create_bearish_alert(self, symbol: str, timestamp: datetime, current_price: float, orb_features: Dict[str, Any]) -> EnhancedORBAlert:
                """Create bearish alert."""
                entry_price = orb_features['orb_low'] * 0.998
                stop_loss = orb_features['orb_high'] * 1.005
                target = entry_price - (orb_features['orb_range'] * 1.5)
                
                confidence = self._calculate_confidence(orb_features, 'bearish')
                expected_return = ((entry_price - target) / entry_price) * 100
                reasoning = (f"PCA-Enhanced: Vol Ratio {orb_features['orb_volume_ratio']:.1f}x, "
                           f"Momentum {orb_features['orb_momentum']:.3f}, "
                           f"Range {orb_features['orb_range_pct']:.1f}%")
                
                return EnhancedORBAlert(
                    symbol=symbol,
                    timestamp=timestamp,
                    alert_type="ENHANCED_BEARISH_BREAKDOWN",
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    volume_ratio=orb_features['orb_volume_ratio'],
                    momentum=orb_features['orb_momentum'],
                    range_pct=orb_features['orb_range_pct'],
                    expected_return=expected_return,
                    reasoning=reasoning,
                    orb_features=orb_features
                )
            
            def _calculate_confidence(self, orb_features: Dict[str, Any], direction: str) -> float:
                """Calculate confidence score."""
                confidence = 0.5
                
                vol_ratio = orb_features['orb_volume_ratio']
                if vol_ratio > 5.0:
                    confidence += 0.3
                elif vol_ratio > 3.0:
                    confidence += 0.2
                elif vol_ratio > 2.5:
                    confidence += 0.1
                
                range_pct = orb_features['orb_range_pct']
                if 15.0 <= range_pct <= 25.0:
                    confidence += 0.15
                elif 10.0 <= range_pct <= 30.0:
                    confidence += 0.1
                
                momentum = orb_features['orb_momentum']
                if direction == 'bullish' and momentum > 0:
                    confidence += 0.1
                elif direction == 'bearish' and momentum < 0:
                    confidence += 0.1
                
                duration = orb_features['orb_duration_minutes']
                if duration >= 15:
                    confidence += 0.05
                
                return min(confidence, 1.0)
        
        return MockEnhancedSystem(self)
    
    def _capture_alert(self, alert: EnhancedORBAlert):
        """Capture generated alert for analysis."""
        self.alerts_captured.append(alert)
        
        # Log alert details
        logging.info(f"Alert captured: {alert.symbol} - {alert.alert_type} at {alert.timestamp}")
        logging.info(f"Confidence: {alert.confidence:.1%}, Entry: ${alert.entry_price:.3f}")
    
    async def run_test(self, speed_multiplier: float = 10.0):
        """Run the test harness."""
        print(f"\n🧪 Starting Enhanced ORB Alert Test Harness")
        print(f"Symbol: {self.symbol}, Date: {self.date}")
        print(f"Speed: {speed_multiplier}x real-time")
        print("="*60)
        
        # Setup callback
        self.simulator.add_callback(self.alert_system.process_market_data)
        
        # Run simulation
        await self.simulator.simulate_realtime_feed(speed_multiplier)
        
        # Analyze results
        self._analyze_results()
        
        # Save results
        self._save_results()
        
        # Create visualization
        self._create_alert_visualization()
        
        print(f"\n✅ Test completed successfully!")
        print(f"   📊 Alerts generated: {len(self.alerts_captured)}")
        print(f"   📁 Results saved to: {self.test_dir}")
    
    def _analyze_results(self):
        """Analyze test results."""
        self.test_results = {
            'symbol': self.symbol,
            'date': self.date,
            'total_alerts': len(self.alerts_captured),
            'total_data_points': len(self.market_data_history),
            'alerts_summary': []
        }
        
        for alert in self.alerts_captured:
            self.test_results['alerts_summary'].append({
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'confidence': alert.confidence,
                'entry_price': alert.entry_price,
                'volume_ratio': alert.volume_ratio,
                'range_pct': alert.range_pct,
                'reasoning': alert.reasoning
            })
        
        print(f"\n📊 Test Results Analysis:")
        print(f"   Total market data points processed: {self.test_results['total_data_points']}")
        print(f"   Total alerts generated: {self.test_results['total_alerts']}")
        
        if self.alerts_captured:
            avg_confidence = sum(alert.confidence for alert in self.alerts_captured) / len(self.alerts_captured)
            print(f"   Average confidence: {avg_confidence:.1%}")
            
            for i, alert in enumerate(self.alerts_captured):
                alert_time = alert.timestamp.strftime('%H:%M:%S')
                print(f"   Alert {i+1}: {alert.alert_type} at {alert_time} - Conf: {alert.confidence:.0%}")
    
    def _save_results(self):
        """Save test results to files."""
        # Save alerts as JSON
        alerts_file = self.test_dir / "captured_alerts.json"
        with open(alerts_file, 'w') as f:
            json.dump([alert.to_dict() for alert in self.alerts_captured], f, indent=2)
        
        # Save test results
        results_file = self.test_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save market data history
        market_data_file = self.test_dir / "market_data_history.csv"
        pd.DataFrame(self.market_data_history).to_csv(market_data_file, index=False)
        
        print(f"💾 Results saved:")
        print(f"   Alerts: {alerts_file}")
        print(f"   Summary: {results_file}")
        print(f"   Market data: {market_data_file}")
    
    def _create_alert_visualization(self):
        """Create candlestick chart with alert timing visualization."""
        if not self.market_data_history:
            print("❌ No market data to visualize")
            return
        
        # Convert market data to DataFrame
        df = pd.DataFrame(self.market_data_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        # Plot candlesticks (simplified)
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
                       width=pd.Timedelta(minutes=1), 
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add ORB levels if we have alerts
        if self.alerts_captured:
            # Use first alert's ORB features
            orb_features = self.alerts_captured[0].orb_features
            orb_high = orb_features['orb_high']
            orb_low = orb_features['orb_low']
            orb_midpoint = orb_features['orb_midpoint']
            
            ax1.axhline(y=orb_high, color='blue', linestyle='--', linewidth=2, 
                       label=f'ORB High: ${orb_high:.3f}', alpha=0.8)
            ax1.axhline(y=orb_low, color='blue', linestyle='--', linewidth=2, 
                       label=f'ORB Low: ${orb_low:.3f}', alpha=0.8)
            ax1.axhline(y=orb_midpoint, color='blue', linestyle=':', linewidth=1, 
                       label=f'ORB Mid: ${orb_midpoint:.3f}', alpha=0.6)
        
        # Add alert timing as vertical bars
        for i, alert in enumerate(self.alerts_captured):
            alert_time = alert.timestamp
            color = 'green' if 'BULLISH' in alert.alert_type else 'red'
            label = f"Alert {i+1}: {alert.alert_type.replace('ENHANCED_', '')}"
            
            ax1.axvline(x=alert_time, color=color, linestyle='-', linewidth=3, 
                       alpha=0.8, label=label)
            
            # Add annotation
            ax1.annotate(f'Alert {i+1}\\n{alert.confidence:.0%}',
                        xy=(alert_time, alert.entry_price),
                        xytext=(alert_time, alert.entry_price + 0.1),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # Format price chart
        ax1.set_title(f'{self.symbol} Enhanced ORB Alert Test - {self.date}\\n'
                     f'Alerts Generated: {len(self.alerts_captured)}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        # Volume chart
        timestamps = df['timestamp']
        volumes = df['volume']
        ax2.bar(timestamps, volumes, width=pd.Timedelta(minutes=1), 
               color='lightblue', alpha=0.7, edgecolor='blue', linewidth=0.5)
        
        # Add alert timing to volume chart
        for alert in self.alerts_captured:
            color = 'green' if 'BULLISH' in alert.alert_type else 'red'
            ax2.axvline(x=alert.timestamp, color=color, linestyle='-', linewidth=3, alpha=0.8)
        
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time (ET)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        plt.tight_layout()
        
        # Save chart
        chart_file = self.test_dir / f"{self.symbol}_enhanced_alerts_test_{self.date.replace('-', '')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        
        pdf_file = self.test_dir / f"{self.symbol}_enhanced_alerts_test_{self.date.replace('-', '')}.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        
        print(f"📊 Alert visualization saved:")
        print(f"   PNG: {chart_file}")
        print(f"   PDF: {pdf_file}")
        
        plt.show()


async def main():
    """Main test runner."""
    symbol = "FTFT"
    date = "2025-07-11"
    
    print("🧪 Enhanced ORB Alert System Test Harness")
    print("="*50)
    
    try:
        # Create and run test harness
        test_harness = EnhancedAlertTestHarness(symbol, date)
        await test_harness.run_test(speed_multiplier=20.0)  # 20x speed for faster testing
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())