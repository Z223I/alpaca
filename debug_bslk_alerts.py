#!/usr/bin/env python3
"""
Debug script to understand why BSLK alerts aren't being generated
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config
from atoms.websocket.alpaca_stream import MarketData
from atoms.indicators.orb_calculator import ORBCalculator
from atoms.alerts.breakout_detector import BreakoutDetector
from atoms.alerts.confidence_scorer import ConfidenceScorer
from atoms.alerts.alert_formatter import AlertFormatter

def debug_bslk_alert_generation():
    """Debug why BSLK alerts aren't being generated."""
    
    # Load the CSV data
    csv_file = "historical_data/2025-07-08/market_data/BSLK_20250708_090100.csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} data points from {csv_file}")
        
        # Convert timestamps to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize components
        orb_calculator = ORBCalculator()
        breakout_detector = BreakoutDetector(orb_calculator)
        confidence_scorer = ConfidenceScorer()
        alert_formatter = AlertFormatter()
        
        print(f"\n🔧 CONFIGURATION:")
        print(f"Min confidence score: {config.min_confidence_score}")
        print(f"Breakout threshold: {config.breakout_threshold}")
        print(f"Volume multiplier: {config.volume_multiplier}")
        print(f"Alert window: {config.alert_window_start} - {config.alert_window_end}")
        
        # Step 1: Calculate ORB levels
        print(f"\n📊 STEP 1: Calculate ORB Levels")
        orb_level = orb_calculator.calculate_orb_levels("BSLK", df)
        if orb_level:
            print(f"✅ ORB High: ${orb_level.orb_high:.3f}")
            print(f"✅ ORB Low: ${orb_level.orb_low:.3f}")
            print(f"✅ ORB Range: ${orb_level.orb_range:.3f}")
            print(f"✅ Sample Count: {orb_level.sample_count}")
            
            # Calculate thresholds
            breakout_threshold = orb_level.get_breakout_threshold(config.breakout_threshold)
            breakdown_threshold = orb_level.get_breakdown_threshold(config.breakout_threshold)
            print(f"✅ Breakout threshold: ${breakout_threshold:.3f}")
            print(f"✅ Breakdown threshold: ${breakdown_threshold:.3f}")
        else:
            print("❌ Failed to calculate ORB levels")
            return
        
        # Step 2: Process each data point after ORB period
        print(f"\n🔍 STEP 2: Process Post-ORB Data Points")
        
        # Find ORB end time (first 15 minutes)
        orb_start = df['timestamp'].iloc[0]
        orb_end = orb_start + timedelta(minutes=15)
        post_orb_data = df[df['timestamp'] > orb_end]
        
        alerts_generated = []
        
        for idx, row in post_orb_data.iterrows():
            current_price = row['close']
            timestamp = row['timestamp']
            
            # Convert to ET for display
            et_tz = pytz.timezone('US/Eastern')
            timestamp_et = timestamp.astimezone(et_tz)
            
            print(f"\n⏰ Processing: {timestamp_et.strftime('%H:%M:%S')} ET - Price: ${current_price:.3f}")
            
            # Check if within alert window
            is_within_window = breakout_detector.is_within_alert_window(timestamp)
            print(f"   Within alert window: {is_within_window}")
            
            if not is_within_window:
                print(f"   ❌ Skipping - outside alert window")
                continue
            
            # Calculate volume ratio (use simple average for now)
            volume_ratio = row['volume'] / df['volume'].mean()
            print(f"   Volume ratio: {volume_ratio:.2f}x avg")
            
            # Check for breakout
            breakout_signal = breakout_detector.detect_breakout(
                symbol="BSLK",
                current_price=current_price,
                volume_ratio=volume_ratio,
                timestamp=timestamp
            )
            
            if breakout_signal:
                print(f"   ✅ Breakout detected: {breakout_signal.breakout_type.value}")
                print(f"   ✅ Breakout percentage: {breakout_signal.breakout_percentage:.2f}%")
                print(f"   ✅ Volume ratio: {breakout_signal.volume_ratio:.2f}x")
                
                # Calculate technical indicators
                historical_data = df[df['timestamp'] <= timestamp]
                technical_indicators = breakout_detector.calculate_technical_indicators(historical_data)
                print(f"   Technical indicators: {technical_indicators}")
                
                # Calculate confidence score
                confidence = confidence_scorer.calculate_confidence_score(
                    breakout_signal, technical_indicators
                )
                
                print(f"   📊 Confidence Components:")
                print(f"      PC1 (ORB momentum): {confidence.pc1_score:.3f}")
                print(f"      PC2 (Volume): {confidence.pc2_score:.3f}")
                print(f"      PC3 (Technical): {confidence.pc3_score:.3f}")
                print(f"      Total score: {confidence.total_score:.3f}")
                
                # Check if alert should be generated
                should_generate = confidence_scorer.should_generate_alert(confidence)
                print(f"   Should generate alert: {should_generate}")
                
                if should_generate:
                    # Create alert
                    alert = alert_formatter.create_alert(breakout_signal, confidence)
                    alerts_generated.append(alert)
                    print(f"   🚨 ALERT GENERATED: {alert.symbol} - {alert.priority.value}")
                    print(f"   📄 Alert message: {alert.alert_message}")
                    
                    # Debug output only - no file saving needed
                else:
                    print(f"   ❌ Alert not generated - confidence too low ({confidence.total_score:.3f} < {config.min_confidence_score})")
            else:
                print(f"   ❌ No breakout detected")
        
        print(f"\n📋 FINAL SUMMARY:")
        print(f"Total alerts generated: {len(alerts_generated)}")
        
        if alerts_generated:
            for alert in alerts_generated:
                print(f"  • {alert.symbol} at {alert.timestamp.strftime('%H:%M:%S')} ET: {alert.priority.value} (confidence: {alert.confidence_score:.3f})")
        
        return alerts_generated
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("🔍 DEBUGGING BSLK ALERT GENERATION")
    print("=" * 80)
    alerts = debug_bslk_alert_generation()