#!/usr/bin/env python3
"""
Ultra-Relaxed VWAP Analysis

Test with very relaxed criteria to see if ANY bounce-like patterns exist.

ANALYSIS RESULTS (2025-08-27):
- Found 12 bounce patterns with relaxed criteria (30% VWAP coverage + both green + second higher)
- ALL patterns had NEGATIVE distances from VWAP (-0.19% to -7.09%)
- Strong gains observed: DEVS (12.03% + 5.38%), FLNT (5.50% + 7.15%), INHD (4.56% + 5.91%)

CRITICAL FINDINGS:
🚨 The original VWAP bounce logic has the distance requirement BACKWARDS:
- Current logic: expects prices 0-7% ABOVE VWAP (first_distance >= 0)
- Reality: bounce patterns occur when prices are BELOW VWAP and bouncing upward

RECOMMENDATIONS:
1. Fix the distance check in vwap_bounce_alerts.py:
   Change: first_within_7_percent = (0 <= first_distance <= 7)  # ABOVE VWAP
   To: first_within_7_percent = (-7 <= first_distance <= 0)     # BELOW VWAP

2. Reduce VWAP coverage requirement from 100% to 30-50% of 30-minute candles

3. With corrected logic, expect several alerts per day with strong volume confirmation

The VWAP bounce concept is viable, but current implementation looks for bounces 
in the wrong direction relative to VWAP.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ultra_relaxed_analysis():
    """Test with very relaxed criteria."""
    
    print(f"🧪 Ultra-Relaxed VWAP Pattern Detection")
    print(f"📋 Criteria: 30% VWAP coverage + both green + second higher (NO distance limit)")
    print("=" * 80)
    
    # Test just one recent date for speed
    historical_data_dir = Path("historical_data")
    test_date = "2025-08-27"
    market_data_dir = historical_data_dir / test_date / "market_data"
    
    if not market_data_dir.exists():
        print(f"❌ No data for {test_date}")
        return
        
    csv_files = list(market_data_dir.glob("*.csv"))
    print(f"📅 Testing {test_date} with {len(csv_files)} files")
    
    alerts = []
    total_checks = 0
    
    for csv_file in csv_files:
        try:
            symbol = csv_file.name.split('_')[0]
            df = pd.read_csv(csv_file)
            
            if len(df) < 40:
                continue
                
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Test every 10 minutes
            for i in range(40, len(df), 10):
                total_checks += 1
                current_data = df.iloc[:i+1].copy()
                
                # Relaxed analysis
                analysis_data = current_data.tail(10).reset_index(drop=True)
                history_data = current_data.iloc[-40:-10].reset_index(drop=True)
                
                # Check 1: 30% VWAP coverage
                above_vwap = (history_data['close'] > history_data['vwap']).sum()
                vwap_pct = (above_vwap / 30) * 100
                
                if vwap_pct < 30:
                    continue
                
                # Create 5-minute candles
                first_5min = {
                    'open': analysis_data.iloc[0]['open'],
                    'high': analysis_data.iloc[:5]['high'].max(),
                    'low': analysis_data.iloc[:5]['low'].min(),
                    'close': analysis_data.iloc[4]['close'],
                    'volume': analysis_data.iloc[:5]['volume'].sum()
                }
                
                second_5min = {
                    'open': analysis_data.iloc[5]['open'],
                    'high': analysis_data.iloc[5:]['high'].max(),
                    'low': analysis_data.iloc[5:]['low'].min(),
                    'close': analysis_data.iloc[9]['close'],
                    'volume': analysis_data.iloc[5:]['volume'].sum()
                }
                
                # Check 2: Both green
                first_green = first_5min['close'] > first_5min['open']
                second_green = second_5min['close'] > second_5min['open']
                
                if not (first_green and second_green):
                    continue
                
                # Check 3: Second higher
                if second_5min['close'] <= first_5min['close']:
                    continue
                
                # Found a pattern!
                current_vwap = analysis_data.iloc[-1]['vwap']
                distance_pct = ((first_5min['close'] / current_vwap) - 1) * 100
                
                alerts.append({
                    'symbol': symbol,
                    'minute': i,
                    'vwap_pct': vwap_pct,
                    'distance_pct': distance_pct,
                    'first_gain': ((first_5min['close'] / first_5min['open']) - 1) * 100,
                    'second_gain': ((second_5min['close'] / second_5min['open']) - 1) * 100,
                    'total_volume': first_5min['volume'] + second_5min['volume'],
                    'vwap': current_vwap
                })
                
                break  # One per symbol
                
        except Exception as e:
            continue
    
    print(f"\n📊 ULTRA-RELAXED RESULTS:")
    print(f"⏰ Total checks: {total_checks:,}")
    print(f"🚨 Patterns found: {len(alerts)}")
    
    if alerts:
        print(f"\n✅ Found {len(alerts)} relaxed patterns:")
        for alert in alerts:
            print(f"  🟢 {alert['symbol']} at minute {alert['minute']}")
            print(f"     VWAP coverage: {alert['vwap_pct']:.1f}%, Distance: {alert['distance_pct']:.2f}%")
            print(f"     Gains: +{alert['first_gain']:.2f}%, +{alert['second_gain']:.2f}%")
            print(f"     Volume: {alert['total_volume']:,}")
        
        # Show distance distribution
        distances = [alert['distance_pct'] for alert in alerts]
        print(f"\n📏 Distance from VWAP distribution:")
        print(f"   Min: {min(distances):.2f}%, Max: {max(distances):.2f}%")
        print(f"   Avg: {sum(distances)/len(distances):.2f}%")
        
        # Check how many would pass 7% rule
        within_7_pct = len([d for d in distances if 0 <= d <= 7])
        print(f"   Within 7%: {within_7_pct}/{len(distances)} ({within_7_pct/len(distances)*100:.1f}%)")
        
        if within_7_pct > 0:
            print(f"\n💡 BREAKTHROUGH: {within_7_pct} patterns would pass ALL original criteria!")
        else:
            print(f"\n⚠️  None pass the 7% distance requirement - this is the main blocker")
            
    else:
        print(f"\n❌ Even ultra-relaxed criteria found NO patterns")
        print(f"💭 The VWAP bounce concept may not be viable with current market data")
    
    print("=" * 80)


if __name__ == "__main__":
    ultra_relaxed_analysis()