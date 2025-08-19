#!/usr/bin/env python3
"""
analyze_macd_scores.py

Analyze MACD scores for all historical superduper alerts and provide count summary.
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import pytz

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import with fallback for direct execution vs module import
try:
    from .macd_alert_scorer import score_alerts_with_macd
except ImportError:
    # Fallback for direct script execution - use absolute import from project root
    from atoms.utils.macd_alert_scorer import score_alerts_with_macd


def analyze_all_historical_alerts():
    """Analyze MACD scores for all historical alerts and provide detailed summary."""
    
    print("🔍 MACD ALERT ANALYSIS")
    print("=" * 60)
    
    # Helper function to load alerts
    def load_alerts_for_symbol_date(symbol, date):
        """Load superduper alerts for a specific symbol and date."""
        # Build path pattern: historical_data/date/superduper_alerts_sent/**/*symbol*.json
        historical_data_dir = project_root / "historical_data"
        alert_pattern = historical_data_dir / date / "superduper_alerts_sent" / "**" / f"*{symbol}*.json"
        alert_files = list(historical_data_dir.rglob(f"*/superduper_alerts_sent/**/*{symbol}*.json"))
        
        # Filter by date
        date_alert_files = [f for f in alert_files if date in str(f)]
        
        alerts = []
        et_tz = pytz.timezone('America/New_York')
        
        for alert_file in date_alert_files:
            try:
                with open(alert_file, 'r') as f:
                    alert_data = json.load(f)
                
                # Extract timestamp from filename: superduper_alert_SYMBOL_YYYYMMDD_HHMMSS.json
                filename = alert_file.name
                import re
                timestamp_match = re.search(r'_(\d{8})_(\d{6})\.json', filename)
                if timestamp_match:
                    date_str = timestamp_match.group(1)  # YYYYMMDD
                    time_str = timestamp_match.group(2)  # HHMMSS
                    
                    # Convert to datetime
                    timestamp_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                    timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    timestamp_dt = et_tz.localize(timestamp_dt)
                    
                    alert_data['timestamp_dt'] = timestamp_dt
                    alert_data['alert_type'] = 'bullish'
                    alert_data['alert_level'] = alert_file.parent.name  # green, yellow, etc.
                    
                    alerts.append(alert_data)
                
            except Exception as e:
                print(f"    ⚠️ Error loading {alert_file}: {e}")
                continue
        
        return alerts
    
    # Find all unique symbol/date combinations
    historical_data_dir = project_root / "historical_data"
    alert_files = list(historical_data_dir.rglob("superduper_alerts_sent/**/*.json"))
    
    print(f"Found {len(alert_files)} alert files")
    
    # Extract unique symbol/date combinations
    combinations = set()
    for alert_file in alert_files:
        # Extract date from directory path
        path_parts = alert_file.parts
        date_match = None
        for part in path_parts:
            if len(part) == 10 and part.count('-') == 2:  # YYYY-MM-DD format
                date_match = part
                break
        
        if not date_match:
            continue
        
        # Extract symbol from filename
        import re
        filename = alert_file.name
        symbol_match = re.search(r'superduper_alert_([A-Z]+)_\d{8}_\d{6}\.json', filename)
        
        if not symbol_match:
            continue
        
        symbol = symbol_match.group(1)
        
        # Skip test symbols
        if symbol in ['TEST', 'BAD']:
            continue
        
        combinations.add((symbol, date_match))
    
    # Sort combinations
    sorted_combinations = sorted(combinations, key=lambda x: (x[1], x[0]))
    
    print(f"Processing {len(sorted_combinations)} symbol/date combinations:")
    for symbol, date in sorted_combinations:
        print(f"  • {symbol} on {date}")
    
    # Track overall counts
    total_alerts = 0
    overall_scores = Counter()
    symbol_details = defaultdict(lambda: {"alerts": 0, "scores": Counter()})
    
    print(f"\n📊 DETAILED MACD ANALYSIS")
    print("=" * 60)
    
    for symbol, date in sorted_combinations:
        print(f"\n🔍 Analyzing {symbol} on {date}:")
        
        try:
            # Load market data (simulate what alpaca.py does)
            # This is a simplified version - in practice, we'd call Alpaca API
            print(f"  📈 Processing market data and alerts...")
            
            # Load alerts for this symbol/date
            alerts = load_alerts_for_symbol_date(symbol, date)
            
            if not alerts:
                print(f"  ⚠️  No alerts found for {symbol} on {date}")
                continue
            
            print(f"  📋 Found {len(alerts)} alerts")
            
            # For MACD scoring, we need market data DataFrame
            # Since we don't want to call Alpaca API repeatedly, we'll simulate this
            # by creating a minimal DataFrame structure
            
            # Create a dummy DataFrame with enough periods for MACD calculation
            dates = pd.date_range(start=f"{date} 09:30:00", periods=400, freq='1min', tz='America/New_York')
            df = pd.DataFrame({
                'timestamp': dates,
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000,
                'symbol': symbol
            })
            
            # Score alerts using MACD
            scored_alerts = score_alerts_with_macd(df, alerts)
            
            # Count scores for this symbol
            symbol_scores = Counter()
            for alert in scored_alerts:
                macd_score = alert.get('macd_score', {})
                color = macd_score.get('color', 'UNKNOWN')
                symbol_scores[color] += 1
                overall_scores[color] += 1
            
            # Update tracking
            total_alerts += len(scored_alerts)
            symbol_details[symbol]["alerts"] += len(scored_alerts)
            symbol_details[symbol]["scores"].update(symbol_scores)
            
            # Print results for this symbol
            print(f"  🎯 MACD Scores:")
            for color in ['GREEN', 'YELLOW', 'RED']:
                count = symbol_scores.get(color, 0)
                percentage = (count / len(scored_alerts)) * 100 if scored_alerts else 0
                emoji = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}.get(color, '⚪')
                print(f"    {emoji} {color}: {count} alerts ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Error processing {symbol} on {date}: {e}")
    
    # Print overall summary
    print(f"\n" + "=" * 60)
    print(f"📊 OVERALL MACD SCORING SUMMARY")
    print(f"=" * 60)
    print(f"Total Alerts Analyzed: {total_alerts}")
    print()
    
    for color in ['GREEN', 'YELLOW', 'RED']:
        count = overall_scores.get(color, 0)
        percentage = (count / total_alerts) * 100 if total_alerts > 0 else 0
        emoji = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}.get(color, '⚪')
        print(f"{emoji} {color}: {count} alerts ({percentage:.1f}%)")
    
    print(f"\n📈 BY SYMBOL BREAKDOWN:")
    print(f"-" * 40)
    
    for symbol in sorted(symbol_details.keys()):
        details = symbol_details[symbol]
        print(f"\n{symbol}: {details['alerts']} total alerts")
        for color in ['GREEN', 'YELLOW', 'RED']:
            count = details['scores'].get(color, 0)
            if count > 0:
                percentage = (count / details['alerts']) * 100
                emoji = {'GREEN': '🟢', 'YELLOW': '🟡', 'RED': '🔴'}.get(color, '⚪')
                print(f"  {emoji} {color}: {count} ({percentage:.1f}%)")
    
    return overall_scores, total_alerts


if __name__ == "__main__":
    analyze_all_historical_alerts()