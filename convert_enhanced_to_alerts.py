#!/usr/bin/env python3
"""
Convert Enhanced ORB Results to Standard Alert Format

This script converts the enhanced ORB analysis results back into the standard
alert format that can be processed by the existing analysis system.
"""

import json
from pathlib import Path
from datetime import datetime
import pytz

def convert_enhanced_results_to_alerts():
    """Convert enhanced ORB results to standard alert format."""
    
    # Load enhanced results
    with open('enhanced_orb_results.json', 'r') as f:
        enhanced_results = json.load(f)
    
    # Create alerts directories if they don't exist
    for date in enhanced_results['dates_analyzed']:
        date_dir = Path(f"historical_data/{date}/alerts")
        (date_dir / "bullish").mkdir(parents=True, exist_ok=True)
        (date_dir / "bearish").mkdir(parents=True, exist_ok=True)
    
    alert_count = 0
    
    # Process each date's results
    for date, date_results in enhanced_results['results'].items():
        print(f"Processing {date}...")
        
        # Process each symbol's results
        for symbol, symbol_data in date_results.items():
            if not symbol_data.get('alerts'):
                continue
                
            orb_features = symbol_data['orb_features']
            
            # Process each alert for this symbol
            for alert in symbol_data['alerts']:
                alert_type = alert['type']
                
                # Create standard alert format
                standard_alert = {
                    "symbol": symbol,
                    "timestamp": f"{date}T10:00:00",  # Approximate ORB end time
                    "current_price": alert['entry_price'],
                    "orb_high": orb_features['orb_high'],
                    "orb_low": orb_features['orb_low'],
                    "orb_range": orb_features['orb_range'],
                    "orb_midpoint": orb_features['orb_midpoint'],
                    "breakout_type": alert_type.lower().replace('enhanced_', ''),
                    "breakout_percentage": abs(alert['expected_return']),  # Use expected return as breakout %
                    "volume_ratio": alert['volume_ratio'],
                    "confidence_score": alert['confidence'],
                    "priority": "HIGH" if alert['confidence'] > 0.8 else "MEDIUM" if alert['confidence'] > 0.6 else "LOW",
                    "confidence_level": "HIGH" if alert['confidence'] > 0.8 else "MEDIUM" if alert['confidence'] > 0.6 else "LOW",
                    "recommended_stop_loss": alert['stop_loss'],
                    "recommended_take_profit": alert['target'],
                    "alert_message": f"PCA-Enhanced {alert_type.replace('ENHANCED_', '')} for {symbol}",
                    
                    # Enhanced fields
                    "pca_enhanced": True,
                    "orb_momentum": orb_features['orb_momentum'],
                    "orb_range_pct": orb_features['orb_range_pct'],
                    "orb_duration_minutes": orb_features['orb_duration_minutes'],
                    "volume_concentration": orb_features['volume_concentration'],
                    "expected_return": alert['expected_return'],
                    "reasoning": alert['reasoning']
                }
                
                # Determine subdirectory
                if 'bullish' in alert_type.lower():
                    subdir = "bullish"
                elif 'bearish' in alert_type.lower():
                    subdir = "bearish"
                else:
                    continue
                
                # Save alert file
                alert_filename = f"enhanced_alert_{symbol}_{date.replace('-', '')}_{alert_count:03d}.json"
                alert_path = Path(f"historical_data/{date}/alerts/{subdir}/{alert_filename}")
                
                with open(alert_path, 'w') as f:
                    json.dump(standard_alert, f, indent=2)
                
                alert_count += 1
                print(f"  Created {alert_filename}")
    
    print(f"\nConverted {alert_count} enhanced alerts to standard format")
    return alert_count

if __name__ == "__main__":
    convert_enhanced_results_to_alerts()