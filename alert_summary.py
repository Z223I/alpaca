#!/usr/bin/env python3
"""
Alert Summary Script

Analyzes historical alert data and summarizes alerts by stock,
breaking out Medium and High priority alerts.

Usage:
    python3 alert_summary.py                          # Use most recent date
    python3 alert_summary.py --date 2025-07-10        # Use specific date
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_most_recent_date(base_path: Path) -> Optional[str]:
    """Find the most recent date directory in historical_data."""
    historical_path = base_path / "historical_data"
    if not historical_path.exists():
        return None
    
    date_dirs = [d.name for d in historical_path.iterdir() if d.is_dir()]
    if not date_dirs:
        return None
    
    # Sort dates and return the most recent
    date_dirs.sort(reverse=True)
    return date_dirs[0]


def get_alert_files(date: str, base_path: Path) -> List[Path]:
    """Get all alert files for a specific date."""
    alert_files = []
    
    # Check main alerts directory
    alerts_dir = base_path / "alerts"
    if alerts_dir.exists():
        for file_path in alerts_dir.glob(f"*_{date.replace('-', '')}_*.json"):
            alert_files.append(file_path)
    
    # Check bullish alerts subdirectory
    bullish_dir = alerts_dir / "bullish"
    if bullish_dir.exists():
        for file_path in bullish_dir.glob(f"*_{date.replace('-', '')}_*.json"):
            alert_files.append(file_path)
    
    # Check bearish alerts subdirectory
    bearish_dir = alerts_dir / "bearish"
    if bearish_dir.exists():
        for file_path in bearish_dir.glob(f"*_{date.replace('-', '')}_*.json"):
            alert_files.append(file_path)
    
    return sorted(alert_files)


def load_alert_data(file_path: Path) -> Optional[Dict]:
    """Load alert data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None


def categorize_priority(priority: str) -> str:
    """Categorize priority levels for summary."""
    priority_upper = priority.upper()
    if priority_upper == "HIGH":
        return "High"
    elif priority_upper == "MEDIUM":
        return "Medium"
    else:
        return "Low"


def analyze_alerts(alert_files: List[Path]) -> Tuple[Dict, Dict]:
    """Analyze alert files and return summaries by stock and priority."""
    stock_summary = defaultdict(lambda: {
        "total_alerts": 0,
        "bullish_alerts": 0,
        "bearish_alerts": 0,
        "high_priority": 0,
        "medium_priority": 0,
        "low_priority": 0,
        "avg_confidence": 0.0,
        "max_breakout_pct": 0.0,
        "alerts": []
    })
    
    priority_summary = defaultdict(lambda: {
        "count": 0,
        "symbols": set(),
        "avg_confidence": 0.0,
        "bullish_count": 0,
        "bearish_count": 0
    })
    
    total_confidence = defaultdict(float)
    
    for file_path in alert_files:
        alert_data = load_alert_data(file_path)
        if not alert_data:
            continue
        
        symbol = alert_data.get("symbol", "UNKNOWN")
        priority = categorize_priority(alert_data.get("priority", "LOW"))
        breakout_type = alert_data.get("breakout_type", "")
        confidence = alert_data.get("confidence_score", 0.0)
        breakout_pct = abs(alert_data.get("breakout_percentage", 0.0))
        
        # Update stock summary
        stock_data = stock_summary[symbol]
        stock_data["total_alerts"] += 1
        stock_data["alerts"].append(alert_data)
        
        if "bullish" in breakout_type.lower():
            stock_data["bullish_alerts"] += 1
        elif "bearish" in breakout_type.lower():
            stock_data["bearish_alerts"] += 1
        
        if priority == "High":
            stock_data["high_priority"] += 1
        elif priority == "Medium":
            stock_data["medium_priority"] += 1
        else:
            stock_data["low_priority"] += 1
        
        # Track max breakout percentage and accumulate confidence
        if breakout_pct > stock_data["max_breakout_pct"]:
            stock_data["max_breakout_pct"] = breakout_pct
        
        total_confidence[symbol] += confidence
        
        # Update priority summary
        priority_data = priority_summary[priority]
        priority_data["count"] += 1
        priority_data["symbols"].add(symbol)
        priority_data["avg_confidence"] += confidence
        
        if "bullish" in breakout_type.lower():
            priority_data["bullish_count"] += 1
        elif "bearish" in breakout_type.lower():
            priority_data["bearish_count"] += 1
    
    # Calculate average confidence scores
    for symbol, data in stock_summary.items():
        if data["total_alerts"] > 0:
            data["avg_confidence"] = total_confidence[symbol] / data["total_alerts"]
    
    for priority, data in priority_summary.items():
        if data["count"] > 0:
            data["avg_confidence"] = data["avg_confidence"] / data["count"]
    
    return dict(stock_summary), dict(priority_summary)


def print_summary(date: str, stock_summary: Dict, priority_summary: Dict):
    """Print formatted summary of alerts."""
    print(f"\n{'='*60}")
    print(f"ALERT SUMMARY FOR {date}")
    print(f"{'='*60}")
    
    # Overall statistics
    total_alerts = sum(data["total_alerts"] for data in stock_summary.values())
    total_stocks = len(stock_summary)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Alerts: {total_alerts}")
    print(f"  Unique Stocks: {total_stocks}")
    
    # Priority breakdown
    print(f"\nPRIORITY BREAKDOWN:")
    for priority in ["High", "Medium", "Low"]:
        if priority in priority_summary:
            data = priority_summary[priority]
            print(f"  {priority} Priority:")
            print(f"    Count: {data['count']}")
            print(f"    Unique Stocks: {len(data['symbols'])}")
            print(f"    Avg Confidence: {data['avg_confidence']:.3f}")
            print(f"    Bullish: {data['bullish_count']}, Bearish: {data['bearish_count']}")
    
    # Top stocks by alert count
    print(f"\nTOP STOCKS BY ALERT COUNT:")
    sorted_stocks = sorted(stock_summary.items(), 
                          key=lambda x: x[1]["total_alerts"], 
                          reverse=True)
    
    for i, (symbol, data) in enumerate(sorted_stocks[:10]):
        print(f"  {i+1:2}. {symbol}: {data['total_alerts']} alerts "
              f"(High: {data['high_priority']}, Med: {data['medium_priority']}, "
              f"Low: {data['low_priority']}) "
              f"[Bull: {data['bullish_alerts']}, Bear: {data['bearish_alerts']}]")
    
    # Medium and High Priority Focus
    print(f"\n{'='*60}")
    print(f"MEDIUM AND HIGH PRIORITY ALERTS")
    print(f"{'='*60}")
    
    medium_high_stocks = {symbol: data for symbol, data in stock_summary.items() 
                         if data["high_priority"] > 0 or data["medium_priority"] > 0}
    
    if medium_high_stocks:
        sorted_priority_stocks = sorted(medium_high_stocks.items(),
                                      key=lambda x: (x[1]["high_priority"], x[1]["medium_priority"]),
                                      reverse=True)
        
        print(f"\nSTOCKS WITH MEDIUM/HIGH PRIORITY ALERTS:")
        for symbol, data in sorted_priority_stocks:
            print(f"\n{symbol}:")
            print(f"  High Priority: {data['high_priority']}")
            print(f"  Medium Priority: {data['medium_priority']}")
            print(f"  Total Alerts: {data['total_alerts']}")
            print(f"  Bullish/Bearish: {data['bullish_alerts']}/{data['bearish_alerts']}")
            print(f"  Avg Confidence: {data['avg_confidence']:.3f}")
            print(f"  Max Breakout %: {data['max_breakout_pct']:.2f}%")
            
            # Show latest alert details for context
            if data["alerts"]:
                latest_alert = max(data["alerts"], key=lambda x: x.get("timestamp", ""))
                timestamp = latest_alert.get("timestamp", "").split("T")[1][:5] if "T" in latest_alert.get("timestamp", "") else "N/A"
                current_price = latest_alert.get("current_price", 0)
                breakout_type = latest_alert.get("breakout_type", "").replace("_", " ").title()
                print(f"  Latest: {timestamp} - ${current_price:.4f} ({breakout_type})")
    else:
        print("\nNo Medium or High priority alerts found for this date.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and summarize ORB alerts by stock and priority",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 alert_summary.py                    # Use most recent date
  python3 alert_summary.py --date 2025-07-10  # Use specific date
        """
    )
    
    parser.add_argument(
        "--date", 
        type=str, 
        help="Date to analyze (YYYY-MM-DD format). If not provided, uses most recent date."
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Get base path (project root)
    base_path = Path(__file__).parent
    
    # Determine date to analyze
    if args.date:
        target_date = args.date
    else:
        target_date = get_most_recent_date(base_path)
        if not target_date:
            print("Error: No historical data directories found and no date specified.")
            sys.exit(1)
        print(f"Using most recent date: {target_date}")
    
    # Get alert files
    alert_files = get_alert_files(target_date, base_path)
    
    if not alert_files:
        print(f"No alert files found for date: {target_date}")
        sys.exit(1)
    
    print(f"Found {len(alert_files)} alert files for {target_date}")
    
    # Analyze alerts
    stock_summary, priority_summary = analyze_alerts(alert_files)
    
    # Print summary
    print_summary(target_date, stock_summary, priority_summary)


if __name__ == "__main__":
    main()