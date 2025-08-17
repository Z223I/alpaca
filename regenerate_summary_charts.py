#!/usr/bin/env python3
"""
Standalone script to regenerate summary charts from existing backtesting run data.
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def analyze_run_results(run_dir: Path):
    """Analyze results from a completed run (same logic as backtesting.py)"""
    results = {
        'superduper_alerts': 0,
        'trades': 0,
        'symbols_processed': 0,
        'alerts_by_symbol': {},
        'trades_by_symbol': {}
    }

    # Count superduper alerts from sent directory
    superduper_dir = run_dir / "historical_data"
    if superduper_dir.exists():
        for alert_file in superduper_dir.rglob("**/superduper_alerts_sent/**/*.json"):
            try:
                with open(alert_file, 'r') as f:
                    alert_data = json.load(f)
                    if isinstance(alert_data, list):
                        results['superduper_alerts'] += len(alert_data)
                        for alert in alert_data:
                            symbol = alert.get('symbol', 'UNKNOWN')
                            results['alerts_by_symbol'][symbol] = results['alerts_by_symbol'].get(symbol, 0) + 1
                    else:
                        results['superduper_alerts'] += 1
                        symbol = alert_data.get('symbol', 'UNKNOWN')
                        results['alerts_by_symbol'][symbol] = results['alerts_by_symbol'].get(symbol, 0) + 1
            except Exception:
                pass

    # Count trades (placeholder - would need actual trade files)
    trades_dir = run_dir / "historical_data"
    if trades_dir.exists():
        for trade_file in trades_dir.rglob("*trades*.json"):
            try:
                with open(trade_file, 'r') as f:
                    trade_data = json.load(f)
                    if isinstance(trade_data, list):
                        results['trades'] += len(trade_data)
                    else:
                        results['trades'] += 1
            except Exception:
                pass

    return results

def extract_parameters_from_run_name(run_name):
    """Extract timeframe and threshold from run directory name."""
    import re
    pattern = r'run_(\d{4}-\d{2}-\d{2})_tf(\d+)_th([\d.]+)_([a-f0-9]+)'
    match = re.match(pattern, run_name)
    if match:
        return {
            'date': match.group(1),
            'timeframe': int(match.group(2)),
            'threshold': float(match.group(3)),
            'uuid': match.group(4)
        }
    return None

def collect_run_data():
    """Collect data from all existing runs"""
    run_results = []
    runs_dir = Path("runs")
    
    # Scan for run directories
    for symbol_dir in runs_dir.glob("*"):
        if not symbol_dir.is_dir() or symbol_dir.name.startswith("analysis") or symbol_dir.name.startswith("summary"):
            continue
            
        for run_dir in symbol_dir.glob("run_*"):
            if not run_dir.is_dir():
                continue
                
            run_name = run_dir.name
            print(f"📊 Analyzing {symbol_dir.name}/{run_name}...")
            
            # Extract parameters
            params = extract_parameters_from_run_name(run_name)
            if not params:
                print(f"   ⚠️ Could not extract parameters from {run_name}")
                continue
            
            # Analyze results
            try:
                results = analyze_run_results(run_dir)
                results.update({
                    'date': params['date'],
                    'timeframe': params['timeframe'],
                    'threshold': params['threshold'],
                    'symbol': symbol_dir.name,
                    'run_dir': str(run_dir)
                })
                run_results.append(results)
                print(f"   ✅ Found {results['superduper_alerts']} alerts, {results['trades']} trades")
                
            except Exception as e:
                print(f"   ⚠️ Error analyzing {run_name}: {e}")
                continue
    
    return run_results

def create_summary_charts(run_results):
    """Create summary charts from run results"""
    if not run_results:
        print("❌ No run results found")
        return
    
    # Prepare data for charts
    alerts_by_date = {}
    trades_by_date = {}
    alerts_by_symbol = {}

    for result in run_results:
        date = result['date']
        symbol = result.get('symbol', 'UNKNOWN')
        alerts_by_date[date] = alerts_by_date.get(date, 0) + result['superduper_alerts']
        trades_by_date[date] = trades_by_date.get(date, 0) + result['trades']
        
        # Use only the main superduper_alerts count (don't double-count from alerts_by_symbol)
        alerts_by_symbol[symbol] = alerts_by_symbol.get(symbol, 0) + result['superduper_alerts']

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create bar chart by symbol
    if alerts_by_symbol:
        plt.figure(figsize=(12, 8))
        symbols = list(alerts_by_symbol.keys())
        alert_counts = list(alerts_by_symbol.values())
        
        bars = plt.bar(symbols, alert_counts, color='steelblue', alpha=0.8)
        plt.title('Superduper Alerts by Symbol', fontsize=16, fontweight='bold')
        plt.suptitle('Data Source: REAL Backtesting Data - Actual superduper_alerts_sent Files', fontsize=12, y=0.02, color='darkgreen', weight='bold')
        plt.xlabel('Symbol', fontsize=12)
        plt.ylabel('Number of Alerts', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, alert_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"runs/summary_alerts_by_symbol_bar_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Created: {filename}")

    # Create pie chart by symbol if we have multiple symbols
    if len(alerts_by_symbol) > 1:
        plt.figure(figsize=(10, 8))
        symbols = list(alerts_by_symbol.keys())
        alert_counts = list(alerts_by_symbol.values())
        
        plt.pie(alert_counts, labels=symbols, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution of Superduper Alerts by Symbol', fontsize=16, fontweight='bold')
        plt.suptitle('Data Source: REAL Backtesting Data - Actual superduper_alerts_sent Files', fontsize=12, y=0.02, color='darkgreen', weight='bold')
        
        filename = f"runs/summary_alerts_by_symbol_pie_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🥧 Created: {filename}")

    # Print summary
    total_alerts = sum(alert_counts) if 'alert_counts' in locals() else 0
    total_trades = sum(trades_by_date.values())
    print(f"\n📈 Summary: {len(run_results)} runs, {total_alerts} total alerts, {total_trades} total trades")

def main():
    print("🔄 Regenerating summary charts from existing backtesting data...")
    
    # Collect data from existing runs
    run_results = collect_run_data()
    
    if not run_results:
        print("❌ No valid run data found")
        return
    
    # Create summary charts
    create_summary_charts(run_results)
    
    print("✅ Summary chart regeneration complete!")

if __name__ == "__main__":
    main()