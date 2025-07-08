#!/usr/bin/env python3
"""
Test runner for ORB analysis on multiple symbols
"""

import os
import sys
import glob
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from orb_analyzer import ORBAnalyzer

def find_all_symbols():
    """Find all available symbols from CSV files."""
    symbols = set()
    
    # Search patterns for CSV files  
    patterns = [
        "../historical_data/*/market_data/*.csv",
        "../tmp/*.csv",
        "../data/*.csv",
        "historical_data/*/market_data/*.csv", 
        "tmp/*.csv",
        "data/*.csv"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            # Extract symbol from filename
            filename = os.path.basename(file)
            # Common patterns: SYMBOL_date.csv, SYMBOL_opening_range.csv, etc.
            if '_' in filename:
                symbol = filename.split('_')[0]
                if symbol.isupper() and len(symbol) <= 5:  # Likely a stock symbol
                    symbols.add(symbol)
    
    return sorted(list(symbols))

def run_batch_analysis(symbols=None, max_symbols=5):
    """Run ORB analysis on multiple symbols."""
    
    if symbols is None:
        symbols = find_all_symbols()
        print(f"🔍 Found {len(symbols)} symbols: {', '.join(symbols)}")
        
        if len(symbols) > max_symbols:
            print(f"⚠️  Limiting to first {max_symbols} symbols (use --all to analyze all)")
            symbols = symbols[:max_symbols]
    
    results = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*80}")
        print(f"📊 ANALYZING {symbol} ({i}/{len(symbols)})")
        print(f"{'='*80}")
        
        try:
            analyzer = ORBAnalyzer(symbol)
            result = analyzer.analyze()
            results[symbol] = result
            
        except Exception as e:
            print(f"❌ Failed to analyze {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 BATCH ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    successful = 0
    with_breakouts = 0
    excellent_setups = 0
    
    for symbol, result in results.items():
        if 'error' in result:
            print(f"❌ {symbol}: {result['error']}")
        else:
            successful += 1
            breakout_count = len(result.get('breakouts', []))
            orb_range_pct = result['metrics']['orb_range_pct']
            
            if breakout_count > 0:
                with_breakouts += 1
                status = f"🎯 {breakout_count} breakout(s)"
                
                if breakout_count > 0 and orb_range_pct > 3:
                    excellent_setups += 1
                    status += " - EXCELLENT"
            else:
                status = "⚪ No breakouts"
            
            print(f"✅ {symbol}: {status} (ORB: {orb_range_pct:.1f}%)")
    
    print(f"\n📊 STATISTICS:")
    print(f"  Total symbols analyzed: {len(symbols)}")
    print(f"  Successful analyses: {successful}")
    print(f"  Symbols with breakouts: {with_breakouts}")
    print(f"  Excellent setups: {excellent_setups}")
    
    if successful > 0:
        print(f"  Breakout rate: {(with_breakouts/successful)*100:.1f}%")
        print(f"  Excellence rate: {(excellent_setups/successful)*100:.1f}%")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ORB analysis on multiple symbols')
    parser.add_argument('symbols', nargs='*', help='Specific symbols to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all found symbols')
    parser.add_argument('--max', type=int, default=5, help='Maximum symbols to analyze (default: 5)')
    
    args = parser.parse_args()
    
    print("🔍 ORB BATCH ANALYZER")
    print("="*80)
    
    if args.symbols:
        # Analyze specific symbols
        symbols = [s.upper() for s in args.symbols]
        run_batch_analysis(symbols)
    else:
        # Auto-discover and analyze
        max_symbols = None if args.all else args.max
        run_batch_analysis(max_symbols=max_symbols)

if __name__ == "__main__":
    main()