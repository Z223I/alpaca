#!/usr/bin/env python3
"""
Alert Performance Analysis CLI Script.

This script provides a command-line interface for analyzing alert performance
using the AlertAnalyzer molecule and underlying atoms.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from molecules.alert_analyzer import AlertAnalyzer


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze alert performance using historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze most recent data
  python alert_analyzer.py

  # Analyze specific date
  python alert_analyzer.py --date 2025-01-15

  # Analyze with custom data paths
  python alert_analyzer.py --market-data historical_data --alerts-data alerts_data

  # Export results to file
  python alert_analyzer.py --output analysis_results.json

  # Verbose output
  python alert_analyzer.py --verbose
        '''
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Specific date to analyze (YYYY-MM-DD format). If not provided, uses most recent data.'
    )
    
    parser.add_argument(
        '--market-data',
        type=str,
        default='historical_data',
        help='Path to market data directory (default: historical_data)'
    )
    
    parser.add_argument(
        '--alerts-data', 
        type=str,
        default='alerts_data',
        help='Path to alerts data directory (default: alerts_data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (JSON format)'
    )
    
    parser.add_argument(
        '--trading-hours-start',
        type=str,
        default='09:30',
        help='Trading hours start time (HH:MM format, default: 09:30)'
    )
    
    parser.add_argument(
        '--trading-hours-end',
        type=str,
        default='16:00',
        help='Trading hours end time (HH:MM format, default: 16:00)'
    )
    
    parser.add_argument(
        '--alert-hours-start',
        type=str,
        default='09:30',
        help='Valid alert hours start time (HH:MM format, default: 09:30)'
    )
    
    parser.add_argument(
        '--alert-hours-end',
        type=str,
        default='15:30',
        help='Valid alert hours end time (HH:MM format, default: 15:30)'
    )
    
    parser.add_argument(
        '--timezone',
        type=str,
        default='US/Eastern',
        help='Timezone for analysis (default: US/Eastern)'
    )
    
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=7.5,
        help='Default stop loss percentage (default: 7.5)'
    )
    
    parser.add_argument(
        '--take-profit',
        type=float, 
        default=15.0,
        help='Default take profit percentage (default: 15.0)'
    )
    
    parser.add_argument(
        '--max-duration',
        type=int,
        default=24,
        help='Maximum trade duration in hours (default: 24)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def print_results_summary(results: dict, verbose: bool = False):
    """Print analysis results summary."""
    print("\n" + "="*60)
    print("ALERT PERFORMANCE ANALYSIS RESULTS")
    print("="*60)
    
    if results['status'] != 'success':
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return
    
    # Data loading summary
    data_load = results.get('data_load', {})
    print(f"\n📊 Data Loading:")
    print(f"  • Market data points: {data_load.get('market_data_count', 0):,}")
    print(f"  • Alerts loaded: {data_load.get('alerts_count', 0):,}")
    
    # Simulation summary
    simulation = results.get('simulation', {})
    print(f"\n🔄 Simulation:")
    print(f"  • Trades simulated: {simulation.get('simulated_trades', 0):,}")
    print(f"  • Successful trades: {simulation.get('successful_trades', 0):,}")
    print(f"  • Failed trades: {simulation.get('failed_trades', 0):,}")
    
    # Performance metrics
    performance = results.get('performance', {})
    if performance.get('status') == 'success':
        summary = performance.get('summary', {})
        returns_metrics = performance.get('returns_metrics', {})
        risk_metrics = performance.get('risk_metrics', {})
        
        print(f"\n📈 Performance Metrics:")
        print(f"  • Success Rate: {summary.get('success_rate', 0):.2f}%")
        print(f"  • Average Return: {summary.get('average_return', 0):.2f}%")
        print(f"  • Total Return: {returns_metrics.get('total_return', 0):.2f}%")
        print(f"  • Profit Factor: {summary.get('profit_factor', 0):.2f}")
        print(f"  • Win/Loss Ratio: {summary.get('win_loss_ratio', 0):.2f}")
        print(f"  • Max Drawdown: {returns_metrics.get('max_drawdown', 0):.2f}%")
        
        # Risk metrics
        if risk_metrics:
            print(f"\n📊 Risk Metrics:")
            print(f"  • Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  • Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.2f}")
            print(f"  • Calmar Ratio: {risk_metrics.get('calmar_ratio', 0):.2f}")
        
        # Symbol performance (top 10)
        symbol_perf = performance.get('symbol_performance', [])
        if symbol_perf and verbose:
            print(f"\n🏆 Top 10 Symbol Performance:")
            for i, symbol in enumerate(symbol_perf[:10]):
                print(f"  {i+1:2d}. {symbol.get('symbol', 'N/A'):6s} - "
                      f"Success Rate: {symbol.get('success_rate', 0):6.2f}% "
                      f"({symbol.get('successful_trades', 0):3d}/"
                      f"{symbol.get('total_trades', 0):3d})")
        
        # Priority performance
        priority_perf = performance.get('priority_performance', [])
        if priority_perf and verbose:
            print(f"\n⭐ Priority Performance:")
            for priority in priority_perf:
                print(f"  • {priority.get('priority', 'N/A'):8s} - "
                      f"Success Rate: {priority.get('success_rate', 0):6.2f}% "
                      f"({priority.get('successful_trades', 0):3d}/"
                      f"{priority.get('total_trades', 0):3d})")
    
    print("\n" + "="*60)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Initialize analyzer
    analyzer = AlertAnalyzer(
        trading_hours_start=args.trading_hours_start,
        trading_hours_end=args.trading_hours_end,
        alert_hours_start=args.alert_hours_start,
        alert_hours_end=args.alert_hours_end,
        timezone=args.timezone
    )
    
    # Print configuration
    if args.verbose:
        print("Alert Performance Analysis Configuration:")
        print(f"  • Market Data Path: {args.market_data}")
        print(f"  • Alerts Data Path: {args.alerts_data}")
        print(f"  • Analysis Date: {args.date or 'Most Recent'}")
        print(f"  • Trading Hours: {args.trading_hours_start}-{args.trading_hours_end} {args.timezone}")
        print(f"  • Alert Hours: {args.alert_hours_start}-{args.alert_hours_end} {args.timezone}")
        print(f"  • Stop Loss: {args.stop_loss}%")
        print(f"  • Take Profit: {args.take_profit}%")
        print(f"  • Max Duration: {args.max_duration} hours")
        print()
    
    # Run analysis
    try:
        if args.verbose:
            print("🔄 Starting alert performance analysis...")
        
        results = analyzer.analyze_alerts(
            market_data_path=args.market_data,
            alerts_data_path=args.alerts_data,
            date=args.date
        )
        
        # Print results
        print_results_summary(results, args.verbose)
        
        # Export results if requested
        if args.output:
            if analyzer.export_results(args.output):
                print(f"\n✅ Results exported to: {args.output}")
            else:
                print(f"\n❌ Failed to export results to: {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if results['status'] == 'success' else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()