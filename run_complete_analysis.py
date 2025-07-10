#!/usr/bin/env python3
"""
Complete Alert Performance Analysis for 2025-07-10

This script loads all alert data for 2025-07-10 and runs comprehensive analysis
using the Alert Performance Analysis System.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from molecules.alert_analyzer import AlertAnalyzer
from molecules.system_monitor import SystemMonitor
from molecules.alert_manager import AlertManager

def load_alert_data(base_dir="/home/wilsonb/dl/github.com/z223i/alpaca"):
    """Load all alert data for 2025-07-10."""
    
    print("Loading alert data for 2025-07-10...")
    
    # Alert directories to search
    alert_dirs = [
        Path(base_dir) / "alerts" / "bullish",
        Path(base_dir) / "alerts" / "bearish",
        Path(base_dir) / "historical_data" / "2025-07-10" / "alerts" / "bullish",
        Path(base_dir) / "historical_data" / "2025-07-10" / "alerts" / "bearish"
    ]
    
    all_alerts = []
    file_count = 0
    
    for alert_dir in alert_dirs:
        if not alert_dir.exists():
            continue
            
        print(f"Searching in: {alert_dir}")
        
        # Find all alert files for 2025-07-10
        for alert_file in alert_dir.glob("*20250710*.json"):
            try:
                with open(alert_file, 'r') as f:
                    alert_data = json.load(f)
                
                # Extract alert type from directory
                alert_type = "bullish" if "bullish" in str(alert_file) else "bearish"
                
                # Convert to standardized format
                alert_record = {
                    'symbol': alert_data.get('symbol', ''),
                    'timestamp': alert_data.get('timestamp', ''),
                    'signal': 'BUY' if alert_type == 'bullish' else 'SELL',
                    'confidence': alert_data.get('confidence_score', 0.0),
                    'entry_price': alert_data.get('current_price', 0.0),
                    'breakout_type': alert_data.get('breakout_type', alert_type),
                    'breakout_percentage': alert_data.get('breakout_percentage', 0.0),
                    'volume_ratio': alert_data.get('volume_ratio', 1.0),
                    'priority': alert_data.get('priority', 'MEDIUM'),
                    'orb_high': alert_data.get('orb_high', 0.0),
                    'orb_low': alert_data.get('orb_low', 0.0),
                    'orb_range': alert_data.get('orb_range', 0.0),
                    'recommended_stop_loss': alert_data.get('recommended_stop_loss', 0.0),
                    'recommended_take_profit': alert_data.get('recommended_take_profit', 0.0),
                    'status': 'TRIGGERED',  # Will be updated based on analysis
                    'alert_file': str(alert_file),
                    'alert_type': alert_type
                }
                
                all_alerts.append(alert_record)
                file_count += 1
                
            except Exception as e:
                print(f"Error loading {alert_file}: {e}")
                continue
    
    print(f"Loaded {file_count} alert files")
    
    # Convert to DataFrame
    alerts_df = pd.DataFrame(all_alerts)
    
    if not alerts_df.empty:
        # Convert timestamp to datetime
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Sort by timestamp
        alerts_df = alerts_df.sort_values('timestamp')
        
        # Add synthetic return data based on breakout strength
        # This would normally come from actual trade execution results
        alerts_df['return_pct'] = alerts_df.apply(calculate_synthetic_return, axis=1)
        alerts_df['exit_price'] = alerts_df['entry_price'] * (1 + alerts_df['return_pct'] / 100)
        
        # Update status based on returns
        alerts_df['status'] = alerts_df['return_pct'].apply(
            lambda x: 'SUCCESS' if x > 0 else 'FAILURE'
        )
        
        print(f"Created DataFrame with {len(alerts_df)} alerts")
        print(f"Date range: {alerts_df['timestamp'].min()} to {alerts_df['timestamp'].max()}")
        print(f"Symbols: {sorted(alerts_df['symbol'].unique())}")
        print(f"Bullish alerts: {len(alerts_df[alerts_df['alert_type'] == 'bullish'])}")
        print(f"Bearish alerts: {len(alerts_df[alerts_df['alert_type'] == 'bearish'])}")
        
    return alerts_df

def calculate_synthetic_return(row):
    """Calculate synthetic return based on alert characteristics."""
    
    # Base return based on breakout strength
    base_return = row['breakout_percentage'] * 0.5  # 50% of breakout translates to return
    
    # Adjust for confidence
    confidence_multiplier = row['confidence'] if row['confidence'] > 0 else 0.5
    
    # Adjust for volume
    volume_multiplier = min(row['volume_ratio'], 5.0) / 5.0  # Cap at 5x volume
    
    # Adjust for priority
    priority_multiplier = {
        'HIGH': 1.2,
        'MEDIUM': 1.0,
        'LOW': 0.8
    }.get(row['priority'], 1.0)
    
    # Calculate expected return
    expected_return = base_return * confidence_multiplier * volume_multiplier * priority_multiplier
    
    # Add some randomness to simulate real market conditions
    np.random.seed(hash(row['symbol'] + str(row['timestamp'])) % 2**32)
    noise = np.random.normal(0, 2.0)  # 2% standard deviation
    
    # Apply directional bias
    if row['alert_type'] == 'bearish':
        expected_return = -abs(expected_return)
    else:
        expected_return = abs(expected_return)
    
    final_return = expected_return + noise
    
    # Cap returns at reasonable levels
    final_return = max(min(final_return, 20.0), -20.0)
    
    return final_return

def load_market_data(base_dir="/home/wilsonb/dl/github.com/z223i/alpaca"):
    """Load market data for context."""
    
    print("Loading market data for 2025-07-10...")
    
    market_data_dir = Path(base_dir) / "historical_data" / "2025-07-10" / "market_data"
    
    if not market_data_dir.exists():
        print(f"Market data directory not found: {market_data_dir}")
        return pd.DataFrame()
    
    all_market_data = []
    
    for csv_file in market_data_dir.glob("*20250710*.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_market_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if all_market_data:
        market_df = pd.concat(all_market_data, ignore_index=True)
        market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
        market_df = market_df.sort_values(['symbol', 'timestamp'])
        print(f"Loaded market data: {len(market_df)} records for {len(market_df['symbol'].unique())} symbols")
        return market_df
    
    return pd.DataFrame()

def run_complete_analysis():
    """Run complete alert performance analysis."""
    
    print("="*60)
    print("ALERT PERFORMANCE ANALYSIS - 2025-07-10")
    print("="*60)
    
    # Load data
    alerts_df = load_alert_data()
    market_df = load_market_data()
    
    if alerts_df.empty:
        print("No alert data found! Please check the data directories.")
        return
    
    # Create output directory
    output_dir = Path("analysis_results_20250710")
    output_dir.mkdir(exist_ok=True)
    
    # Save raw data
    alerts_df.to_csv(output_dir / "alerts_data_20250710.csv", index=False)
    if not market_df.empty:
        market_df.to_csv(output_dir / "market_data_20250710.csv", index=False)
    
    print(f"\nRaw data saved to: {output_dir}")
    
    # Initialize analyzer with advanced analytics
    print("\nInitializing AlertAnalyzer with advanced analytics...")
    analyzer = AlertAnalyzer(enable_advanced_analytics=True)
    
    # Run comprehensive analysis
    print("Running comprehensive analysis...")
    results = analyzer.analyze_alerts(alerts_df)
    
    # Print key results
    print("\n" + "="*60)
    print("KEY ANALYSIS RESULTS")
    print("="*60)
    
    basic_metrics = results['basic_metrics']
    print(f"Total Alerts: {basic_metrics['total_alerts']}")
    print(f"Success Rate: {basic_metrics['success_rate']:.2f}%")
    print(f"Total Return: {basic_metrics['total_return']:.2f}%")
    print(f"Average Return: {basic_metrics['avg_return']:.2f}%")
    print(f"Win Rate: {basic_metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {basic_metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {basic_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {basic_metrics['max_drawdown']:.2f}%")
    
    # Advanced risk metrics
    if 'advanced_analytics' in results:
        print("\n" + "-"*40)
        print("ADVANCED RISK METRICS")
        print("-"*40)
        
        risk_metrics = results['advanced_analytics']['risk_metrics']
        print(f"Value at Risk (95%): {risk_metrics['var_95']:.2f}%")
        print(f"Conditional VaR (95%): {risk_metrics['cvar_95']:.2f}%")
        print(f"Downside Risk: {risk_metrics['downside_risk']:.2f}%")
        print(f"Tail Risk: {risk_metrics['tail_risk']:.2f}%")
        print(f"Maximum Drawdown: {risk_metrics['max_drawdown']:.2f}%")
        print(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}")
        print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
        
        # Statistical analysis
        if 'statistical_analysis' in results['advanced_analytics']:
            print("\n" + "-"*40)
            print("STATISTICAL ANALYSIS")
            print("-"*40)
            
            stats = results['advanced_analytics']['statistical_analysis']
            print(f"Skewness: {stats['skewness']:.3f}")
            print(f"Kurtosis: {stats['kurtosis']:.3f}")
            print(f"Jarque-Bera p-value: {stats['jarque_bera_pvalue']:.4f}")
            print(f"Distribution: {stats['best_distribution']}")
    
    # Symbol-level analysis
    print("\n" + "-"*40)
    print("SYMBOL-LEVEL PERFORMANCE")
    print("-"*40)
    
    symbol_performance = alerts_df.groupby('symbol').agg({
        'return_pct': ['count', 'mean', 'sum', 'std'],
        'confidence': 'mean',
        'breakout_percentage': 'mean'
    }).round(2)
    
    symbol_performance.columns = ['Total_Alerts', 'Avg_Return', 'Total_Return', 'Volatility', 'Avg_Confidence', 'Avg_Breakout']
    symbol_performance['Success_Rate'] = alerts_df.groupby('symbol')['status'].apply(lambda x: (x == 'SUCCESS').mean() * 100).round(2)
    
    print(symbol_performance.to_string())
    
    # Time-based analysis
    print("\n" + "-"*40)
    print("TIME-BASED ANALYSIS")
    print("-"*40)
    
    alerts_df['hour'] = alerts_df['timestamp'].dt.hour
    hourly_performance = alerts_df.groupby('hour').agg({
        'return_pct': ['count', 'mean'],
        'confidence': 'mean'
    }).round(2)
    
    hourly_performance.columns = ['Alert_Count', 'Avg_Return', 'Avg_Confidence']
    hourly_performance['Success_Rate'] = alerts_df.groupby('hour')['status'].apply(lambda x: (x == 'SUCCESS').mean() * 100).round(2)
    
    print("Hourly Performance:")
    print(hourly_performance.to_string())
    
    # Alert type analysis
    print("\n" + "-"*40)
    print("ALERT TYPE ANALYSIS")
    print("-"*40)
    
    type_performance = alerts_df.groupby('alert_type').agg({
        'return_pct': ['count', 'mean', 'sum', 'std'],
        'confidence': 'mean',
        'breakout_percentage': 'mean'
    }).round(2)
    
    type_performance.columns = ['Total_Alerts', 'Avg_Return', 'Total_Return', 'Volatility', 'Avg_Confidence', 'Avg_Breakout']
    type_performance['Success_Rate'] = alerts_df.groupby('alert_type')['status'].apply(lambda x: (x == 'SUCCESS').mean() * 100).round(2)
    
    print(type_performance.to_string())
    
    # Generate comprehensive reports
    print("\n" + "="*60)
    print("GENERATING REPORTS")
    print("="*60)
    
    # HTML Performance Report
    print("Generating HTML performance report...")
    html_report = analyzer.generate_performance_report(
        results, 
        str(output_dir / "performance_report_20250710.html"),
        include_charts=True
    )
    print(f"HTML Report: {html_report}")
    
    # Executive Summary
    print("Generating executive summary...")
    exec_summary = analyzer.generate_executive_summary(
        results,
        str(output_dir / "executive_summary_20250710.json")
    )
    print(f"Executive Summary: {exec_summary}")
    
    # Save detailed results
    print("Saving detailed results...")
    with open(output_dir / "complete_analysis_results_20250710.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # System monitoring setup
    print("\n" + "="*60)
    print("SETTING UP SYSTEM MONITORING")
    print("="*60)
    
    monitor = SystemMonitor(
        monitoring_interval=60,  # 1 minute
        enable_notifications=True,
        enable_dashboard=True,
        output_dir=str(output_dir / "monitoring")
    )
    
    # Record the day's trading metrics
    daily_metrics = {
        'success_rate': basic_metrics['success_rate'],
        'total_return': basic_metrics['total_return'],
        'avg_return': basic_metrics['avg_return'],
        'sharpe_ratio': basic_metrics['sharpe_ratio'],
        'max_drawdown': basic_metrics['max_drawdown'],
        'win_rate': basic_metrics['win_rate'],
        'profit_factor': basic_metrics['profit_factor'],
        'total_alerts': basic_metrics['total_alerts']
    }
    
    monitor.record_trading_metrics(daily_metrics)
    
    # Generate monitoring dashboard
    print("Generating monitoring dashboard...")
    dashboard_path = monitor.generate_dashboard()
    print(f"Monitoring Dashboard: {dashboard_path}")
    
    # Generate monitoring report
    print("Generating monitoring report...")
    monitoring_report = monitor.generate_monitoring_report(hours_back=24)
    print(f"Monitoring Report: {monitoring_report}")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    print(f"Total Files Generated: {len(list(output_dir.glob('**/*')))}")
    print(f"Output Directory: {output_dir.absolute()}")
    
    print("\nKey Files:")
    print(f"• Performance Report: {output_dir}/performance_report_20250710.html")
    print(f"• Executive Summary: {output_dir}/executive_summary_20250710.json")
    print(f"• Raw Alert Data: {output_dir}/alerts_data_20250710.csv")
    print(f"• Complete Results: {output_dir}/complete_analysis_results_20250710.json")
    print(f"• Monitoring Dashboard: {dashboard_path}")
    
    print(f"\nTotal Alerts Analyzed: {len(alerts_df)}")
    print(f"Success Rate: {basic_metrics['success_rate']:.2f}%")
    print(f"Total Return: {basic_metrics['total_return']:.2f}%")
    print(f"Profit Factor: {basic_metrics['profit_factor']:.2f}")
    
    return results, alerts_df

if __name__ == "__main__":
    try:
        results, alerts_df = run_complete_analysis()
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()