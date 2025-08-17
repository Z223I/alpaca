#!/usr/bin/env python3
"""
Backtesting Results Analysis - Parameter Optimization Charts

This script analyzes completed backtesting runs and generates comprehensive charts
showing how different parameter combinations affect superduper alerts generation.

Usage:
    python3 code/analyze_backtesting_results.py
    
Charts generated:
    - Heatmap: timeframe vs threshold with alerts as color intensity
    - Line plot: alerts by timeframe for each threshold
    - Line plot: alerts by threshold for each timeframe
    - Bar chart: total alerts by parameter combination
    - 3D surface plot: parameter relationships
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from code.backtesting import BacktestingSystem


class BacktestingAnalyzer:
    """Analyzes backtesting results and generates parameter optimization charts."""
    
    def __init__(self):
        self.results_data = []
        self.param_data = None
        self.chart_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.using_real_data = False
        
    def extract_parameters_from_run_name(self, run_name):
        """Extract timeframe and threshold from run directory name.
        
        Example: run_2025-08-04_tf10_th0.65_e101f2f1
        Returns: {'timeframe': 10, 'threshold': 0.65, 'date': '2025-08-04'}
        """
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
    
    def read_simulation_results(self, run_dir):
        """Read ACTUAL alert results from superduper_alerts_sent directory (not simulation logs)."""
        
        # Count actual alerts that were sent (like backtesting system does)
        total_alerts = 0
        alerts_by_symbol = {}
        
        # Look for superduper_alerts_sent files in historical_data
        historical_data_dir = run_dir / "historical_data"
        if historical_data_dir.exists():
            for alert_file in historical_data_dir.rglob("**/superduper_alerts_sent/**/*.json"):
                try:
                    with open(alert_file, 'r') as f:
                        alert_data = json.load(f)
                        if isinstance(alert_data, list):
                            total_alerts += len(alert_data)
                            # Track by symbol for each alert in list
                            for alert in alert_data:
                                symbol = alert.get('symbol', 'UNKNOWN')
                                alerts_by_symbol[symbol] = alerts_by_symbol.get(symbol, 0) + 1
                        else:
                            total_alerts += 1
                            # Track by symbol for single alert
                            symbol = alert_data.get('symbol', 'UNKNOWN')
                            alerts_by_symbol[symbol] = alerts_by_symbol.get(symbol, 0) + 1
                except Exception as e:
                    print(f"   ⚠️ Error reading {alert_file}: {e}")
        
        return {
            'total_alerts': total_alerts,
            'alerts_generated': total_alerts,  # Use actual count  
            'symbol_results': [],  # Not needed
            'alerts_by_symbol': alerts_by_symbol
        }

    def collect_run_data(self):
        """Collect data from all completed backtesting runs."""
        print("🔍 Analyzing backtesting runs...")
        
        runs_dir = Path("runs")
        if not runs_dir.exists():
            print("❌ No runs directory found")
            return
            
        # Scan nested directory structure: runs/{date}/{symbol}/run_*  
        for date_dir in runs_dir.glob("2025-*"):  # Date directories
            if not date_dir.is_dir():
                continue
                
            for symbol_dir in date_dir.glob("*"):  # Symbol directories  
                if not symbol_dir.is_dir():
                    continue
                    
                for run_dir in symbol_dir.glob("run_*"):  # Individual runs
                    if not run_dir.is_dir():
                        continue
                        
                    run_name = run_dir.name
                    print(f"📊 Analyzing {date_dir.name}/{symbol_dir.name}/{run_name}...")
                    
                    # Extract parameters from run directory name
                    params = self.extract_parameters_from_run_name(run_name)
                    if not params:
                        print(f"   ⚠️ Could not extract parameters from {run_name}")
                        continue
                    
                    # Read actual simulation results
                    try:
                        results = self.read_simulation_results(run_dir)
                        
                        # Add the results with extracted parameters
                        run_data = {
                            'run_id': run_name,
                            'symbol': symbol_dir.name,
                            'date': params['date'],
                            'timeframe': params['timeframe'],
                            'threshold': params['threshold'],
                            'uuid': params['uuid'],
                            'total_alerts': results['total_alerts'],
                            'alerts_generated': results['alerts_generated'],
                            'symbol_results': results['symbol_results'],
                            'alerts_by_symbol': results['alerts_by_symbol'],
                            'superduper_alerts_sent': results['total_alerts'],  # Use actual alert count
                            'parameter_combo': f"{params['timeframe']}min_{params['threshold']:.2f}th"
                        }
                        
                        self.results_data.append(run_data)
                        print(f"   ✅ Found {results['total_alerts']} total alerts generated")
                        
                    except Exception as e:
                        print(f"   ⚠️ Error analyzing {run_name}: {e}")
                        continue
        
        print(f"📈 Collected data from {len(self.results_data)} runs")
        
        if len(self.results_data) == 0:
            print("❌ No valid run data found - cannot create charts from real data")
        else:
            print("✅ Using REAL backtesting data (not synthetic)")
        
    def create_parameter_combinations(self):
        """Convert real run data to parameter analysis format."""
        print("🎯 Creating parameter combination analysis from REAL data...")
        
        if not self.results_data:
            print("❌ FATAL ERROR: No real results data found!")
            print("❌ CANNOT GENERATE CHARTS: Only real data is permitted")
            print("❌ Please run actual backtesting first to generate real data")
            return
        
        # Convert real results data to DataFrame
        real_data = []
        
        for run_data in self.results_data:
            real_data.append({
                'timeframe': run_data['timeframe'],
                'threshold': run_data['threshold'],
                'superduper_alerts_sent': run_data['total_alerts'],
                'parameter_combo': run_data['parameter_combo'],
                'symbol': run_data['symbol'],
                'date': run_data['date'],
                'run_id': run_data['run_id']
            })
        
        df = pd.DataFrame(real_data)
        
        # Aggregate data by parameter combination (sum alerts across all symbols/dates)
        self.param_data = df.groupby(['timeframe', 'threshold']).agg({
            'superduper_alerts_sent': 'sum',
            'parameter_combo': 'first',  # Take first since they're all the same for a combination
            'symbol': lambda x: ','.join(sorted(set(x))),  # List all symbols tested
            'date': lambda x: ','.join(sorted(set(x))),    # List all dates tested
            'run_id': 'count'  # Count how many runs per combination
        }).rename(columns={'run_id': 'total_runs'}).reset_index()
        
        print(f"✅ Using REAL data from {len(real_data)} backtesting runs")
        print(f"📊 Parameter combinations: {len(self.param_data)} unique (aggregated across symbols)")
        
        # Show alert count summary
        total_alerts = self.param_data['superduper_alerts_sent'].sum()
        max_alerts = self.param_data['superduper_alerts_sent'].max()
        print(f"📊 Alert summary: {total_alerts} total, {max_alerts} max per combination")
        
        
    def create_heatmap(self):
        """Create heatmap showing alerts by timeframe vs threshold."""
        print("🎨 Creating parameter heatmap...")
        
        if self.param_data is None or len(self.param_data) == 0:
            print("❌ No parameter data available - cannot create heatmap")
            return
            
        # Data source is always real - no synthetic option exists
        data_source = "REAL Backtesting Data"
        
        # Pivot data for heatmap
        heatmap_data = self.param_data.pivot(index='threshold', columns='timeframe', values='superduper_alerts_sent')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Superduper Alerts Sent'})
        plt.title('Superduper Alerts by Timeframe and Threshold', fontsize=16, fontweight='bold')
        plt.suptitle(f'Data Source: {data_source} - Actual superduper_alerts_sent Files', fontsize=12, y=0.02, color='darkgreen', weight='bold')
        plt.xlabel('Timeframe (minutes)', fontsize=12)
        plt.ylabel('Green Threshold', fontsize=12)
        plt.gca().invert_yaxis()  # Fix inverted Y-axis
        plt.tight_layout()
        
        filename = f"runs/analysis_heatmap_{self.chart_timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {filename}")
        
    def create_timeframe_analysis(self):
        """Create line plot showing alerts by timeframe for each threshold."""
        print("📈 Creating timeframe analysis...")
        
        plt.figure(figsize=(12, 8))
        
        for threshold in sorted(self.param_data['threshold'].unique()):
            threshold_data = self.param_data[self.param_data['threshold'] == threshold]
            # Sort by timeframe for smooth lines
            threshold_data = threshold_data.sort_values('timeframe')
            plt.plot(threshold_data['timeframe'], threshold_data['superduper_alerts_sent'], 
                    marker='o', linewidth=2, label=f'Threshold {threshold:.2f}')
        
        plt.title('Superduper Alerts by Timeframe', fontsize=16, fontweight='bold')
        plt.suptitle('Data Source: REAL Backtesting Data - Actual superduper_alerts_sent Files', fontsize=12, y=0.02, color='darkgreen', weight='bold')
        plt.xlabel('Timeframe (minutes)', fontsize=12)
        plt.ylabel('Superduper Alerts Sent', fontsize=12)
        plt.legend(title='Green Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"runs/analysis_timeframe_{self.chart_timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {filename}")
        
    def create_threshold_analysis(self):
        """Create line plot showing alerts by threshold for each timeframe."""
        print("📊 Creating threshold analysis...")
        
        plt.figure(figsize=(12, 8))
        
        for timeframe in sorted(self.param_data['timeframe'].unique()):
            timeframe_data = self.param_data[self.param_data['timeframe'] == timeframe]
            # Sort by threshold for smooth lines
            timeframe_data = timeframe_data.sort_values('threshold')
            plt.plot(timeframe_data['threshold'], timeframe_data['superduper_alerts_sent'], 
                    marker='s', linewidth=2, label=f'{timeframe} min')
        
        plt.title('Superduper Alerts by Green Threshold', fontsize=16, fontweight='bold')
        plt.suptitle('Data Source: REAL Backtesting Data - Actual superduper_alerts_sent Files', fontsize=12, y=0.02, color='darkgreen', weight='bold')
        plt.xlabel('Green Threshold', fontsize=12)
        plt.ylabel('Superduper Alerts Sent', fontsize=12)
        plt.legend(title='Timeframe', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"runs/analysis_threshold_{self.chart_timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {filename}")
        
    def create_combination_bar_chart(self):
        """Create bar chart showing total alerts by parameter combination."""
        print("📊 Creating parameter combination bar chart...")
        
        # Sort by alerts sent for better visualization
        sorted_data = self.param_data.sort_values('superduper_alerts_sent', ascending=False)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(sorted_data)), sorted_data['superduper_alerts_sent'], 
                      color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, alerts) in enumerate(zip(bars, sorted_data['superduper_alerts_sent'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(alerts), ha='center', va='bottom', fontweight='bold')
        
        plt.title('Superduper Alerts by Parameter Combination', fontsize=16, fontweight='bold')
        plt.suptitle('Data Source: REAL Backtesting Data - Actual superduper_alerts_sent Files', fontsize=12, y=0.02, color='darkgreen', weight='bold')
        plt.xlabel('Parameter Combination (Timeframe_Threshold)', fontsize=12)
        plt.ylabel('Superduper Alerts Sent', fontsize=12)
        plt.xticks(range(len(sorted_data)), sorted_data['parameter_combo'], rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"runs/analysis_combinations_{self.chart_timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {filename}")
        
    def create_3d_surface_plot(self):
        """Create 3D surface plot showing parameter relationships."""
        print("🎲 Creating 3D surface plot...")
        
        # Prepare data for 3D plot
        timeframes = sorted(self.param_data['timeframe'].unique())
        thresholds = sorted(self.param_data['threshold'].unique())
        
        X, Y = np.meshgrid(timeframes, thresholds)
        Z = np.zeros_like(X)
        
        for i, threshold in enumerate(thresholds):
            for j, timeframe in enumerate(timeframes):
                alerts = self.param_data[
                    (self.param_data['timeframe'] == timeframe) & 
                    (self.param_data['threshold'] == threshold)
                ]['superduper_alerts_sent'].iloc[0]
                Z[i, j] = alerts
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        
        # Add contour lines
        contours = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-2, cmap='viridis', alpha=0.6)
        
        ax.set_title('3D Parameter Optimization Surface', fontsize=16, fontweight='bold')
        fig.suptitle('Data Source: REAL Backtesting Data - Actual superduper_alerts_sent Files', fontsize=12, y=0.02, color='darkgreen', weight='bold')
        ax.set_xlabel('Timeframe (minutes)', fontsize=12)
        ax.set_ylabel('Green Threshold', fontsize=12)
        ax.set_zlabel('Superduper Alerts Sent', fontsize=12)
        
        # Add colorbar
        fig.colorbar(surface, shrink=0.5, aspect=5, label='Alerts Sent')
        
        filename = f"runs/analysis_3d_surface_{self.chart_timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {filename}")
        
    def create_optimization_summary(self):
        """Create summary statistics and best parameter recommendations."""
        print("📋 Creating optimization summary...")
        
        # Find best performing combinations
        best_combo = self.param_data.loc[self.param_data['superduper_alerts_sent'].idxmax()]
        worst_combo = self.param_data.loc[self.param_data['superduper_alerts_sent'].idxmin()]
        
        # Calculate statistics by parameter
        timeframe_stats = self.param_data.groupby('timeframe')['superduper_alerts_sent'].agg(['mean', 'std', 'max'])
        threshold_stats = self.param_data.groupby('threshold')['superduper_alerts_sent'].agg(['mean', 'std', 'max'])
        
        summary = f"""
BACKTESTING PARAMETER OPTIMIZATION SUMMARY
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BEST PERFORMING COMBINATION:
- Timeframe: {best_combo['timeframe']} minutes
- Threshold: {best_combo['threshold']:.2f}
- Alerts Sent: {best_combo['superduper_alerts_sent']}

WORST PERFORMING COMBINATION:
- Timeframe: {worst_combo['timeframe']} minutes  
- Threshold: {worst_combo['threshold']:.2f}
- Alerts Sent: {worst_combo['superduper_alerts_sent']}

TIMEFRAME ANALYSIS:
{timeframe_stats.to_string()}

THRESHOLD ANALYSIS:
{threshold_stats.to_string()}

OPTIMIZATION INSIGHTS:
- Total parameter combinations tested: {len(self.param_data)}
- Average alerts per combination: {self.param_data['superduper_alerts_sent'].mean():.1f}
- Standard deviation: {self.param_data['superduper_alerts_sent'].std():.1f}
- Best timeframe (by avg): {timeframe_stats['mean'].idxmax()} minutes
- Best threshold (by avg): {threshold_stats['mean'].idxmax():.2f}

CHARTS GENERATED:
- analysis_heatmap_{self.chart_timestamp}.png
- analysis_timeframe_{self.chart_timestamp}.png  
- analysis_threshold_{self.chart_timestamp}.png
- analysis_combinations_{self.chart_timestamp}.png
- analysis_3d_surface_{self.chart_timestamp}.png
"""
        
        summary_file = f"runs/analysis_summary_{self.chart_timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"   💾 Saved: {summary_file}")
        print("\n" + "="*50)
        print("OPTIMIZATION SUMMARY")
        print("="*50)
        print(f"Best combination: {best_combo['timeframe']}min, {best_combo['threshold']:.2f} threshold")
        print(f"Best performance: {best_combo['superduper_alerts_sent']} alerts sent")
        print(f"Charts saved with timestamp: {self.chart_timestamp}")
        
    def validate_data_source(self):
        """Validate that we have real data - no synthetic option exists."""
        if not self.results_data:
            print("❌ FATAL ERROR: No real backtesting data found!")
            print("❌ CHARTS CANNOT BE GENERATED: Only real data is permitted")
            self.using_real_data = False
            return False
        else:
            print(f"✅ VALIDATION: Using real data from {len(self.results_data)} actual runs")
            print(f"✅ VALIDATION: Data sources - {set(run['symbol'] for run in self.results_data)}")
            print(f"✅ VALIDATION: Date range - {set(run['date'] for run in self.results_data)}")
            self.using_real_data = True
            return True
    
    def run_analysis(self):
        """Run complete backtesting analysis and generate all charts."""
        print("🚀 Starting backtesting parameter analysis...")
        
        # Collect actual run data (if available)
        self.collect_run_data()
        
        # Validate data source
        self.validate_data_source()
        
        # Create parameter analysis data
        self.create_parameter_combinations()
        
        # Final validation - only real data is allowed
        if self.param_data is not None and len(self.param_data) > 0 and self.using_real_data:
            print(f"✅ VALIDATION: Chart data ready - {len(self.param_data)} data points")
            print("✅ VALIDATION: Charts will show REAL backtesting results ONLY")
        else:
            print("❌ VALIDATION FAILED: No real data available for charts")
            print("❌ CHARTS WILL NOT BE GENERATED: Only real data is permitted")
            return
        
        # Generate all charts
        self.create_heatmap()
        self.create_timeframe_analysis()
        self.create_threshold_analysis()
        self.create_combination_bar_chart()
        self.create_3d_surface_plot()
        
        # Generate summary
        self.create_optimization_summary()
        
        print("\n🎉 Analysis complete! Check the runs/ directory for generated charts.")
        print("✅ ALL CHARTS GENERATED FROM REAL BACKTESTING DATA ONLY")


def main():
    """Main entry point for backtesting analysis."""
    analyzer = BacktestingAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()