#!/usr/bin/env python3
"""
ORB Analysis Script - Opening Range Impact Study

This script systematically tests different opening range periods (10-30 minutes by 5s)
using the full date range available in ./data directory. It compares PCA results
to determine the impact of opening range period on market pattern detection.

Usage:
    python code/orb_analysis.py

Output:
    Creates orb_analysis_results.txt with comprehensive comparison results.
"""

import os
import sys
import glob
import subprocess
import json
import re
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
import traceback
import matplotlib.pyplot as plt
import numpy as np

class ORBAnalyzer:
    """Comprehensive ORB analysis across different opening range periods."""

    def __init__(self):
        self.data_directory = 'data'
        self.opening_range_periods = [10, 15, 20, 25, 30]  # minutes
        self.results = {}
        self.analysis_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def discover_date_range(self) -> Tuple[Optional[date], Optional[date]]:
        """
        Discover the available date range from CSV files in data directory.

        Returns:
            Tuple of (start_date, end_date) or (None, None) if no valid files
        """
        try:
            csv_pattern = os.path.join(self.data_directory, '*.csv')
            csv_files = glob.glob(csv_pattern)

            # Filter to only include files matching YYYYMMDD.csv format
            valid_dates = []
            for filepath in csv_files:
                filename = os.path.basename(filepath)
                if filename == 'symbols.csv':  # Skip symbols.csv
                    continue

                try:
                    # Remove .csv extension and check if it's 8 digits
                    date_str = filename.replace('.csv', '')
                    if len(date_str) == 8 and date_str.isdigit():
                        file_date = datetime.strptime(date_str, '%Y%m%d').date()
                        valid_dates.append(file_date)
                except ValueError:
                    continue

            if not valid_dates:
                return None, None

            valid_dates.sort()
            return valid_dates[0], valid_dates[-1]

        except Exception as e:
            print(f"Error discovering date range: {e}")
            return None, None

    def extract_pca_metrics(self, output: str) -> Dict[str, Any]:
        """
        Extract PCA analysis metrics from orb.py output.

        Args:
            output: stdout from orb.py execution

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {
            'total_symbols': 0,
            'total_rows': 0,
            'pca_components': 0,
            'total_variance_explained': 0.0,
            'first_component_variance': 0.0,
            'processing_success': False,
            'error_message': None,
            'execution_summary': {}
        }

        try:
            # Extract basic processing info
            if "✓ Multi-date PCA analysis completed successfully!" in output:
                metrics['processing_success'] = True
            elif "✗ Multi-date PCA analysis failed" in output:
                metrics['processing_success'] = False
                # Try to find error message
                lines = output.split('\n')
                for line in lines:
                    if 'error' in line.lower() or 'failed' in line.lower():
                        metrics['error_message'] = line.strip()
                        break

            # Extract symbol and data counts
            symbol_match = re.search(r'Total symbols: (\d+)', output)
            if symbol_match:
                metrics['total_symbols'] = int(symbol_match.group(1))

            rows_match = re.search(r'Total rows: ([\d,]+)', output)
            if rows_match:
                rows_str = rows_match.group(1).replace(',', '')
                metrics['total_rows'] = int(rows_str)

            # Extract PCA results
            components_match = re.search(r'PCA completed with (\d+) components', output)
            if components_match:
                metrics['pca_components'] = int(components_match.group(1))

            total_var_match = re.search(r'Total variance explained: ([\d.]+)%', output)
            if total_var_match:
                metrics['total_variance_explained'] = float(total_var_match.group(1))

            first_var_match = re.search(r'First component explains: ([\d.]+)%', output)
            if first_var_match:
                metrics['first_component_variance'] = float(first_var_match.group(1))

            # Extract processing summary
            summary_match = re.search(r'Processed files: (\d+)/(\d+)', output)
            if summary_match:
                metrics['execution_summary']['processed_files'] = int(summary_match.group(1))
                metrics['execution_summary']['total_files'] = int(summary_match.group(2))
                metrics['execution_summary']['success_rate'] = (
                    int(summary_match.group(1)) / int(summary_match.group(2)) * 100
                )

            # Extract dates processed
            date_range_match = re.search(r'Date range: ([\d-]+) to ([\d-]+)', output)
            if date_range_match:
                metrics['execution_summary']['start_date'] = date_range_match.group(1)
                metrics['execution_summary']['end_date'] = date_range_match.group(2)

        except Exception as e:
            metrics['error_message'] = f"Error parsing output: {str(e)}"

        return metrics

    def run_orb_analysis(self, start_date: date, end_date: date, opening_range: int) -> Dict[str, Any]:
        """
        Run ORB analysis for a specific opening range period.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            opening_range: Opening range period in minutes

        Returns:
            Dictionary with analysis results and metrics
        """
        print(f"Running ORB analysis with {opening_range}-minute opening range...")

        # Construct command - use shell command format for conda environment
        cmd_str = f"~/miniconda3/envs/alpaca/bin/python code/orb.py --start {start_date.strftime('%Y-%m-%d')} --end {end_date.strftime('%Y-%m-%d')} --opening-range {opening_range}"

        result = {
            'opening_range_minutes': opening_range,
            'command': cmd_str,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': 0,
            'exit_code': None,
            'stdout': '',
            'stderr': '',
            'metrics': {},
            'success': False
        }

        try:
            start_time = datetime.now()

            # Run the analysis using shell command
            process = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                shell=True,  # Need shell=True for the conda path to work
                timeout=1800  # 30 minute timeout
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result.update({
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'exit_code': process.returncode,
                'stdout': process.stdout,
                'stderr': process.stderr,
                'success': process.returncode == 0
            })

            # Extract metrics from output
            if result['success']:
                result['metrics'] = self.extract_pca_metrics(process.stdout)
            else:
                result['metrics'] = {
                    'processing_success': False,
                    'error_message': f"Process failed with exit code {process.returncode}",
                    'stderr': process.stderr
                }

            print(f"✓ Completed {opening_range}-minute analysis in {duration:.1f} seconds")

        except subprocess.TimeoutExpired:
            result.update({
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 1800,
                'exit_code': -1,
                'success': False,
                'metrics': {
                    'processing_success': False,
                    'error_message': "Analysis timed out after 30 minutes"
                }
            })
            print(f"✗ {opening_range}-minute analysis timed out")

        except Exception as e:
            result.update({
                'end_time': datetime.now().isoformat(),
                'success': False,
                'metrics': {
                    'processing_success': False,
                    'error_message': f"Execution error: {str(e)}"
                }
            })
            print(f"✗ {opening_range}-minute analysis failed: {e}")

        return result

    def analyze_results(self) -> str:
        """
        Perform comparative analysis of results across different opening range periods.

        Returns:
            Formatted analysis report as string
        """
        analysis = []
        analysis.append("=" * 80)
        analysis.append("ORB OPENING RANGE IMPACT ANALYSIS")
        analysis.append("=" * 80)
        analysis.append(f"Analysis conducted: {self.analysis_timestamp}")
        analysis.append("")

        # Summary table
        analysis.append("SUMMARY OF RESULTS")
        analysis.append("-" * 50)
        analysis.append(f"{'Period':<8} {'Success':<8} {'Symbols':<8} {'Rows':<10} {'PCA Comp':<9} {'Total Var %':<12} {'1st Comp %':<10}")
        analysis.append("-" * 50)

        successful_results = []

        for period in self.opening_range_periods:
            if period in self.results:
                result = self.results[period]
                metrics = result.get('metrics', {})

                success = "✓" if result.get('success', False) else "✗"
                symbols = metrics.get('total_symbols', 0)
                rows = f"{metrics.get('total_rows', 0):,}"
                components = metrics.get('pca_components', 0)
                total_var = f"{metrics.get('total_variance_explained', 0):.1f}"
                first_var = f"{metrics.get('first_component_variance', 0):.1f}"

                analysis.append(f"{period:<8} {success:<8} {symbols:<8} {rows:<10} {components:<9} {total_var:<12} {first_var:<10}")

                if result.get('success', False):
                    successful_results.append((period, metrics))
            else:
                analysis.append(f"{period:<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<9} {'N/A':<12} {'N/A':<10}")

        analysis.append("")

        # Detailed analysis if we have successful results
        if successful_results:
            analysis.append("DETAILED COMPARATIVE ANALYSIS")
            analysis.append("-" * 40)

            # Variance explained analysis
            analysis.append("1. PCA Variance Explained Comparison:")
            var_data = [(period, metrics['total_variance_explained']) for period, metrics in successful_results]
            var_data.sort(key=lambda x: x[1], reverse=True)

            for i, (period, variance) in enumerate(var_data):
                rank = i + 1
                analysis.append(f"   {rank}. {period} minutes: {variance:.2f}% total variance")

            best_period, best_variance = var_data[0]
            analysis.append(f"   → Best performing period: {best_period} minutes ({best_variance:.2f}%)")
            analysis.append("")

            # First component analysis
            analysis.append("2. First Principal Component Strength:")
            first_comp_data = [(period, metrics['first_component_variance']) for period, metrics in successful_results]
            first_comp_data.sort(key=lambda x: x[1], reverse=True)

            for i, (period, variance) in enumerate(first_comp_data):
                rank = i + 1
                analysis.append(f"   {rank}. {period} minutes: {variance:.2f}% first component")

            strongest_period, strongest_variance = first_comp_data[0]
            analysis.append(f"   → Strongest first component: {strongest_period} minutes ({strongest_variance:.2f}%)")
            analysis.append("")

            # Data efficiency analysis
            analysis.append("3. Data Processing Efficiency:")
            for period, metrics in successful_results:
                symbols = metrics.get('total_symbols', 0)
                components = metrics.get('pca_components', 0)
                if symbols > 0:
                    efficiency = components / symbols * 100
                    analysis.append(f"   {period} minutes: {components} components from {symbols} symbols ({efficiency:.1f}% efficiency)")
            analysis.append("")

            # Pattern strength trends
            analysis.append("4. Opening Range Period Impact Assessment:")

            # Calculate correlations and trends
            periods = [period for period, _ in successful_results]
            total_vars = [metrics['total_variance_explained'] for _, metrics in successful_results]
            first_vars = [metrics['first_component_variance'] for _, metrics in successful_results]

            if len(periods) >= 3:
                # Simple trend analysis
                if total_vars == sorted(total_vars):
                    trend = "increasing"
                elif total_vars == sorted(total_vars, reverse=True):
                    trend = "decreasing"
                else:
                    trend = "mixed"

                analysis.append(f"   Total variance trend with longer periods: {trend}")

                # Find optimal range
                max_var_period = periods[total_vars.index(max(total_vars))]
                analysis.append(f"   Optimal period for maximum variance explained: {max_var_period} minutes")

                # Pattern strength assessment
                avg_total_var = sum(total_vars) / len(total_vars)
                strong_periods = [p for p, v in zip(periods, total_vars) if v > avg_total_var]

                if strong_periods:
                    analysis.append(f"   Above-average periods: {', '.join(map(str, strong_periods))} minutes")

            analysis.append("")

            # Recommendations
            analysis.append("5. RECOMMENDATIONS:")
            if successful_results:
                best_overall = max(successful_results, key=lambda x: x[1]['total_variance_explained'])
                best_period = best_overall[0]
                best_metrics = best_overall[1]

                analysis.append(f"   → RECOMMENDED OPENING RANGE: {best_period} minutes")
                analysis.append(f"     - Captures {best_metrics['total_variance_explained']:.2f}% of total variance")
                analysis.append(f"     - First component explains {best_metrics['first_component_variance']:.2f}%")
                analysis.append(f"     - Processes {best_metrics['total_symbols']} symbols effectively")

                # Secondary recommendations
                if len(successful_results) > 1:
                    second_best = sorted(successful_results, key=lambda x: x[1]['total_variance_explained'], reverse=True)[1]
                    analysis.append(f"   → ALTERNATIVE: {second_best[0]} minutes ({second_best[1]['total_variance_explained']:.2f}% variance)")

        else:
            analysis.append("No successful analyses to compare.")

        analysis.append("")
        analysis.append("=" * 80)

        return "\n".join(analysis)

    def generate_plots(self) -> None:
        """
        Generate visualization plots comparing different opening range periods.
        """
        if not self.results:
            print("No results available for plotting")
            return

        # Extract successful results for plotting
        successful_results = []
        for period in self.opening_range_periods:
            if period in self.results and self.results[period].get('success', False):
                metrics = self.results[period]['metrics']
                successful_results.append((period, metrics))

        if len(successful_results) < 2:
            print("Need at least 2 successful results for meaningful plots")
            return

        # Create plots directory if it doesn't exist
        plots_dir = 'orb_analysis_plots'
        os.makedirs(plots_dir, exist_ok=True)

        # Extract data for plotting
        periods = [period for period, _ in successful_results]
        total_vars = [metrics['total_variance_explained'] for _, metrics in successful_results]
        first_vars = [metrics['first_component_variance'] for _, metrics in successful_results]
        components = [metrics['pca_components'] for _, metrics in successful_results]
        symbols = [metrics['total_symbols'] for _, metrics in successful_results]
        rows = [metrics['total_rows'] for _, metrics in successful_results]

        # Set up the plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        # Create a comprehensive figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ORB Opening Range Period Impact Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Total Variance Explained vs Opening Range Period
        ax1.plot(periods, total_vars, 'o-', linewidth=2, markersize=8, color='navy', alpha=0.8)
        ax1.set_xlabel('Opening Range Period (minutes)')
        ax1.set_ylabel('Total Variance Explained (%)')
        ax1.set_title('PCA Total Variance Explained')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(min(total_vars) * 0.98, max(total_vars) * 1.02)

        # Add value labels on points
        for i, (period, var) in enumerate(zip(periods, total_vars)):
            ax1.annotate(f'{var:.2f}%', (period, var), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)

        # Plot 2: First Component Variance vs Opening Range Period
        ax2.plot(periods, first_vars, 's-', linewidth=2, markersize=8, color='darkred', alpha=0.8)
        ax2.set_xlabel('Opening Range Period (minutes)')
        ax2.set_ylabel('First Component Variance (%)')
        ax2.set_title('First Principal Component Strength')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(min(first_vars) * 0.95, max(first_vars) * 1.05)

        # Add value labels on points
        for i, (period, var) in enumerate(zip(periods, first_vars)):
            ax2.annotate(f'{var:.2f}%', (period, var), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)

        # Plot 3: Comparison Bar Chart
        x_pos = np.arange(len(periods))
        width = 0.35

        ax3.bar(x_pos - width/2, total_vars, width, label='Total Variance %', 
               color='navy', alpha=0.7)
        ax3.bar(x_pos + width/2, first_vars, width, label='First Component %', 
               color='darkred', alpha=0.7)

        ax3.set_xlabel('Opening Range Period (minutes)')
        ax3.set_ylabel('Variance Explained (%)')
        ax3.set_title('Variance Comparison by Period')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{p} min' for p in periods])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (total, first) in enumerate(zip(total_vars, first_vars)):
            ax3.annotate(f'{total:.1f}', (i - width/2, total), 
                        textcoords="offset points", xytext=(0,3), 
                        ha='center', fontsize=8)
            ax3.annotate(f'{first:.1f}', (i + width/2, first), 
                        textcoords="offset points", xytext=(0,3), 
                        ha='center', fontsize=8)

        # Plot 4: Data Efficiency Analysis
        efficiency = [comp/sym * 100 for comp, sym in zip(components, symbols)]

        ax4.bar(periods, efficiency, color='green', alpha=0.7, width=2)
        ax4.set_xlabel('Opening Range Period (minutes)')
        ax4.set_ylabel('PCA Efficiency (Components/Symbols %)')
        ax4.set_title('Data Processing Efficiency')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for period, eff in zip(periods, efficiency):
            ax4.annotate(f'{eff:.1f}%', (period, eff), 
                        textcoords="offset points", xytext=(0,3), 
                        ha='center', fontsize=9)

        # Adjust layout and save
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, 'orb_analysis_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plots saved to: {plot_file}")

        # Create a detailed variance trend plot
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot both metrics on same chart with different y-axes
        color1 = 'tab:blue'
        ax.set_xlabel('Opening Range Period (minutes)')
        ax.set_ylabel('Total Variance Explained (%)', color=color1)
        line1 = ax.plot(periods, total_vars, 'o-', color=color1, linewidth=3, 
                       markersize=10, label='Total Variance')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, alpha=0.3)

        # Create second y-axis for first component
        ax2 = ax.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('First Component Variance (%)', color=color2)
        line2 = ax2.plot(periods, first_vars, 's-', color=color2, linewidth=3, 
                        markersize=10, label='First Component')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add title and annotations
        ax.set_title('ORB Opening Range Period Impact on PCA Performance', 
                    fontsize=14, fontweight='bold', pad=20)

        # Add value annotations
        for period, total, first in zip(periods, total_vars, first_vars):
            ax.annotate(f'{total:.2f}%', (period, total), 
                       textcoords="offset points", xytext=(0,15), 
                       ha='center', fontsize=10, color=color1, fontweight='bold')
            ax2.annotate(f'{first:.2f}%', (period, first), 
                        textcoords="offset points", xytext=(0,-20), 
                        ha='center', fontsize=10, color=color2, fontweight='bold')

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        # Save detailed plot
        detail_plot_file = os.path.join(plots_dir, 'orb_variance_trend.png')
        plt.savefig(detail_plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Detailed variance trend plot saved to: {detail_plot_file}")

        # Create summary statistics plot
        fig3, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Calculate statistics
        mean_total = np.mean(total_vars)
        std_total = np.std(total_vars)
        mean_first = np.mean(first_vars)
        std_first = np.std(first_vars)

        # Plot with error bars
        ax.errorbar(periods, total_vars, yerr=std_total, 
                   fmt='o-', linewidth=2, markersize=8, capsize=5,
                   label=f'Total Variance (μ={mean_total:.2f}%, σ={std_total:.3f})')
        ax.errorbar(periods, first_vars, yerr=std_first, 
                   fmt='s-', linewidth=2, markersize=8, capsize=5,
                   label=f'First Component (μ={mean_first:.2f}%, σ={std_first:.3f})')

        ax.set_xlabel('Opening Range Period (minutes)')
        ax.set_ylabel('Variance Explained (%)')
        ax.set_title('ORB Analysis: Variance Statistics by Period')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add horizontal lines for means
        ax.axhline(y=mean_total, color='blue', linestyle='--', alpha=0.5)
        ax.axhline(y=mean_first, color='orange', linestyle='--', alpha=0.5)

        stats_plot_file = os.path.join(plots_dir, 'orb_statistics.png')
        plt.savefig(stats_plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Statistics plot saved to: {stats_plot_file}")

        plt.close('all')  # Close all figures to free memory

        # Generate summary text file with plot locations
        summary_file = os.path.join(plots_dir, 'plot_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("ORB ANALYSIS VISUALIZATION SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("GENERATED PLOTS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. orb_analysis_comparison.png - Four-panel comprehensive comparison\n")
            f.write("2. orb_variance_trend.png - Detailed variance trend analysis\n")
            f.write("3. orb_statistics.png - Statistical analysis with error bars\n\n")

            f.write("KEY FINDINGS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Variance Range: {min(total_vars):.2f}% - {max(total_vars):.2f}%\n")
            f.write(f"First Component Range: {min(first_vars):.2f}% - {max(first_vars):.2f}%\n")
            f.write(f"Most Effective Period: {periods[total_vars.index(max(total_vars))]} minutes\n")
            f.write(f"Strongest First Component: {periods[first_vars.index(max(first_vars))]} minutes\n")

        print(f"✓ Plot summary saved to: {summary_file}")
        print(f"\n📊 Generated {len(os.listdir(plots_dir))} visualization files in '{plots_dir}' directory")

    def generate_detailed_report(self, output_file: str) -> None:
        """
        Generate comprehensive report with all results and analysis.

        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w') as f:
            f.write("ORB OPENING RANGE IMPACT ANALYSIS - DETAILED REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {self.analysis_timestamp}\n")
            f.write(f"Script: {__file__}\n\n")

            # Executive Summary
            f.write(self.analyze_results())
            f.write("\n\n")

            # Detailed results for each period
            f.write("DETAILED EXECUTION RESULTS\n")
            f.write("=" * 50 + "\n\n")

            for period in self.opening_range_periods:
                f.write(f"OPENING RANGE PERIOD: {period} MINUTES\n")
                f.write("-" * 40 + "\n")

                if period in self.results:
                    result = self.results[period]

                    # Basic info
                    f.write(f"Command: {result.get('command', 'N/A')}\n")
                    f.write(f"Start Time: {result.get('start_time', 'N/A')}\n")
                    f.write(f"Duration: {result.get('duration_seconds', 0):.1f} seconds\n")
                    f.write(f"Exit Code: {result.get('exit_code', 'N/A')}\n")
                    f.write(f"Success: {result.get('success', False)}\n\n")

                    # Metrics
                    metrics = result.get('metrics', {})
                    f.write("PCA Analysis Metrics:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

                    # Error information if failed
                    if not result.get('success', False):
                        f.write("Error Information:\n")
                        f.write(f"  Error Message: {metrics.get('error_message', 'Unknown error')}\n")
                        if result.get('stderr'):
                            f.write(f"  Stderr: {result['stderr']}\n")
                        f.write("\n")

                    # Stdout (first 2000 chars to avoid huge files)
                    stdout = result.get('stdout', '')
                    if stdout:
                        f.write("Execution Output (first 2000 characters):\n")
                        f.write("-" * 30 + "\n")
                        f.write(stdout[:2000])
                        if len(stdout) > 2000:
                            f.write(f"\n... (truncated, full output was {len(stdout)} characters)")
                        f.write("\n\n")
                else:
                    f.write("No results available for this period.\n\n")

                f.write("\n")

    def run_complete_analysis(self) -> bool:
        """
        Run the complete ORB analysis across all opening range periods.

        Returns:
            True if analysis completed, False if setup failed
        """
        print("=" * 60)
        print("ORB OPENING RANGE IMPACT ANALYSIS")
        print("=" * 60)
        print(f"Analysis started: {self.analysis_timestamp}")

        # Discover date range
        print("\n1. Discovering available date range...")
        start_date, end_date = self.discover_date_range()

        if not start_date or not end_date:
            print("✗ No valid date range found in data directory")
            return False

        print(f"✓ Found date range: {start_date} to {end_date}")
        print(f"  Total analysis period: {(end_date - start_date).days + 1} days")

        # Run analysis for each opening range period
        print(f"\n2. Running analysis for {len(self.opening_range_periods)} opening range periods...")

        for i, period in enumerate(self.opening_range_periods, 1):
            print(f"\n--- Analysis {i}/{len(self.opening_range_periods)}: {period} minutes ---")

            result = self.run_orb_analysis(start_date, end_date, period)
            self.results[period] = result

            # Brief status
            if result['success']:
                metrics = result['metrics']
                symbols = metrics.get('total_symbols', 0)
                variance = metrics.get('total_variance_explained', 0)
                print(f"  Result: {symbols} symbols, {variance:.1f}% variance explained")
            else:
                error_msg = result['metrics'].get('error_message', 'Unknown error')
                print(f"  Result: FAILED - {error_msg}")

        # Generate reports
        print("\n3. Generating analysis reports...")

        # Console summary
        print("\n" + self.analyze_results())

        # Generate visualizations
        print("\n4. Generating visualization plots...")
        self.generate_plots()

        # Detailed file report
        output_file = 'orb_analysis_results.txt'
        self.generate_detailed_report(output_file)
        print(f"\n✓ Detailed report saved to: {output_file}")

        # Success summary
        successful_count = sum(1 for result in self.results.values() if result.get('success', False))
        print(f"\n📊 Analysis Summary:")
        print(f"   - Total periods tested: {len(self.opening_range_periods)}")
        print(f"   - Successful analyses: {successful_count}")
        print(f"   - Date range: {start_date} to {end_date}")
        print(f"   - Results file: {output_file}")

        return True


def main():
    """Main execution function."""
    try:
        analyzer = ORBAnalyzer()
        success = analyzer.run_complete_analysis()

        if success:
            print("\n✓ ORB opening range impact analysis completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ ORB analysis failed during setup")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()