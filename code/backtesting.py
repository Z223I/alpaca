#!/usr/bin/env python3
"""
Backtesting System - Trade Generator

This system performs parametric backtesting of ORB trading strategies by:
1. Running nested loops with different parameter combinations
2. Executing historical ORB pipeline simulations for each parameter set
3. Generating performance metrics and visualization reports

Usage:
    python3 code/backtesting.py                    # Run full backtesting suite
    python3 code/backtesting.py --dry-run          # Show what would be run without executing
    python3 code/backtesting.py --verbose          # Show detailed output
"""

import sys
import json
import shutil
import subprocess
import uuid
import argparse
import logging
import pandas as pd
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from atoms.alerts.config import (  # noqa: E402
    DEFAULT_PLOTS_ROOT_DIR, DEFAULT_DATA_ROOT_DIR, DEFAULT_LOGS_ROOT_DIR,
    DEFAULT_HISTORICAL_ROOT_DIR, DEFAULT_PRICE_MOMENTUM_CONFIG
)
from atoms.telegram.telegram_post import TelegramPoster  # noqa: E402


class BacktestingSystem:
    """
    Main backtesting system that manages parametric testing of ORB strategies.
    """

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.run_results = []
        self.config_restored = False

        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Load parameters and symbols
        self.parameters = self._load_parameters()
        self.symbols_by_date = self._load_symbols_by_date()

        self.logger.info(f"Loaded {len(self.parameters['trend_analysis_timeframe_minutes'])} timeframe values")
        self.logger.info(f"Loaded {len(self.parameters['green_threshold'])} threshold values")
        self.logger.info(f"Found {len(self.symbols_by_date)} dates with active symbols")

    def _load_parameters(self) -> Dict:
        """Load parametric testing values from parameters.json"""
        params_file = Path("data/backtesting/parameters.json")
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")

        with open(params_file, 'r') as f:
            return json.load(f)

    def _load_symbols_by_date(self) -> Dict[str, List[str]]:
        """Load symbols from symbols.json filtered by active='yes' and grouped by date"""
        symbols_file = Path("data/backtesting/symbols.json")
        if not symbols_file.exists():
            raise FileNotFoundError(f"Symbols file not found: {symbols_file}")

        with open(symbols_file, 'r') as f:
            data = json.load(f)

        symbols_by_date = {}
        for item in data['symbols']:
            if item['active'] == 'yes':
                date = item['date']
                symbol = item['symbol']
                if date not in symbols_by_date:
                    symbols_by_date[date] = []
                symbols_by_date[date].append(symbol)

        return symbols_by_date

    def _send_run_notification(self, target_run_dir: Path, timeframe: int, threshold: float,
                               date: str, symbol: str):
        """Send Telegram notification at the start of each backtesting run"""
        if self.dry_run:
            self.logger.info(f"Would send Telegram notification for run: {target_run_dir}")
            return

        try:
            telegram_poster = TelegramPoster()

            message = (f"🧪 **Backtesting Run Started**\n\n"
                       f"**Target Directory:** `{target_run_dir.name}`\n"
                       f"**Date:** {date}\n"
                       f"**Timeframe:** {timeframe} minutes\n"
                       f"**Green Threshold:** {threshold}\n"
                       f"**Symbol:** {symbol}\n\n"
                       f"Running ORB pipeline simulation...")

            result = telegram_poster.send_message_to_user(message, "bruce", urgent=False)

            if result['success']:
                self.logger.info("✅ Sent run notification to Bruce via Telegram")
            else:
                self.logger.warning(f"❌ Failed to send run notification: "
                                    f"{result.get('errors', [])}")

        except Exception as e:
            self.logger.error(f"Error sending run notification via Telegram: {e}")

    def _signal_handler(self, signum, _frame):
        """Handle CTRL+C and other termination signals"""
        self.logger.info(f"Received signal {signum}, initiating clean shutdown...")
        self._restore_original_config()
        sys.exit(0)

    def _setup_current_run_config(self):
        """Copy config_current_run.py to config.py at startup"""
        config_current_run = Path("atoms/alerts/config_current_run.py")
        config_active = Path("atoms/alerts/config.py")
        
        if not config_current_run.exists():
            raise FileNotFoundError(f"Current run config not found: {config_current_run}")
            
        if not self.dry_run:
            shutil.copy2(config_current_run, config_active)
            # Ensure the runs/current directory exists
            current_run_dir = Path("./runs/current")
            current_run_dir.mkdir(parents=True, exist_ok=True)
            
        self.logger.info("✅ Configured all processes to use ./runs/current directory")

    def _restore_original_config(self):
        """Copy config_orig.py to config.py at shutdown"""
        if self.config_restored:
            return  # Already restored
            
        config_orig = Path("atoms/alerts/config_orig.py")
        config_active = Path("atoms/alerts/config.py")
        
        if not config_orig.exists():
            self.logger.error(f"Original config not found: {config_orig}")
            return
            
        if not self.dry_run:
            shutil.copy2(config_orig, config_active)
            
        self.config_restored = True
        self.logger.info("✅ Restored original config file")

    def _update_config_for_run(self, timeframe: int, threshold: float):
        """Update the current run config file with run-specific parameters"""
        config_current_run = Path("atoms/alerts/config_current_run.py")
        
        if self.dry_run:
            self.logger.info(f"Would update config with timeframe={timeframe}, threshold={threshold}")
            return

        # Read the template config
        with open(config_current_run, 'r') as f:
            content = f.read()

        # Update momentum config parameters using line-based replacement
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('DEFAULT_PRICE_MOMENTUM_CONFIG = '):
                lines[i] = (f'DEFAULT_PRICE_MOMENTUM_CONFIG = PriceMomentumConfig('
                           f'momentum=MomentumThresholds(green_threshold={threshold}), '
                           f'trend_analysis_timeframe_minutes={timeframe})')
                break
        
        # Write back to both current_run config and active config
        updated_content = '\n'.join(lines)
        
        # Update current_run config (template for next startup)
        with open(config_current_run, 'w') as f:
            f.write(updated_content)
            
        # Update active config (what processes will use)
        config_active = Path("atoms/alerts/config.py")
        with open(config_active, 'w') as f:
            f.write(updated_content)
        
        # Allow file system to propagate changes
        import time
        time.sleep(0.5)
        
        self.logger.info(f"✅ Config updated: timeframe={timeframe}, threshold={threshold}")

    def _get_target_run_directory(self, symbol: str, date: str, timeframe: int, threshold: float) -> Path:
        """Get target run directory name for this combination of parameters"""
        run_id = str(uuid.uuid4())[:8]  # Short UUID
        return Path(f"runs/{date}/{symbol}/run_{date}_tf{timeframe}_th{threshold}_{run_id}")
    
    def _move_current_to_target(self, target_dir: Path) -> bool:
        """Move ./runs/current to target directory using bash mv command"""
        current_dir = Path("./runs/current")
        
        if self.dry_run:
            self.logger.info(f"Would move {current_dir} to {target_dir}")
            return True
            
        if not current_dir.exists():
            self.logger.warning(f"Current run directory does not exist: {current_dir}")
            return False
        
        try:
            # Ensure parent directory exists
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Use bash mv command to move the directory
            result = subprocess.run(
                ["mv", str(current_dir), str(target_dir)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Moved {current_dir} to {target_dir}")
                return True
            else:
                self.logger.error(f"❌ Failed to move directory: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error moving directory: {e}")
            return False

    def _prepare_symbols_file(self, date: str, symbol: str) -> Path:
        """Prepare symbols CSV file for the specific date and single symbol in current run directory"""
        # Convert date format from YYYY-MM-DD to YYYYMMDD for CSV filename
        csv_date = date.replace('-', '')
        source_csv = Path(f"data/{csv_date}.csv")
        current_run_dir = Path("./runs/current")
        target_csv = current_run_dir / "data" / f"{csv_date}.csv"

        if not source_csv.exists():
            self.logger.warning(f"Source CSV not found: {source_csv}")
            return target_csv

        if self.dry_run:
            self.logger.info(f"Would copy and filter {source_csv} to {target_csv} for symbol {symbol}")
            return target_csv

        # Ensure target directory exists
        target_csv.parent.mkdir(parents=True, exist_ok=True)

        # Read source CSV and filter by single symbol
        df = pd.read_csv(source_csv)
        if 'Symbol' in df.columns:
            filtered_df = df[df['Symbol'] == symbol]
        elif 'symbol' in df.columns:
            filtered_df = df[df['symbol'] == symbol]
        else:
            self.logger.warning(f"No Symbol/symbol column found in {source_csv}")
            filtered_df = df

        # Save filtered CSV
        filtered_df.to_csv(target_csv, index=False)
        self.logger.info(f"Created filtered symbols file: {target_csv} (1 symbol: {symbol})")
        return target_csv

    def _run_orb_pipeline(self, date: str, symbols_file: Path, symbol: str) -> bool:
        """Execute the ORB pipeline with concurrent processes using file watchers"""
        # Define all processes that should run concurrently
        # Testing with live superduper alerts to test trade processor directly
        processes = [
            {
                'name': 'simulator',
                'cmd': f"python3 code/orb_pipeline_simulator.py --symbols-file {symbols_file} --date {date} "
                       f"--save-alerts --speed 1 --verbose",
                'primary': True  # This drives the pipeline, others watch for its output
            },
            # Skip regular monitor and superduper monitor since we're copying superduper alerts directly
            # {
            #     'name': 'monitor',
            #     'cmd': f"python3 code/orb_alerts_monitor.py --symbols-file {symbols_file} --date {date} "
            #            f"--no-telegram --verbose",
            #     'primary': False  # Watches for simulator alerts
            # },
            {
                'name': 'superduper_monitor',
                'cmd': f"python3 code/orb_alerts_monitor_superduper.py --date {date} --verbose",
                'primary': False  # Watches for super_alerts (will be copied)
            },
            {
                'name': 'trade_processor',
                'cmd': f"python3 code/orb_alerts_trade_stocks.py --date {date} --no-telegram --verbose",
                'primary': False  # Watches for superduper alerts (will be copied)
            },
            {
                'name': 'super_alert_copier',
                'cmd': f"python3 code/copy_super_alerts.py {date} ./runs/current {symbol}",
                'primary': False  # Copies existing super alerts for the current symbol at 1 per 3 seconds
            }
        ]

        if self.dry_run:
            for proc in processes:
                self.logger.info(f"Would start {proc['name']}: {proc['cmd']}")
            return True

        # Start all processes concurrently
        running_processes = []

        try:
            for proc_info in processes:
                self.logger.info(f"Starting {proc_info['name']}: {proc_info['cmd']}")

                process = subprocess.Popen(
                    proc_info['cmd'].split(),
                    cwd=Path.cwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                running_processes.append({
                    'process': process,
                    'info': proc_info
                })

            # Wait for the primary process (simulator) to complete first
            # The others will process files as they're created by the simulator
            simulator_process = next(p for p in running_processes if p['info']['name'] == 'simulator')

            self.logger.info("Waiting for pipeline simulator to complete...")
            try:
                _stdout, stderr = simulator_process['process'].communicate(timeout=1800)  # 30 min timeout

                if simulator_process['process'].returncode != 0:
                    self.logger.error(f"Pipeline simulator failed: {stderr}")
                    return False

                self.logger.info("Pipeline simulator completed successfully")

            except subprocess.TimeoutExpired:
                self.logger.error("Pipeline simulator timed out")
                simulator_process['process'].kill()
                return False

            # Give watchers time to process all generated files and monitor progress
            import time
            self.logger.info("Monitoring file processing...")
            print("Monitoring file processing...")
            
            # Track last three count sets to detect when processing is complete
            previous_counts = []
            consecutive_identical = 0
            max_iterations = 12  # Maximum 2 minutes as fallback
            current_run_dir = Path("./runs/current")
            
            for i in range(max_iterations):
                alert_count = len(list((current_run_dir / "historical_data" / date / "alerts" / "bullish").glob("*.json")))
                super_count = len(list((current_run_dir / "historical_data" / date / "super_alerts" / "bullish").glob("*.json")))
                superduper_count = len(list((current_run_dir / "historical_data" / date / "superduper_alerts" / "bullish").glob("*.json")))
                superduper_green_count = len(list((current_run_dir / "historical_data" / date / "superduper_alerts_sent" / "bullish" / "green").glob("*.json")))
                trade_count = len(list((current_run_dir / "historical_data" / date).glob("*trade*.json")))
                
                current_counts = (alert_count, super_count, superduper_count, superduper_green_count, trade_count)
                progress_msg = f"Files: {alert_count} alerts → {super_count} super → {superduper_count} superduper → {superduper_green_count} green → {trade_count} trades"
                self.logger.info(progress_msg)
                print(progress_msg)
                
                # Check for consecutive identical counts
                if previous_counts and current_counts == previous_counts[-1]:
                    consecutive_identical += 1
                    if consecutive_identical >= 2:  # Three consecutive identical (current + 2 previous)
                        self.logger.info("✅ File processing complete - counts stabilized")
                        print("✅ File processing complete - counts stabilized")
                        break
                else:
                    consecutive_identical = 0
                
                previous_counts.append(current_counts)
                if len(previous_counts) > 2:  # Keep only last 2 for comparison
                    previous_counts.pop(0)
                
                if superduper_green_count > 0:
                    self.logger.info("✅ Superduper green alerts detected!")
                    print("✅ Superduper green alerts detected!")
                    if trade_count > 0:
                        self.logger.info("✅ Trades detected!")
                        print("✅ Trades detected!")
                        # Don't break here - let it stabilize with identical counts
                
                if i < max_iterations - 1:  # Don't sleep after the last iteration
                    time.sleep(10)

            # Terminate all remaining processes gracefully
            for proc in running_processes:
                if proc['process'].poll() is None:  # Still running
                    self.logger.info(f"Terminating {proc['info']['name']}")
                    proc['process'].terminate()
                    try:
                        proc['process'].communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc['process'].kill()

            return True

        except Exception as e:
            self.logger.error(f"Pipeline execution error: {e}")

            # Clean up any remaining processes
            for proc in running_processes:
                if proc['process'].poll() is None:
                    proc['process'].kill()

            return False

    def _generate_symbol_plots(self, date: str, symbol: str) -> bool:
        """Generate plots for the symbol"""
        plot_cmd = f"python3 code/alpaca.py --plot --symbol {symbol} --date {date}"

        if self.dry_run:
            self.logger.info(f"Would run: {plot_cmd}")
            return True

        try:
            result = subprocess.run(
                plot_cmd.split(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per plot
            )

            if result.returncode != 0:
                self.logger.warning(f"Plot generation failed for {symbol}: {result.stderr}")
                return False

        except Exception as e:
            self.logger.warning(f"Plot error for {symbol}: {e}")
            return False

        return True

    def _analyze_run_results(self, run_dir: Path) -> Dict:
        """Analyze results from a completed run"""
        results = {
            'superduper_alerts': 0,
            'trades': 0,
            'symbols_processed': 0,
            'alerts_by_symbol': {},
            'trades_by_symbol': {}
        }

        if self.dry_run:
            return results

        # Count superduper alerts from sent directory (these were actually sent via Telegram)
        superduper_dir = run_dir / "historical_data"
        if superduper_dir.exists():
            for alert_file in superduper_dir.rglob("**/superduper_alerts_sent/**/*.json"):
                try:
                    with open(alert_file, 'r') as f:
                        alert_data = json.load(f)
                        if isinstance(alert_data, list):
                            results['superduper_alerts'] += len(alert_data)
                            # Track by symbol for each alert in list
                            for alert in alert_data:
                                symbol = alert.get('symbol', 'UNKNOWN')
                                results['alerts_by_symbol'][symbol] = results['alerts_by_symbol'].get(symbol, 0) + 1
                        else:
                            results['superduper_alerts'] += 1
                            # Track by symbol for single alert
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

    def _create_summary_charts(self):
        """Create summary pie charts for alerts and trades by date, and bar chart by symbol"""
        if self.dry_run or not self.run_results:
            return

        # Prepare data for charts - now scanning symbol subdirectories
        alerts_by_date = {}
        trades_by_date = {}
        alerts_by_symbol = {}

        for result in self.run_results:
            date = result['date']
            symbol = result.get('symbol', 'UNKNOWN')
            alerts_by_date[date] = alerts_by_date.get(date, 0) + result['superduper_alerts']
            trades_by_date[date] = trades_by_date.get(date, 0) + result['trades']
            
            # Aggregate symbol data from individual symbol runs
            alerts_by_symbol[symbol] = alerts_by_symbol.get(symbol, 0) + result['superduper_alerts']
            
            # Also include any detailed symbol breakdown from results
            for sym, count in result.get('alerts_by_symbol', {}).items():
                alerts_by_symbol[sym] = alerts_by_symbol.get(sym, 0) + count

        # Create runs directory if it doesn't exist
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create alerts by symbol pie chart
        if alerts_by_symbol and sum(alerts_by_symbol.values()) > 0:
            plt.figure(figsize=(10, 8))
            plt.pie(alerts_by_symbol.values(), labels=alerts_by_symbol.keys(), autopct='%1.1f%%', startangle=90)
            plt.title('Superduper Alerts by Symbol', fontsize=16, fontweight='bold')
            alerts_filename = runs_dir / f"summary_alerts_by_symbol_pie_{timestamp}.png"
            plt.savefig(alerts_filename, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Created {alerts_filename}")

        # Create trades by date pie chart
        if trades_by_date and sum(trades_by_date.values()) > 0:
            plt.figure(figsize=(10, 8))
            plt.pie(trades_by_date.values(), labels=trades_by_date.keys(), autopct='%1.1f%%')
            plt.title('Trades by Date')
            trades_filename = runs_dir / f"summary_trades_by_date_{timestamp}.png"
            plt.savefig(trades_filename)
            plt.close()
            self.logger.info(f"Created {trades_filename}")

        # Create alerts by symbol bar chart
        if alerts_by_symbol and sum(alerts_by_symbol.values()) > 0:
            plt.figure(figsize=(12, 8))
            symbols = list(alerts_by_symbol.keys())
            counts = list(alerts_by_symbol.values())
            
            bars = plt.bar(symbols, counts, color='steelblue', alpha=0.7)
            plt.title('Superduper Alerts by Symbol', fontsize=16, fontweight='bold')
            plt.xlabel('Symbol', fontsize=12)
            plt.ylabel('Number of Alerts', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels on top of bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            symbols_filename = runs_dir / f"summary_alerts_by_symbol_bar_{timestamp}.png"
            plt.savefig(symbols_filename, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Created {symbols_filename}")

    def _run_parameter_analysis(self):
        """Run parameter optimization analysis at the end of backtesting."""
        if self.dry_run:
            self.logger.info("Would run parameter optimization analysis")
            return

        self.logger.info("🚀 Running parameter optimization analysis...")
        
        try:
            # Import and run the analysis tool
            import subprocess
            
            analysis_cmd = "python3 code/analyze_backtesting_results.py"
            result = subprocess.run(
                analysis_cmd.split(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Parameter optimization analysis completed successfully")
                # Log the output for visibility
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        self.logger.info(f"   {line}")
            else:
                self.logger.error(f"❌ Parameter analysis failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.error("⏰ Parameter analysis timed out")
        except Exception as e:
            self.logger.error(f"💥 Parameter analysis error: {e}")

    def _calculate_total_runs(self):
        """Calculate the total number of runs for progress tracking"""
        total_runs = 0
        for timeframe in self.parameters['trend_analysis_timeframe_minutes']:
            for threshold in self.parameters['green_threshold']:
                for date, symbols in self.symbols_by_date.items():
                    for symbol in symbols:
                        total_runs += 1
        return total_runs

    def run_backtesting(self):
        """Execute the full backtesting suite"""
        self.logger.info("Starting backtesting system")

        try:
            # Calculate total runs for progress tracking
            total_runs = self._calculate_total_runs()
            self.logger.info(f"📊 Total runs to execute: {total_runs}")
            
            # Setup current run config at startup
            self._setup_current_run_config()

            # Nested loops for parametric testing - now with symbol isolation
            current_run = 0
            for timeframe in self.parameters['trend_analysis_timeframe_minutes']:
                for threshold in self.parameters['green_threshold']:
                    for date, symbols in self.symbols_by_date.items():
                        for symbol in symbols:  # Process each symbol separately
                            current_run += 1


                            self.logger.info(f"🏃 Run {current_run} of {total_runs} - Processing: timeframe={timeframe}, threshold={threshold}, date={date}, symbol={symbol}")

                            # Get target directory name for this run
                            target_run_dir = self._get_target_run_directory(symbol, date, timeframe, threshold)

                            # Update config for this run (parameters only)
                            self._update_config_for_run(timeframe, threshold)

                            # Send run notification
                            self._send_run_notification(target_run_dir, timeframe, threshold, date, symbol)

                            # Prepare symbols file for single symbol in current directory
                            symbols_file = self._prepare_symbols_file(date, symbol)

                            # Run ORB pipeline (writes to ./runs/current)
                            success = self._run_orb_pipeline(date, symbols_file, symbol)

                            if success:
                                # Generate plots for this symbol
                                # self._generate_symbol_plots(date, symbol)  # Temporarily commented out

                                # Move current directory to target with symbol isolation
                                move_success = self._move_current_to_target(target_run_dir)
                                
                                if move_success:
                                    # Analyze results from moved directory
                                    results = self._analyze_run_results(target_run_dir)
                                    results.update({
                                        'date': date,
                                        'timeframe': timeframe,
                                        'threshold': threshold,
                                        'symbol': symbol,
                                        'run_dir': str(target_run_dir)
                                    })
                                    self.run_results.append(results)

                                    self.logger.info(f"✅ Run {current_run} of {total_runs} completed for {symbol}: {results['superduper_alerts']} alerts, "
                                                     f"{results['trades']} trades")
                                else:
                                    self.logger.error(f"❌ Run {current_run} of {total_runs} failed to move run directory for {date}/{symbol}")
                            else:
                                self.logger.error(f"❌ Run {current_run} of {total_runs} failed for {date}/{symbol}")

            # Create summary reports
            self._create_summary_charts()

            # Run parameter optimization analysis
            self._run_parameter_analysis()

            # Print summary
            total_alerts = sum(r['superduper_alerts'] for r in self.run_results)
            total_trades = sum(r['trades'] for r in self.run_results)
            self.logger.info(f"Backtesting complete: {len(self.run_results)} runs, "
                             f"{total_alerts} total alerts, {total_trades} total trades")

        finally:
            # Always restore original config
            self._restore_original_config()


def main():
    parser = argparse.ArgumentParser(description='ORB Backtesting System')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    backtester = BacktestingSystem(dry_run=args.dry_run, verbose=args.verbose)
    backtester.run_backtesting()


if __name__ == "__main__":
    main()
