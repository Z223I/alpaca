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
        self.config_backup_path = None
        self.run_results = []

        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

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

    def _send_run_notification(self, run_dir: Path, timeframe: int, threshold: float,
                               date: str, symbols: List[str]):
        """Send Telegram notification at the start of each backtesting run"""
        if self.dry_run:
            self.logger.info(f"Would send Telegram notification for run: {run_dir}")
            return

        try:
            telegram_poster = TelegramPoster()

            symbols_str = ", ".join(symbols)
            message = (f"🧪 **Backtesting Run Started**\n\n"
                       f"**Run Directory:** `{run_dir.name}`\n"
                       f"**Date:** {date}\n"
                       f"**Timeframe:** {timeframe} minutes\n"
                       f"**Green Threshold:** {threshold}\n"
                       f"**Symbols:** {symbols_str} ({len(symbols)} total)\n\n"
                       f"Running ORB pipeline simulation...")

            result = telegram_poster.send_message_to_user(message, "bruce", urgent=False)

            if result['success']:
                self.logger.info("✅ Sent run notification to Bruce via Telegram")
            else:
                self.logger.warning(f"❌ Failed to send run notification: "
                                    f"{result.get('errors', [])}")

        except Exception as e:
            self.logger.error(f"Error sending run notification via Telegram: {e}")

    def _backup_config(self):
        """Create backup of atoms/alerts/config.py"""
        config_path = Path("atoms/alerts/config.py")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config_backup_path = config_path.with_suffix('.py.backup')
        if not self.dry_run:
            shutil.copy2(config_path, self.config_backup_path)

        self.logger.info(f"Backed up config to: {self.config_backup_path}")

    def _restore_config(self):
        """Restore backup of atoms/alerts/config.py"""
        if self.config_backup_path and self.config_backup_path.exists():
            config_path = Path("atoms/alerts/config.py")
            if not self.dry_run:
                shutil.copy2(self.config_backup_path, config_path)
                self.config_backup_path.unlink()  # Remove backup file
            self.logger.info("Restored original config file")

    def _update_config_for_run(self, run_dir: Path, timeframe: int, threshold: float):
        """Update atoms/alerts/config.py with run-specific settings and force reload"""
        config_path = Path("atoms/alerts/config.py")

        if self.dry_run:
            self.logger.info(f"Would update config for run: {run_dir}")
            return

        # Read current config
        with open(config_path, 'r') as f:
            content = f.read()

        # Update default instances to point to run directory
        run_dir_str = str(run_dir.absolute())

        # Replace the default instances - handle both patterns
        import re
        
        # Replace PLOTS_ROOT_DIR line
        content = re.sub(
            r'DEFAULT_PLOTS_ROOT_DIR = PlotsRootDir\([^)]*\)',
            f'DEFAULT_PLOTS_ROOT_DIR = PlotsRootDir(root_path="{run_dir_str}")',
            content
        )
        
        # Replace DATA_ROOT_DIR line
        content = re.sub(
            r'DEFAULT_DATA_ROOT_DIR = DataRootDir\([^)]*\)',
            f'DEFAULT_DATA_ROOT_DIR = DataRootDir(root_path="{run_dir_str}")',
            content
        )
        
        # Replace LOGS_ROOT_DIR line
        content = re.sub(
            r'DEFAULT_LOGS_ROOT_DIR = LogsRootDir\([^)]*\)',
            f'DEFAULT_LOGS_ROOT_DIR = LogsRootDir(root_path="{run_dir_str}")',
            content
        )
        
        # Replace HISTORICAL_ROOT_DIR line
        content = re.sub(
            r'DEFAULT_HISTORICAL_ROOT_DIR = HistoricalRootDir\([^)]*\)',
            f'DEFAULT_HISTORICAL_ROOT_DIR = HistoricalRootDir(root_path="{run_dir_str}")',
            content
        )

        # Update momentum config with parameters - use line-based replacement for robustness
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('DEFAULT_PRICE_MOMENTUM_CONFIG = '):
                lines[i] = (f'DEFAULT_PRICE_MOMENTUM_CONFIG = PriceMomentumConfig('
                           f'momentum=MomentumThresholds(green_threshold={threshold}), '
                           f'trend_analysis_timeframe_minutes={timeframe})')
                break
        content = '\n'.join(lines)

        # Write updated config
        with open(config_path, 'w') as f:
            f.write(content)

        # Allow file system to propagate changes
        import time
        time.sleep(0.5)
        
        # Force Python to reload the config module so all processes see the new values
        self._reload_config_module()
        
        # Verify the update worked
        self._validate_config_update(run_dir, timeframe, threshold)
        
        # Additional delay to ensure subprocesses pick up new config
        time.sleep(1.0)

        self.logger.info(f"✅ Config updated and validated: timeframe={timeframe}, threshold={threshold}, run_dir={run_dir}")
    
    def _reload_config_module(self):
        """Force reload of config module to pick up file changes"""
        try:
            import importlib
            import sys
            
            # Remove config module from cache if it exists
            if 'atoms.alerts.config' in sys.modules:
                del sys.modules['atoms.alerts.config']
            
            # Clear any cached imports
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('atoms.alerts.config'):
                    del sys.modules[module_name]
            
            # Force reimport
            import atoms.alerts.config
            importlib.reload(atoms.alerts.config)
            
            self.logger.info("🔄 Config module reloaded successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Config reload failed: {e}")
    
    def _validate_config_update(self, expected_run_dir: Path, expected_timeframe: int, expected_threshold: float):
        """Validate that config update worked correctly"""
        try:
            # Import fresh config
            from atoms.alerts.config import (
                DEFAULT_HISTORICAL_ROOT_DIR, DEFAULT_PRICE_MOMENTUM_CONFIG,
                DEFAULT_PLOTS_ROOT_DIR, DEFAULT_DATA_ROOT_DIR, DEFAULT_LOGS_ROOT_DIR
            )
            
            # Check directory paths
            expected_path = str(expected_run_dir.absolute())
            
            actual_historical = str(DEFAULT_HISTORICAL_ROOT_DIR.root_path)
            actual_plots = str(DEFAULT_PLOTS_ROOT_DIR.root_path)
            actual_data = str(DEFAULT_DATA_ROOT_DIR.root_path)
            actual_logs = str(DEFAULT_LOGS_ROOT_DIR.root_path)
            
            # Check momentum config
            actual_timeframe = DEFAULT_PRICE_MOMENTUM_CONFIG.trend_analysis_timeframe_minutes
            actual_threshold = DEFAULT_PRICE_MOMENTUM_CONFIG.momentum.green_threshold
            
            # Validate all paths match
            path_errors = []
            if actual_historical != expected_path:
                path_errors.append(f"HISTORICAL: expected {expected_path}, got {actual_historical}")
            if actual_plots != expected_path:
                path_errors.append(f"PLOTS: expected {expected_path}, got {actual_plots}")
            if actual_data != expected_path:
                path_errors.append(f"DATA: expected {expected_path}, got {actual_data}")
            if actual_logs != expected_path:
                path_errors.append(f"LOGS: expected {expected_path}, got {actual_logs}")
            
            # Validate momentum config
            config_errors = []
            if actual_timeframe != expected_timeframe:
                config_errors.append(f"TIMEFRAME: expected {expected_timeframe}, got {actual_timeframe}")
            if actual_threshold != expected_threshold:
                config_errors.append(f"THRESHOLD: expected {expected_threshold}, got {actual_threshold}")
            
            if path_errors or config_errors:
                error_msg = "❌ Config validation failed!\n"
                if path_errors:
                    error_msg += "Path errors:\n" + "\n".join(f"  - {e}" for e in path_errors) + "\n"
                if config_errors:
                    error_msg += "Config errors:\n" + "\n".join(f"  - {e}" for e in config_errors)
                
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.logger.info("✅ Config validation passed - all paths and parameters correct")
            
        except ImportError as e:
            error_msg = f"❌ Config validation failed - import error: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _create_run_directory(self, date: str) -> Path:
        """Create run directory with format ./runs/run_YYYY-MM-DD_<UUID>"""
        run_id = str(uuid.uuid4())[:8]  # Short UUID
        run_dir = Path(f"runs/run_{date}_{run_id}")

        if not self.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            # Create required subdirectories
            (run_dir / "data").mkdir(exist_ok=True)
            (run_dir / "plots").mkdir(exist_ok=True)
            (run_dir / "logs").mkdir(exist_ok=True)
            (run_dir / "historical_data").mkdir(exist_ok=True)

        self.logger.info(f"Created run directory: {run_dir}")
        return run_dir

    def _prepare_symbols_file(self, date: str, symbols: List[str], run_dir: Path) -> Path:
        """Prepare symbols CSV file for the specific date and symbols"""
        # Convert date format from YYYY-MM-DD to YYYYMMDD for CSV filename
        csv_date = date.replace('-', '')
        source_csv = Path(f"data/{csv_date}.csv")
        target_csv = run_dir / "data" / f"{csv_date}.csv"

        if not source_csv.exists():
            self.logger.warning(f"Source CSV not found: {source_csv}")
            return target_csv

        if self.dry_run:
            self.logger.info(f"Would copy and filter {source_csv} to {target_csv}")
            return target_csv

        # Read source CSV and filter by symbols
        df = pd.read_csv(source_csv)
        if 'Symbol' in df.columns:
            filtered_df = df[df['Symbol'].isin(symbols)]
        elif 'symbol' in df.columns:
            filtered_df = df[df['symbol'].isin(symbols)]
        else:
            self.logger.warning(f"No Symbol/symbol column found in {source_csv}")
            filtered_df = df

        # Save filtered CSV
        filtered_df.to_csv(target_csv, index=False)
        self.logger.info(f"Created filtered symbols file: {target_csv} ({len(filtered_df)} symbols)")
        return target_csv

    def _run_orb_pipeline(self, date: str, symbols_file: Path, run_dir: Path) -> bool:
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
                'cmd': f"python3 code/copy_super_alerts.py {date} {run_dir}",
                'primary': False  # Copies existing VERB super alerts at 1 per 3 seconds
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
                stdout, stderr = simulator_process['process'].communicate(timeout=1800)  # 30 min timeout

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
            
            for i in range(12):  # Check every 10 seconds for 2 minutes
                alert_count = len(list((run_dir / "historical_data" / date / "alerts" / "bullish").glob("*.json")))
                super_count = len(list((run_dir / "historical_data" / date / "super_alerts" / "bullish").glob("*.json")))
                superduper_count = len(list((run_dir / "historical_data" / date / "superduper_alerts" / "bullish").glob("*.json")))
                superduper_green_count = len(list((run_dir / "historical_data" / date / "superduper_alerts_sent" / "bullish" / "green").glob("*.json")))
                trade_count = len(list((run_dir / "historical_data" / date).glob("*trade*.json")))
                
                progress_msg = f"Files: {alert_count} alerts → {super_count} super → {superduper_count} superduper → {superduper_green_count} green → {trade_count} trades"
                self.logger.info(progress_msg)
                print(progress_msg)
                
                if superduper_green_count > 0:
                    self.logger.info("✅ Superduper green alerts detected!")
                    print("✅ Superduper green alerts detected!")
                    if trade_count > 0:
                        self.logger.info("✅ Trades detected!")
                        print("✅ Trades detected!")
                        break
                
                if i < 11:  # Don't sleep after the last iteration
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

    def _generate_symbol_plots(self, date: str, symbols: List[str]) -> bool:
        """Generate plots for each symbol"""
        for symbol in symbols:
            plot_cmd = f"python3 code/alpaca.py --plot --symbol {symbol} --date {date}"

            if self.dry_run:
                self.logger.info(f"Would run: {plot_cmd}")
                continue

            try:
                result = subprocess.run(
                    plot_cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per plot
                )

                if result.returncode != 0:
                    self.logger.warning(f"Plot generation failed for {symbol}: {result.stderr}")

            except Exception as e:
                self.logger.warning(f"Plot error for {symbol}: {e}")

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

        # Count superduper alerts
        superduper_dir = run_dir / "historical_data"
        if superduper_dir.exists():
            for alert_file in superduper_dir.rglob("*superduper_alerts*/*.json"):
                try:
                    with open(alert_file, 'r') as f:
                        alert_data = json.load(f)
                        if isinstance(alert_data, list):
                            results['superduper_alerts'] += len(alert_data)
                        else:
                            results['superduper_alerts'] += 1
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
        """Create summary pie charts for alerts and trades by date"""
        if self.dry_run or not self.run_results:
            return

        # Prepare data for charts
        alerts_by_date = {}
        trades_by_date = {}

        for result in self.run_results:
            date = result['date']
            alerts_by_date[date] = alerts_by_date.get(date, 0) + result['superduper_alerts']
            trades_by_date[date] = trades_by_date.get(date, 0) + result['trades']

        # Create runs directory if it doesn't exist
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create alerts by date pie chart
        if alerts_by_date and sum(alerts_by_date.values()) > 0:
            plt.figure(figsize=(10, 8))
            plt.pie(alerts_by_date.values(), labels=alerts_by_date.keys(), autopct='%1.1f%%')
            plt.title('Superduper Alerts by Date')
            alerts_filename = runs_dir / f"summary_alerts_by_date_{timestamp}.png"
            plt.savefig(alerts_filename)
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

    def run_backtesting(self):
        """Execute the full backtesting suite"""
        self.logger.info("Starting backtesting system")

        try:
            # Backup config
            self._backup_config()

            # Nested loops for parametric testing
            for timeframe in self.parameters['trend_analysis_timeframe_minutes']:
                for threshold in self.parameters['green_threshold']:
                    for date, symbols in self.symbols_by_date.items():

                        self.logger.info(f"Processing: timeframe={timeframe}, threshold={threshold}, date={date}")

                        # Create run directory
                        run_dir = self._create_run_directory(date)

                        # Update config for this run
                        self._update_config_for_run(run_dir, timeframe, threshold)

                        # Send run notification
                        self._send_run_notification(run_dir, timeframe, threshold, date, symbols)

                        # Prepare symbols file
                        symbols_file = self._prepare_symbols_file(date, symbols, run_dir)

                        # Run ORB pipeline
                        success = self._run_orb_pipeline(date, symbols_file, run_dir)

                        if success:
                            # Generate plots
                            self._generate_symbol_plots(date, symbols)

                            # Analyze results
                            results = self._analyze_run_results(run_dir)
                            results.update({
                                'date': date,
                                'timeframe': timeframe,
                                'threshold': threshold,
                                'run_dir': str(run_dir),
                                'symbols': symbols
                            })
                            self.run_results.append(results)

                            self.logger.info(f"Run completed: {results['superduper_alerts']} alerts, "
                                             f"{results['trades']} trades")
                        else:
                            self.logger.error(f"Run failed for {date}")

            # Create summary reports
            self._create_summary_charts()

            # Print summary
            total_alerts = sum(r['superduper_alerts'] for r in self.run_results)
            total_trades = sum(r['trades'] for r in self.run_results)
            self.logger.info(f"Backtesting complete: {len(self.run_results)} runs, "
                             f"{total_alerts} total alerts, {total_trades} total trades")

        finally:
            # Always restore config
            self._restore_config()


def main():
    parser = argparse.ArgumentParser(description='ORB Backtesting System')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    backtester = BacktestingSystem(dry_run=args.dry_run, verbose=args.verbose)
    backtester.run_backtesting()


if __name__ == "__main__":
    main()
