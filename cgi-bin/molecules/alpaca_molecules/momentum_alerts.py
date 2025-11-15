#!/usr/bin/env python3
"""
Momentum Alerts System

This system monitors stocks from multiple sources (market open gainers, premarket gainers,
volume surge, Oracle data) and generates momentum alerts based on VWAP and EMA9 criteria.
It follows the specification in specs/momentum_alert.md.

Process:
1. Premarket Scanning: Run premarket_top_gainers.py from 04:00 to 20:00 ET every 20 minutes
   - Runs on separate CPU core (core 1) using taskset
   - Generates top_gainers_nasdaq_amex.csv in ./historical_data/{YYYY-MM-DD}/market/
   - Monitors premarket top gainers continuously
   - Runs every day including weekends for continuous data collection
2. Market Open Scanning: Run market_open_top_gainers.py starting at 9:40 ET, every 20 minutes
   - Generates gainers_nasdaq_amex.csv
   - Runs every day including weekends for continuous data collection
   - Automatically reschedules after each run
3. Volume Surge: Run volume surge scanner hourly at :45 from 06:45 to 12:45 ET on weekdays only
   - Runs at: 06:45, 07:45, 08:45, 09:45, 10:45, 11:45, 12:45 ET (7 runs per weekday)
   - Generates relative_volume_nasdaq_amex.csv
   - Automatically reschedules after each run
4. Data Integration: Load symbols from all sources (premarket, market open, volume surge, Oracle)
   - Track source with boolean fields: from_premarket, from_gainers, from_volume_surge, oracle
   - Merge and deduplicate symbols across all sources
5. Stock Monitoring: Every minute, collect 30 minutes of 1-minute candlesticks for each stock
6. Momentum Alerts: Check stocks above VWAP, above EMA9, and pass urgency filter
7. Integration: Send alerts to all users with momentum_alerts=true via Telegram

CSV Files Generated:
- ./historical_data/{YYYY-MM-DD}/market/top_gainers_nasdaq_amex.csv (premarket)
- ./historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv (market open)
- ./historical_data/{YYYY-MM-DD}/volume_surge/relative_volume_nasdaq_amex.csv
- ./historical_data/{YYYY-MM-DD}/scanner/symbol_list.csv

Usage:
    python3 code/momentum_alerts.py
    python3 code/momentum_alerts.py --test
    python3 code/momentum_alerts.py --verbose
"""

import asyncio
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import pytz

# Add project root to path
# This file is at: cgi-bin/molecules/alpaca_molecules/momentum_alerts.py
# Need to go up 3 levels to reach project root
script_dir = os.path.dirname(os.path.abspath(__file__))  # cgi-bin/molecules/alpaca_molecules/
molecules_dir = os.path.dirname(script_dir)  # cgi-bin/molecules/
cgi_bin_dir = os.path.dirname(molecules_dir)  # cgi-bin/
project_root = os.path.dirname(cgi_bin_dir)  # project root
sys.path.insert(0, project_root)

import alpaca_trade_api as tradeapi
from atoms.api.init_alpaca_client import init_alpaca_client
from atoms.api.stock_halt_detector import is_stock_halted, get_halt_status_emoji
from atoms.api.fundamental_data import FundamentalDataFetcher
from atoms.alerts.breakout_detector import BreakoutDetector
from atoms.telegram.telegram_post import TelegramPoster
from atoms.telegram.user_manager import UserManager

# Import yfinance-based fetcher using direct import to avoid namespace conflict
import importlib.util
cgi_api_atoms_dir = os.path.join(cgi_bin_dir, 'api', 'atoms', 'alpaca_api')
yfinance_fetcher_path = os.path.join(cgi_api_atoms_dir, 'fundamental_data.py')
spec = importlib.util.spec_from_file_location("yfinance_fundamental_data", yfinance_fetcher_path)
yfinance_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yfinance_module)
YFinanceFetcher = yfinance_module.FundamentalDataFetcher
from momentum_alerts_config import (
    get_momentum_alerts_config, get_volume_color_emoji,
    get_momentum_standard_color_emoji, get_momentum_short_color_emoji,
    get_urgency_level_dual, get_squeeze_emoji
)


class MomentumAlertsSystem:
    """Main momentum alerts system orchestrator."""

    def __init__(self, test_mode: bool = False, verbose: bool = False):
        """
        Initialize Momentum Alerts System.

        Args:
            test_mode: Run in test mode (no actual alerts)
            verbose: Enable verbose logging
        """
        # Setup logging
        self.logger = self._setup_logging(verbose)
        self.test_mode = test_mode
        self.verbose = verbose

        # Eastern Time zone for all operations
        self.et_tz = pytz.timezone('US/Eastern')

        # Get today's date for file monitoring
        self.today = datetime.now(self.et_tz).strftime('%Y-%m-%d')

        # Historical data directory path
        self.historical_data_dir = Path("historical_data") / self.today / "market"
        self.csv_file_path = self.historical_data_dir / "gainers_nasdaq_amex.csv"

        # Momentum alerts data directories
        self.momentum_alerts_dir = Path("historical_data") / self.today / "momentum_alerts" / "bullish"
        self.momentum_alerts_sent_dir = Path("historical_data") / self.today / "momentum_alerts_sent" / "bullish"

        # Volume surge data directories
        self.volume_surge_dir = Path("historical_data") / self.today / "volume_surge"

        # Create momentum alert directories
        self.momentum_alerts_dir.mkdir(parents=True, exist_ok=True)
        self.momentum_alerts_sent_dir.mkdir(parents=True, exist_ok=True)

        # Create volume surge directory
        self.volume_surge_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Alpaca client for stock data
        self.historical_client = None
        try:
            self.historical_client = init_alpaca_client(
                provider="alpaca",
                account="Bruce",
                environment="paper"
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize historical data client: {e}")

        # Initialize components
        self.breakout_detector = BreakoutDetector()
        self.telegram_poster = TelegramPoster()
        self.user_manager = UserManager()
        self.momentum_config = get_momentum_alerts_config()

        # Use yfinance-based fetcher for float shares (replaces Polygon/Yahoo fallback)
        self.fundamental_fetcher = YFinanceFetcher(verbose=verbose)

        # Tracking - store dict with symbol metadata including market_open_price
        self.monitored_symbols: Dict[str, Dict] = {}
        self.last_csv_check = None
        self.startup_runs_completed = 0
        self.startup_schedule = []  # List of scheduled startup times

        # Volume surge data
        self.volume_surge_csv_path = self.volume_surge_dir / "relative_volume_nasdaq_amex.csv"
        self.volume_surge_completed = False
        self.volume_surge_schedule = []  # List of scheduled volume surge times
        self.volume_surge_runs_completed = 0

        # Scanner data directory
        self.scanner_dir = Path("historical_data") / self.today / "scanner"
        self.scanner_dir.mkdir(parents=True, exist_ok=True)
        self.symbol_list_csv_path = self.scanner_dir / "symbol_list.csv"

        # Manual symbols data file (JSON)
        self.manual_symbols_file = Path("data") / "manual_symbols.json"
        self.last_manual_symbols_check = None

        # Premarket data directory
        self.premarket_csv_path = self.historical_data_dir / "top_gainers_nasdaq_amex.csv"
        self.premarket_schedule = []  # List of scheduled premarket times
        self.premarket_runs_completed = 0

        # State
        self.running = False
        self.startup_processes = {}  # Track running startup scripts
        self.premarket_processes = {}  # Track running premarket scripts

        # Debug mode - send random alerts for testing
        self.debug_mode = False  # Set to False to disable debug alerts
        self.debug_symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'GOOGL', 'MSFT', 'AMZN', 'META']
        self.last_debug_alert_time = None

        self.logger.info(f"🔧 Momentum Alerts System initialized in {'TEST' if test_mode else 'LIVE'} mode")
        if self.debug_mode:
            self.logger.warning(f"⚠️ DEBUG MODE ENABLED - Will send random test alerts every minute")
        self.logger.info(f"📅 Monitoring date: {self.today}")
        self.logger.info(f"📁 CSV file path: {self.csv_file_path}")
        self.logger.info(f"📊 Historical data client: "
                         f"{'Available' if self.historical_client else 'Not available'}")
        self.logger.info(f"⚙️ Momentum periods: {self.momentum_config.momentum_period}min / "
                         f"{self.momentum_config.momentum_short_period}min")

    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Setup logging configuration with Eastern Time."""
        class EasternFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                et_tz = pytz.timezone('US/Eastern')
                et_time = datetime.fromtimestamp(record.created, et_tz)
                if datefmt:
                    return et_time.strftime(datefmt)
                return et_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] + ' ET'

        logger = logging.getLogger(__name__)
        if not logger.handlers:
            # Setup console handler
            console_handler = logging.StreamHandler()
            formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Setup file handler
            try:
                log_dir = Path("logs") / "momentum_alerts"
                log_dir.mkdir(parents=True, exist_ok=True)

                et_tz = pytz.timezone('US/Eastern')
                log_filename = f"momentum_alerts_{datetime.now(et_tz).strftime('%Y%m%d_%H%M%S')}.log"
                log_file_path = log_dir / log_filename

                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")

            logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        return logger

    def _schedule_startup_runs(self):
        """
        Schedule the startup script to run starting at 9:40 ET, then every 20 minutes.

        Runs every day (including weekends) for continuous data collection.
        Simple approach: Calculate next run time based on current time.
        """
        current_time = datetime.now(self.et_tz)

        # Calculate next run time starting at 9:40 ET, every 20 minutes
        # Base time is 9:40 ET (hour=9, minute=40)

        # Get minutes since midnight
        current_minutes_since_midnight = current_time.hour * 60 + current_time.minute

        # First run at 9:40 ET = 580 minutes since midnight
        first_run_minutes = 9 * 60 + 40  # 580 minutes

        # If before 9:40 today, schedule for 9:40 today
        if current_minutes_since_midnight < first_run_minutes:
            next_run = current_time.replace(hour=9, minute=40, second=0, microsecond=0)
        else:
            # Calculate how many 20-minute intervals have passed since 9:40 today
            minutes_since_940 = current_minutes_since_midnight - first_run_minutes
            intervals_passed = minutes_since_940 // 20

            # Next run is the next 20-minute interval
            next_interval_minutes = first_run_minutes + ((intervals_passed + 1) * 20)

            # If we've gone past midnight (into next day), schedule for 9:40 tomorrow
            if next_interval_minutes >= 24 * 60:
                next_run = (current_time + timedelta(days=1)).replace(hour=9, minute=40, second=0, microsecond=0)
            else:
                next_run_hour = next_interval_minutes // 60
                next_run_minute = next_interval_minutes % 60
                next_run = current_time.replace(hour=next_run_hour, minute=next_run_minute, second=0, microsecond=0)

        # Store the next scheduled run
        self.startup_schedule = [next_run]

        self.logger.info(f"📅 Next startup script run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S ET')}")
        self.logger.info(f"⏰ Runs every 20 minutes starting at 9:40 ET daily (including weekends)")

    def _schedule_premarket_runs(self):
        """
        Schedule the premarket script to run from 04:00 to 20:00 ET, every 20 minutes.

        Runs every day (including weekends) for continuous data collection.
        Simple approach: Calculate next run time based on current time.
        """
        current_time = datetime.now(self.et_tz)

        # Calculate next run time from 04:00 to 20:00 ET, every 20 minutes
        # Base time is 04:00 ET (hour=4, minute=0)

        # Get minutes since midnight
        current_minutes_since_midnight = current_time.hour * 60 + current_time.minute

        # First run at 04:00 ET = 240 minutes since midnight
        first_run_minutes = 4 * 60  # 240 minutes

        # Last run at 20:00 ET = 1200 minutes since midnight
        last_run_minutes = 20 * 60  # 1200 minutes

        # If before 04:00 today, schedule for 04:00 today
        if current_minutes_since_midnight < first_run_minutes:
            next_run = current_time.replace(hour=4, minute=0, second=0, microsecond=0)
        elif current_minutes_since_midnight >= last_run_minutes:
            # After 20:00 today, schedule for 04:00 tomorrow
            next_run = (current_time + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
        else:
            # Between 04:00 and 20:00 - calculate how many 20-minute intervals have passed since 04:00 today
            minutes_since_0400 = current_minutes_since_midnight - first_run_minutes
            intervals_passed = minutes_since_0400 // 20

            # Next run is the next 20-minute interval
            next_interval_minutes = first_run_minutes + ((intervals_passed + 1) * 20)

            # If we've gone past 20:00, schedule for 04:00 tomorrow
            if next_interval_minutes >= last_run_minutes:
                next_run = (current_time + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
            else:
                next_run_hour = next_interval_minutes // 60
                next_run_minute = next_interval_minutes % 60
                next_run = current_time.replace(hour=next_run_hour, minute=next_run_minute, second=0, microsecond=0)

        # Store the next scheduled run
        self.premarket_schedule = [next_run]

        self.logger.info(f"🌅 Next premarket script run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S ET')}")
        self.logger.info(f"⏰ Runs every 20 minutes from 04:00 to 20:00 ET daily (including weekends)")

    def _schedule_volume_surge_runs(self):
        """
        Schedule the volume surge scanner to run hourly at :45 from 06:45 to 12:45 ET on weekdays only.

        Runs: 06:45, 07:45, 08:45, 09:45, 10:45, 11:45, 12:45 ET (7 runs per weekday)
        """
        current_time = datetime.now(self.et_tz)

        # Check if today is a weekday (Monday=0, Sunday=6)
        is_weekday = current_time.weekday() < 5

        # Volume surge runs at :45 of each hour from 06:45 to 12:45 ET
        # First run at 06:45 ET = 405 minutes since midnight (6*60 + 45)
        # Last run at 12:45 ET = 765 minutes since midnight (12*60 + 45)

        # Get minutes since midnight
        current_minutes_since_midnight = current_time.hour * 60 + current_time.minute

        # Define run times (in minutes since midnight)
        run_times = [
            6 * 60 + 45,   # 06:45 (405)
            7 * 60 + 45,   # 07:45 (465)
            8 * 60 + 45,   # 08:45 (525)
            9 * 60 + 45,   # 09:45 (585)
            10 * 60 + 45,  # 10:45 (645)
            11 * 60 + 45,  # 11:45 (705)
            12 * 60 + 45   # 12:45 (765)
        ]

        if not is_weekday:
            # If today is weekend, schedule for next Monday at 06:45
            days_until_monday = (7 - current_time.weekday()) % 7
            if days_until_monday == 0:  # Today is Sunday
                days_until_monday = 1

            next_run = (current_time + timedelta(days=days_until_monday)).replace(
                hour=6, minute=45, second=0, microsecond=0
            )
        elif current_minutes_since_midnight < run_times[0]:
            # Before 06:45 today, schedule for 06:45 today
            next_run = current_time.replace(hour=6, minute=45, second=0, microsecond=0)
        elif current_minutes_since_midnight >= run_times[-1]:
            # After 12:45 today, schedule for 06:45 tomorrow (if tomorrow is weekday)
            tomorrow = current_time + timedelta(days=1)
            if tomorrow.weekday() < 5:  # Tomorrow is weekday
                next_run = tomorrow.replace(hour=6, minute=45, second=0, microsecond=0)
            else:
                # Tomorrow is weekend, find next Monday
                days_until_monday = (7 - tomorrow.weekday()) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                next_run = (tomorrow + timedelta(days=days_until_monday)).replace(
                    hour=6, minute=45, second=0, microsecond=0
                )
        else:
            # Between 06:45 and 12:45 - find next scheduled time
            next_run_minutes = None
            for run_minute in run_times:
                if current_minutes_since_midnight < run_minute:
                    next_run_minutes = run_minute
                    break

            if next_run_minutes:
                next_run_hour = next_run_minutes // 60
                next_run_minute = next_run_minutes % 60
                next_run = current_time.replace(
                    hour=next_run_hour, minute=next_run_minute, second=0, microsecond=0
                )
            else:
                # Shouldn't reach here, but fallback to next day
                tomorrow = current_time + timedelta(days=1)
                if tomorrow.weekday() < 5:
                    next_run = tomorrow.replace(hour=6, minute=45, second=0, microsecond=0)
                else:
                    days_until_monday = (7 - tomorrow.weekday()) % 7
                    if days_until_monday == 0:
                        days_until_monday = 7
                    next_run = (tomorrow + timedelta(days=days_until_monday)).replace(
                        hour=6, minute=45, second=0, microsecond=0
                    )

        # Store the next scheduled run
        self.volume_surge_schedule = [next_run]

        self.logger.info(f"📈 Next volume surge scanner run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S ET')}")
        self.logger.info(f"⏰ Runs hourly at :45 from 06:45 to 12:45 ET on weekdays only")

    async def _run_startup_script(self) -> bool:
        """
        Run the market_open_top_gainers.py script.

        Returns:
            True if script ran successfully, False otherwise
        """
        self.logger.info("🚀 Running startup script: market_open_top_gainers.py")

        script_path = Path("code") / "market_open_top_gainers.py"
        cmd = [
            "~/miniconda3/envs/alpaca/bin/python",
            str(script_path),
            "--exchanges", "NASDAQ", "AMEX",
            "--max-symbols", "7000",
            "--min-price", "0.75",
            "--max-price", "100.00",
            "--min-volume", "250000",
            "--top-gainers", "40",
            "--export-csv", "gainers_nasdaq_amex.csv",
            "--verbose"
        ]

        try:
            # Expand the tilde in the Python path
            cmd[0] = os.path.expanduser(cmd[0])

            # Create environment with PYTHONPATH set to project root
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root

            # Create process with logging
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path.cwd()),
                env=env
            )

            # Store process for monitoring
            process_id = f"startup_{self.startup_runs_completed + 1}"
            self.startup_processes[process_id] = {
                'process': process,
                'start_time': datetime.now(self.et_tz),
                'cmd': ' '.join(cmd)
            }

            self.logger.info(f"📊 Started startup script (PID: {process.pid})")
            self.logger.info("🕒 Expected completion: up to 20 minutes")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start startup script: {e}")
            return False

    async def _run_premarket_script(self) -> bool:
        """
        Run the premarket_top_gainers.py script on a separate CPU core.

        Returns:
            True if script ran successfully, False otherwise
        """
        self.logger.info("🌅 Running premarket script: premarket_top_gainers.py")

        script_path = Path("code") / "premarket_top_gainers.py"
        cmd = [
            "taskset", "-c", "1",  # Run on CPU core 1 (separate from main process)
            os.path.expanduser("~/miniconda3/envs/alpaca/bin/python"),
            str(script_path),
            "--account", "live",  # Use live account with SIP feed access
            "--exchanges", "NASDAQ", "AMEX",
            "--max-symbols", "7000",
            "--min-price", "0.75",
            "--max-price", "40.00",
            "--min-volume", "250000",
            "--top-gainers", "40",
            "--export-csv", "top_gainers_nasdaq_amex.csv",
            "--verbose"
        ]

        try:
            # Create environment with PYTHONPATH set to project root
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root

            # Create process with logging
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path.cwd()),
                env=env
            )

            # Store process for monitoring
            process_id = f"premarket_{self.premarket_runs_completed + 1}"
            self.premarket_processes[process_id] = {
                'process': process,
                'start_time': datetime.now(self.et_tz),
                'cmd': ' '.join(cmd)
            }

            self.logger.info(f"🌅 Started premarket script on CPU core 1 (PID: {process.pid})")
            self.logger.info("🕒 Expected completion: up to 20 minutes")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start premarket script: {e}")
            return False

    def _check_startup_schedule(self):
        """
        Check if it's time to run a startup script.

        After each run, automatically schedule the next one (every 20 minutes).
        """
        current_time = datetime.now(self.et_tz)

        # Check if we have a scheduled run and it's time to execute
        if self.startup_schedule and current_time >= self.startup_schedule[0]:
            # Run the script
            asyncio.create_task(self._run_startup_script())
            self.startup_runs_completed += 1

            # Immediately schedule the next run (20 minutes from now)
            next_run = current_time + timedelta(minutes=20)
            # Align to the next 20-minute boundary based on 9:40 start
            # Round to nearest 20-minute mark: 9:40, 10:00, 10:20, 10:40, etc.
            minutes_since_midnight = next_run.hour * 60 + next_run.minute
            first_run_minutes = 9 * 60 + 40  # 580 minutes (9:40)

            if minutes_since_midnight < first_run_minutes:
                # Before 9:40, schedule for 9:40
                next_run = next_run.replace(hour=9, minute=40, second=0, microsecond=0)
            else:
                # After 9:40, round to next 20-minute interval from 9:40
                minutes_since_940 = minutes_since_midnight - first_run_minutes
                intervals_from_940 = (minutes_since_940 // 20) + 1
                next_interval_minutes = first_run_minutes + (intervals_from_940 * 20)

                if next_interval_minutes >= 24 * 60:
                    # Past midnight, schedule for 9:40 tomorrow
                    next_run = (next_run + timedelta(days=1)).replace(hour=9, minute=40, second=0, microsecond=0)
                else:
                    next_run_hour = next_interval_minutes // 60
                    next_run_minute = next_interval_minutes % 60
                    next_run = next_run.replace(hour=next_run_hour, minute=next_run_minute, second=0, microsecond=0)

            self.startup_schedule = [next_run]
            self.logger.info(f"⏰ Next startup script run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S ET')}")

    def _check_premarket_schedule(self):
        """
        Check if it's time to run a premarket script.

        After each run, automatically schedule the next one (every 20 minutes).
        Only runs between 04:00 and 20:00 ET.
        """
        current_time = datetime.now(self.et_tz)

        # Check if we have a scheduled run and it's time to execute
        if self.premarket_schedule and current_time >= self.premarket_schedule[0]:
            # Run the script
            asyncio.create_task(self._run_premarket_script())
            self.premarket_runs_completed += 1

            # Immediately schedule the next run (20 minutes from now)
            next_run = current_time + timedelta(minutes=20)
            # Align to the next 20-minute boundary based on 04:00 start
            # Round to nearest 20-minute mark: 04:00, 04:20, 04:40, 05:00, etc.
            minutes_since_midnight = next_run.hour * 60 + next_run.minute
            first_run_minutes = 4 * 60  # 240 minutes (04:00)
            last_run_minutes = 20 * 60  # 1200 minutes (20:00)

            if minutes_since_midnight < first_run_minutes:
                # Before 04:00, schedule for 04:00
                next_run = next_run.replace(hour=4, minute=0, second=0, microsecond=0)
            elif minutes_since_midnight >= last_run_minutes:
                # After 20:00, schedule for 04:00 tomorrow
                next_run = (next_run + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
            else:
                # Between 04:00 and 20:00, round to next 20-minute interval from 04:00
                minutes_since_0400 = minutes_since_midnight - first_run_minutes
                intervals_from_0400 = (minutes_since_0400 // 20) + 1
                next_interval_minutes = first_run_minutes + (intervals_from_0400 * 20)

                if next_interval_minutes >= last_run_minutes:
                    # Past 20:00, schedule for 04:00 tomorrow
                    next_run = (next_run + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0)
                else:
                    next_run_hour = next_interval_minutes // 60
                    next_run_minute = next_interval_minutes % 60
                    next_run = next_run.replace(hour=next_run_hour, minute=next_run_minute, second=0, microsecond=0)

            self.premarket_schedule = [next_run]
            self.logger.info(f"🌅 Next premarket script run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S ET')}")

    def _check_volume_surge_schedule(self):
        """
        Check if it's time to run a volume surge scanner.

        After each run, automatically schedule the next one (hourly at :45).
        Only runs from 06:45 to 12:45 ET on weekdays.
        """
        current_time = datetime.now(self.et_tz)

        # Check if we have a scheduled run and it's time to execute
        if self.volume_surge_schedule and current_time >= self.volume_surge_schedule[0]:
            # Run the script
            asyncio.create_task(self._run_volume_surge_scanner())
            self.volume_surge_runs_completed += 1

            # Schedule the next run (next hourly :45 time slot)
            # Run times: 06:45, 07:45, 08:45, 09:45, 10:45, 11:45, 12:45 ET
            run_times = [
                6 * 60 + 45,   # 06:45 (405)
                7 * 60 + 45,   # 07:45 (465)
                8 * 60 + 45,   # 08:45 (525)
                9 * 60 + 45,   # 09:45 (585)
                10 * 60 + 45,  # 10:45 (645)
                11 * 60 + 45,  # 11:45 (705)
                12 * 60 + 45   # 12:45 (765)
            ]

            # Find the next run time
            current_minutes_since_midnight = current_time.hour * 60 + current_time.minute
            next_run = None

            # Look for next run time today
            for run_minute in run_times:
                if current_minutes_since_midnight < run_minute:
                    next_run_hour = run_minute // 60
                    next_run_minute = run_minute % 60
                    next_run = current_time.replace(
                        hour=next_run_hour, minute=next_run_minute, second=0, microsecond=0
                    )
                    break

            # If no more runs today, schedule for next weekday at 06:45
            if next_run is None:
                tomorrow = current_time + timedelta(days=1)

                # Find next weekday
                while tomorrow.weekday() >= 5:  # Skip weekends
                    tomorrow += timedelta(days=1)

                next_run = tomorrow.replace(hour=6, minute=45, second=0, microsecond=0)

            # Check if next_run is a weekday, if not, find next Monday
            if next_run.weekday() >= 5:
                days_until_monday = (7 - next_run.weekday()) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                next_run = (next_run + timedelta(days=days_until_monday)).replace(
                    hour=6, minute=45, second=0, microsecond=0
                )

            self.volume_surge_schedule = [next_run]
            self.logger.info(f"📈 Next volume surge scanner run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S ET')}")

    async def _check_startup_processes(self):
        """Check the status of running startup processes."""
        completed_processes = []

        for process_id, process_info in self.startup_processes.items():
            process = process_info['process']

            if process.poll() is not None:  # Process has completed
                return_code = process.returncode
                runtime = datetime.now(self.et_tz) - process_info['start_time']

                if return_code == 0:
                    if process_id == 'volume_surge':
                        self.logger.info(f"✅ Volume surge scanner completed successfully (Runtime: {runtime})")
                        self.volume_surge_completed = True
                    else:
                        self.logger.info(f"✅ Startup script completed successfully (Runtime: {runtime})")
                        # Send the generated top gainers file to Bruce
                        await self._send_top_gainers_file_to_bruce()
                else:
                    if process_id == 'volume_surge':
                        self.logger.error(f"❌ Volume surge scanner failed with return code {return_code} (Runtime: {runtime})")
                    else:
                        self.logger.error(f"❌ Startup script failed with return code {return_code} (Runtime: {runtime})")

                # Log any output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stdout:
                        self.logger.debug(f"Startup script output: {stdout[-500:]}")  # Last 500 chars
                    if stderr:
                        self.logger.debug(f"Startup script stderr: {stderr[-500:]}")  # Last 500 chars
                except Exception:
                    pass

                completed_processes.append(process_id)

        # Remove completed processes
        for process_id in completed_processes:
            del self.startup_processes[process_id]

    async def _check_premarket_processes(self):
        """Check the status of running premarket processes."""
        completed_processes = []

        for process_id, process_info in self.premarket_processes.items():
            process = process_info['process']

            if process.poll() is not None:  # Process has completed
                return_code = process.returncode
                runtime = datetime.now(self.et_tz) - process_info['start_time']

                if return_code == 0:
                    self.logger.info(f"✅ Premarket script completed successfully (Runtime: {runtime})")
                    # Send the generated premarket file to Bruce
                    await self._send_premarket_file_to_bruce()
                else:
                    self.logger.error(f"❌ Premarket script failed with return code {return_code} (Runtime: {runtime})")

                # Log any output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stdout:
                        self.logger.debug(f"Premarket script output: {stdout[-500:]}")  # Last 500 chars
                    if stderr:
                        self.logger.debug(f"Premarket script stderr: {stderr[-500:]}")  # Last 500 chars
                except Exception:
                    pass

                completed_processes.append(process_id)

        # Remove completed processes
        for process_id in completed_processes:
            del self.premarket_processes[process_id]

    async def _send_top_gainers_file_to_bruce(self):
        """
        Send the generated top gainers file contents to Bruce via Telegram.
        """
        try:
            # Check if the CSV file exists
            if not self.csv_file_path.exists():
                self.logger.warning(f"⚠️ Top gainers file not found: {self.csv_file_path}")
                return

            # Read the CSV file contents
            try:
                with open(self.csv_file_path, 'r') as f:
                    file_contents = f.read().strip()

                if not file_contents:
                    self.logger.warning(f"⚠️ Top gainers file is empty: {self.csv_file_path}")
                    return

                # Get file metadata
                file_time = datetime.fromtimestamp(self.csv_file_path.stat().st_mtime, self.et_tz)

                # Count rows (excluding header)
                lines = file_contents.split('\n')
                total_rows = len(lines) - 1 if len(lines) > 1 else 0

                # Create message with file contents
                message_parts = [
                    "📊 **TOP GAINERS FILE GENERATED**",
                    "",
                    "📁 **File:** `gainers_nasdaq_amex.csv`",
                    f"⏰ **Generated:** {file_time.strftime('%H:%M:%S ET')}",
                    f"📅 **Date:** {file_time.strftime('%Y-%m-%d')}",
                    f"📈 **Total Symbols:** {total_rows}",
                    "",
                    "📋 **File Contents:**",
                    "```csv",
                    file_contents,
                    "```",
                    "",
                    f"📂 **Path:** `{self.csv_file_path}`"
                ]

                message = "\n".join(message_parts)

                if self.test_mode:
                    self.logger.info(f"[TEST MODE] Top gainers file contents: {len(file_contents)} characters")
                else:
                    # Send to Bruce
                    result = self.telegram_poster.send_message_to_user(message, "bruce", urgent=False)

                    if result['success']:
                        self.logger.info(f"✅ Top gainers file contents sent to Bruce ({len(file_contents)} characters)")
                    else:
                        errors = result.get('errors', ['Unknown error'])
                        error_msg = ', '.join(errors) if isinstance(errors, list) else str(errors)
                        self.logger.error(f"❌ Failed to send top gainers file contents to Bruce: {error_msg}")

            except Exception as file_error:
                self.logger.error(f"❌ Error reading top gainers file: {file_error}")

                # Send basic notification even if file reading fails
                basic_message = (
                    "📊 **TOP GAINERS FILE GENERATED**\n\n"
                    "📁 **File:** `gainers_nasdaq_amex.csv`\n"
                    f"⏰ **Time:** {datetime.now(self.et_tz).strftime('%H:%M:%S ET')}\n"
                    f"📂 **Path:** `{self.csv_file_path}`\n\n"
                    "⚠️ **Note:** Could not read file contents for sending"
                )

                if not self.test_mode:
                    self.telegram_poster.send_message_to_user(basic_message, "bruce", urgent=False)

        except Exception as e:
            self.logger.error(f"❌ Error sending top gainers file contents: {e}")

    async def _send_premarket_file_to_bruce(self):
        """
        Send the generated premarket file contents to Bruce via Telegram.
        """
        try:
            # Check if the CSV file exists
            if not self.premarket_csv_path.exists():
                self.logger.warning(f"⚠️ Premarket file not found: {self.premarket_csv_path}")
                return

            # Read the CSV file contents
            try:
                with open(self.premarket_csv_path, 'r') as f:
                    file_contents = f.read().strip()

                if not file_contents:
                    self.logger.warning(f"⚠️ Premarket file is empty: {self.premarket_csv_path}")
                    return

                # Get file metadata
                file_time = datetime.fromtimestamp(self.premarket_csv_path.stat().st_mtime, self.et_tz)

                # Count rows (excluding header)
                lines = file_contents.split('\n')
                total_rows = len(lines) - 1 if len(lines) > 1 else 0

                # Create message with file contents
                message_parts = [
                    "🌅 **PREMARKET TOP GAINERS FILE GENERATED**",
                    "",
                    "📁 **File:** `top_gainers_nasdaq_amex.csv`",
                    f"⏰ **Generated:** {file_time.strftime('%H:%M:%S ET')}",
                    f"📅 **Date:** {file_time.strftime('%Y-%m-%d')}",
                    f"📈 **Total Symbols:** {total_rows}",
                    "",
                    "📋 **File Contents:**",
                    "```csv",
                    file_contents,
                    "```",
                    "",
                    f"📂 **Path:** `{self.premarket_csv_path}`"
                ]

                message = "\n".join(message_parts)

                if self.test_mode:
                    self.logger.info(f"[TEST MODE] Premarket file contents: {len(file_contents)} characters")
                else:
                    # Send to Bruce
                    result = self.telegram_poster.send_message_to_user(message, "bruce", urgent=False)

                    if result['success']:
                        self.logger.info(f"✅ Premarket file contents sent to Bruce ({len(file_contents)} characters)")
                    else:
                        errors = result.get('errors', ['Unknown error'])
                        error_msg = ', '.join(errors) if isinstance(errors, list) else str(errors)
                        self.logger.error(f"❌ Failed to send premarket file contents to Bruce: {error_msg}")

            except Exception as file_error:
                self.logger.error(f"❌ Error reading premarket file: {file_error}")

                # Send basic notification even if file reading fails
                basic_message = (
                    "🌅 **PREMARKET TOP GAINERS FILE GENERATED**\n\n"
                    "📁 **File:** `top_gainers_nasdaq_amex.csv`\n"
                    f"⏰ **Time:** {datetime.now(self.et_tz).strftime('%H:%M:%S ET')}\n"
                    f"📂 **Path:** `{self.premarket_csv_path}`\n\n"
                    "⚠️ **Note:** Could not read file contents for sending"
                )

                if not self.test_mode:
                    self.telegram_poster.send_message_to_user(basic_message, "bruce", urgent=False)

        except Exception as e:
            self.logger.error(f"❌ Error sending premarket file contents: {e}")

    async def _run_volume_surge_scanner(self) -> bool:
        """
        Run the volume surge scanner once at startup.

        Returns:
            True if scanner ran successfully, False otherwise
        """
        self.logger.info("📈 Running volume surge scanner at startup")

        script_path = Path("code") / "alpaca_screener.py"
        cmd = [
            "~/miniconda3/envs/alpaca/bin/python",
            str(script_path),
            "--symbols-file", "data_master/master.csv",
            "--min-volume", "250000",
            "--min-percent-change", "5.0",
            "--surge-days", "50",
            "--volume-surge", "5.0",
            "--export-csv", "relative_volume_nasdaq_amex.csv",
            "--verbose"
        ]

        try:
            # Expand the tilde in the Python path
            cmd[0] = os.path.expanduser(cmd[0])

            # Create environment with PYTHONPATH set to project root
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root

            # Create process with logging
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path.cwd()),
                env=env
            )

            # Store process for monitoring
            self.startup_processes['volume_surge'] = {
                'process': process,
                'start_time': datetime.now(self.et_tz),
                'cmd': ' '.join(cmd)
            }

            self.logger.info(f"📈 Started volume surge scanner (PID: {process.pid})")
            self.logger.info("🕒 Expected completion: up to 10 minutes")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start volume surge scanner: {e}")
            return False

    def _run_symbol_volume_screener(self, symbols: List[str]) -> bool:
        """
        Run the alpaca screener for collected symbols to get volume surge data.

        Args:
            symbols: List of symbols to analyze

        Returns:
            True if screener ran successfully, False otherwise
        """
        if not symbols:
            self.logger.debug("⚠️ No symbols provided to volume screener")
            return False

        self.logger.info(f"📊 Running volume screener for {len(symbols)} symbols")

        script_path = Path("code") / "alpaca_screener.py"

        # Build command with symbol list
        cmd = [
            "~/miniconda3/envs/alpaca/bin/python",
            str(script_path),
            "--symbols"
        ]
        cmd.extend(symbols)
        cmd.extend([
            "--surge-days", "50",
            "--volume-surge", "5.0",
            "--export-csv", "symbol_list.csv",
            "--verbose"
        ])

        try:
            # Expand the tilde in the Python path
            cmd[0] = os.path.expanduser(cmd[0])

            # Create environment with PYTHONPATH set to project root
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root

            # Run synchronously with timeout
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(Path.cwd()),
                env=env
            )

            if result.returncode == 0:
                self.logger.info(f"✅ Volume screener completed successfully for {len(symbols)} symbols")
                return True
            else:
                self.logger.warning(f"⚠️ Volume screener completed with return code {result.returncode}")
                if self.verbose:
                    self.logger.debug(f"Output: {result.stdout[-500:]}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ Volume screener timed out after 5 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to run volume screener: {e}")
            return False

    def _load_symbol_volume_data(self) -> Dict[str, Dict]:
        """
        Load volume surge data from symbol_list.csv.

        Returns:
            Dictionary mapping symbols to volume surge data
        """
        volume_data = {}

        if not self.symbol_list_csv_path.exists():
            self.logger.debug(f"⚠️ Symbol list CSV not found: {self.symbol_list_csv_path}")
            return volume_data

        try:
            with open(self.symbol_list_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('symbol', '').strip().upper()
                    if symbol:
                        volume_surge_detected = row.get('volume_surge_detected', 'False').strip()
                        volume_surge_ratio = row.get('volume_surge_ratio', '')

                        # Convert volume_surge_detected to boolean
                        detected = volume_surge_detected.lower() in ('true', '1', 'yes')

                        # Convert volume_surge_ratio to float if available
                        ratio = None
                        if volume_surge_ratio and volume_surge_ratio.strip():
                            try:
                                ratio = float(volume_surge_ratio)
                            except ValueError:
                                pass

                        volume_data[symbol] = {
                            'volume_surge_detected': detected,
                            'volume_surge_ratio': ratio
                        }

            self.logger.info(f"📊 Loaded volume surge data for {len(volume_data)} symbols from {self.symbol_list_csv_path}")

        except Exception as e:
            self.logger.error(f"❌ Error loading symbol volume data: {e}")

        return volume_data

    def _fetch_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch fundamental data for symbols.

        Args:
            symbols: List of symbols to fetch fundamental data for

        Returns:
            Dictionary mapping symbols to fundamental data
        """
        fundamental_data = {}

        if not symbols:
            return fundamental_data

        self.logger.info(f"📊 Fetching fundamental data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                data = self.fundamental_fetcher.get_fundamental_data(symbol)

                # Only store if we got valid data
                if data and data.get('source') != 'none':
                    fundamental_data[symbol] = data

                    if self.verbose:
                        shares_str = f"{data['shares_outstanding']:,}" if data['shares_outstanding'] else "N/A"
                        float_str = f"{data['float_shares']:,}" if data['float_shares'] else "N/A"
                        cap_str = f"${data['market_cap']:,}" if data['market_cap'] else "N/A"
                        self.logger.debug(
                            f"  {symbol}: Shares: {shares_str} | Float: {float_str} | "
                            f"Cap: {cap_str} | Source: {data['source']}")

            except Exception as e:
                self.logger.debug(f"⚠️ Error fetching fundamental data for {symbol}: {e}")
                continue

        if fundamental_data:
            self.logger.info(
                f"✅ Retrieved fundamental data for {len(fundamental_data)}/{len(symbols)} symbols")
        else:
            self.logger.warning(
                "⚠️ No fundamental data retrieved. Ensure yfinance library is installed "
                "(pip install yfinance)")

        return fundamental_data

    def _get_market_open_price(self, symbol: str) -> Optional[float]:
        """
        Get the market open price (9:30 AM ET) for a symbol using Alpaca API.

        Args:
            symbol: Stock symbol

        Returns:
            Market open price if found, None otherwise
        """
        if not self.historical_client:
            return None

        try:
            # Get today's date in ET
            current_et = datetime.now(self.et_tz)
            today = current_et.date()

            # If before market open, use previous trading day
            market_open_time = dt_time(9, 30)  # 9:30 AM
            if current_et.time() < market_open_time:
                today = today - timedelta(days=1)

            # Skip backwards to most recent weekday
            while today.weekday() >= 5:  # Skip weekends
                today = today - timedelta(days=1)

            # Create target time for market open (9:30 AM ET)
            market_open_datetime = self.et_tz.localize(
                datetime.combine(today, market_open_time)
            )

            # Get bars from 9:25 AM to 9:35 AM ET (10 minute window around market open)
            start_time = market_open_datetime - timedelta(minutes=5)
            end_time = market_open_datetime + timedelta(minutes=5)

            # Fetch 1-minute bars
            bars = self.historical_client.get_bars(
                symbol,
                tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Minute),
                start=start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                end=end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                limit=20,
                feed='sip'
            )

            if not bars or len(bars) == 0:
                self.logger.debug(f"⚠️ No bars found for {symbol} around market open")
                return None

            # Find the bar closest to 9:30 AM ET
            closest_bar = None
            min_time_diff = None

            for bar in bars:
                bar_time = bar.t
                if bar_time.tzinfo is None:
                    bar_time = pytz.UTC.localize(bar_time)
                bar_time_et = bar_time.astimezone(self.et_tz)

                time_diff = abs((bar_time_et - market_open_datetime).total_seconds())

                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_bar = bar

            if closest_bar and min_time_diff <= 300:  # Within 5 minutes
                market_open_price = float(closest_bar.o)
                self.logger.debug(
                    f"✅ {symbol}: Retrieved market open price ${market_open_price:.2f} "
                    f"from Alpaca API")
                return market_open_price
            else:
                self.logger.debug(
                    f"⚠️ {symbol}: No bar found within 5 minutes of market open")
                return None

        except Exception as e:
            self.logger.debug(f"⚠️ Error fetching market open price for {symbol}: {e}")
            return None

    def _load_csv_symbols(self) -> Dict[str, Dict]:
        """
        Load symbols from the gainers CSV file, volume surge CSV file, and additional data/{YYYYMMDD}.csv file.

        Limits to first 40 symbols from each source that don't end in 'W'.
        Tracks source with boolean fields: from_gainers, from_volume_surge, oracle.

        Returns:
            Dictionary mapping symbols to their metadata (including market_open_price)
        """
        symbols_dict = {}  # Use dict to store symbol metadata

        # Load from gainers CSV file - keep symbols with gain >= 20% that don't end in 'W'
        if self.csv_file_path.exists():
            try:
                gainers_count = 0
                with open(self.csv_file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbol = row.get('symbol', '').strip().upper()
                        gain_percent = row.get('gain_percent', None)

                        # Filter: must have symbol, not end in 'W', and gain >= 20%
                        if symbol and not symbol.endswith('W'):
                            try:
                                gain = float(gain_percent) if gain_percent else 0
                                if gain >= 20.0:  # Only keep stocks with gain >= 20%
                                    # Store symbol with market open price from CSV
                                    market_open_price = row.get('market_open_price', None)
                                    symbols_dict[symbol] = {
                                        'source': 'gainers_csv',
                                        'market_open_price': float(market_open_price) if market_open_price else None,
                                        'from_gainers': True,
                                        'from_volume_surge': False,
                                        'oracle': False
                                    }
                                    gainers_count += 1
                            except ValueError:
                                # Skip if gain_percent cannot be converted to float
                                continue

                self.logger.info(f"📊 Loaded {gainers_count} symbols from gainers CSV (gain >= 20%, non-W symbols)")

            except Exception as e:
                self.logger.error(f"❌ Error loading gainers CSV file: {e}")

        # Load from volume surge CSV file - keep first 40 symbols that don't end in 'W'
        if self.volume_surge_csv_path.exists():
            try:
                volume_surge_count = 0
                volume_surge_updated_count = 0
                with open(self.volume_surge_csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbol = row.get('symbol', '').strip().upper()
                        # Filter: must have symbol and not end in 'W'
                        if symbol and not symbol.endswith('W'):
                            if symbol in symbols_dict:
                                # Symbol already exists - update from_volume_surge flag
                                symbols_dict[symbol]['from_volume_surge'] = True
                                volume_surge_updated_count += 1
                            elif volume_surge_count < 40:  # Only add first 40 new symbols
                                # Store symbol without market open price (volume surge CSV doesn't have it)
                                symbols_dict[symbol] = {
                                    'source': 'volume_surge_csv',
                                    'market_open_price': None,
                                    'from_gainers': False,
                                    'from_volume_surge': True,
                                    'oracle': False
                                }
                                volume_surge_count += 1
                            else:
                                continue  # Skip additional symbols beyond first 40 new ones

                self.logger.info(
                    f"📈 Added {volume_surge_count} unique symbols from "
                    f"volume surge CSV (first 40 non-W symbols), updated "
                    f"{volume_surge_updated_count} existing: "
                    f"{self.volume_surge_csv_path}")

            except Exception as e:
                self.logger.error(f"❌ Error loading volume surge CSV file {self.volume_surge_csv_path}: {e}")
        else:
            self.logger.debug(f"📈 Volume surge CSV file not found: {self.volume_surge_csv_path}")

        # Load from additional data/{YYYYMMDD}.csv file - all symbols (Oracle source)
        compact_date = datetime.now(self.et_tz).strftime('%Y%m%d')  # YYYYMMDD format
        data_csv_path = Path("data") / f"{compact_date}.csv"

        if data_csv_path.exists():
            try:
                additional_count = 0
                oracle_updated_count = 0
                with open(data_csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbol = row.get('symbol', '').strip().upper()
                        # Filter: must have symbol and not end in 'W'
                        if symbol and not symbol.endswith('W'):
                            if symbol in symbols_dict:
                                # Symbol already exists - update oracle flag
                                symbols_dict[symbol]['oracle'] = True
                                oracle_updated_count += 1
                            else:
                                # Store symbol without market open price (data CSV may not have it)
                                symbols_dict[symbol] = {
                                    'source': 'data_csv',
                                    'market_open_price': None,
                                    'from_gainers': False,
                                    'from_volume_surge': False,
                                    'oracle': True
                                }
                                additional_count += 1

                self.logger.info(
                    f"📊 Added {additional_count} unique symbols from "
                    f"data CSV (Oracle source), updated "
                    f"{oracle_updated_count} existing: {data_csv_path}")

            except Exception as e:
                self.logger.error(f"❌ Error loading data CSV file {data_csv_path}: {e}")
        else:
            self.logger.debug(f"📄 Data CSV file not found: {data_csv_path}")

        # Load from premarket CSV file - keep first 40 symbols that don't end in 'W'
        if self.premarket_csv_path.exists():
            try:
                premarket_count = 0
                premarket_updated_count = 0
                with open(self.premarket_csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbol = row.get('symbol', '').strip().upper()
                        # Filter: must have symbol and not end in 'W'
                        if symbol and not symbol.endswith('W'):
                            if symbol in symbols_dict:
                                # Symbol already exists - mark as from_premarket
                                symbols_dict[symbol]['from_premarket'] = True
                                premarket_updated_count += 1
                            elif premarket_count < 40:  # Only add first 40 new symbols
                                # Store symbol with premarket price from CSV if available
                                premarket_price = row.get('premarket_price', None)
                                symbols_dict[symbol] = {
                                    'source': 'premarket_csv',
                                    'market_open_price': float(premarket_price) if premarket_price else None,
                                    'from_gainers': False,
                                    'from_volume_surge': False,
                                    'oracle': False,
                                    'from_premarket': True
                                }
                                premarket_count += 1
                            else:
                                continue  # Skip additional symbols beyond first 40 new ones

                self.logger.info(
                    f"🌅 Added {premarket_count} unique symbols from "
                    f"premarket CSV (first 40 non-W symbols), updated "
                    f"{premarket_updated_count} existing: "
                    f"{self.premarket_csv_path}")

            except Exception as e:
                self.logger.error(f"❌ Error loading premarket CSV file {self.premarket_csv_path}: {e}")
        else:
            self.logger.debug(f"🌅 Premarket CSV file not found: {self.premarket_csv_path}")

        # Initialize from_premarket field for symbols that don't have it
        for symbol in symbols_dict:
            if 'from_premarket' not in symbols_dict[symbol]:
                symbols_dict[symbol]['from_premarket'] = False

        # Fetch market open prices for symbols that don't have them
        if symbols_dict and self.historical_client:
            symbols_without_price = [
                symbol for symbol, metadata in symbols_dict.items()
                if metadata.get('market_open_price') is None
            ]

            if symbols_without_price:
                self.logger.info(
                    f"📊 Fetching market open prices for {len(symbols_without_price)} symbols "
                    f"from Alpaca API...")

                fetch_count = 0
                for symbol in symbols_without_price:
                    market_open_price = self._get_market_open_price(symbol)
                    if market_open_price is not None:
                        symbols_dict[symbol]['market_open_price'] = market_open_price
                        fetch_count += 1

                self.logger.info(
                    f"✅ Successfully fetched market open prices for {fetch_count}/{len(symbols_without_price)} symbols")

        # Run volume screener for collected symbols to get volume surge data
        if symbols_dict:
            symbols_list = sorted(list(symbols_dict.keys()))
            self.logger.info(f"📊 Running volume screener for {len(symbols_list)} collected symbols...")

            # Call alpaca_screener.py with the collected symbols
            screener_success = self._run_symbol_volume_screener(symbols_list)

            if screener_success:
                # Load volume surge data from the generated CSV
                volume_data = self._load_symbol_volume_data()

                # Merge volume data into symbol metadata
                if volume_data:
                    for symbol, vol_data in volume_data.items():
                        if symbol in symbols_dict:
                            symbols_dict[symbol]['volume_surge_detected'] = vol_data.get('volume_surge_detected', False)
                            symbols_dict[symbol]['volume_surge_ratio'] = vol_data.get('volume_surge_ratio', None)
                        else:
                            self.logger.debug(f"⚠️ Volume data found for {symbol} but symbol not in monitored list")

                    # Log statistics
                    surge_count = sum(
                        1 for metadata in symbols_dict.values()
                        if metadata.get('volume_surge_detected', False)
                    )
                    self.logger.info(
                        f"✅ Volume surge data integrated: {surge_count}/{len(symbols_dict)} symbols have surge detected")
                else:
                    self.logger.warning("⚠️ No volume surge data loaded from screener CSV")
            else:
                self.logger.warning("⚠️ Volume screener failed, proceeding without volume surge data")

            # Fetch fundamental data for collected symbols
            self.logger.info(f"📊 Fetching fundamental data for {len(symbols_list)} symbols...")
            fundamental_data = self._fetch_fundamental_data(symbols_list)

            # Merge fundamental data into symbol metadata
            if fundamental_data:
                for symbol, fund_data in fundamental_data.items():
                    if symbol in symbols_dict:
                        symbols_dict[symbol]['shares_outstanding'] = fund_data.get('shares_outstanding')
                        symbols_dict[symbol]['float_shares'] = fund_data.get('float_shares')
                        symbols_dict[symbol]['market_cap'] = fund_data.get('market_cap')
                        symbols_dict[symbol]['fundamental_source'] = fund_data.get('source')

                # Log statistics
                fund_count = sum(
                    1 for metadata in symbols_dict.values()
                    if metadata.get('shares_outstanding') is not None
                )
                self.logger.info(
                    f"✅ Fundamental data integrated: {fund_count}/{len(symbols_dict)} symbols have fundamental data")
            else:
                # Set None values for all symbols
                for symbol in symbols_dict:
                    symbols_dict[symbol]['shares_outstanding'] = None
                    symbols_dict[symbol]['float_shares'] = None
                    symbols_dict[symbol]['market_cap'] = None
                    symbols_dict[symbol]['fundamental_source'] = 'none'

        # Return dictionary with symbol metadata
        if symbols_dict:
            symbols_list = sorted(list(symbols_dict.keys()))
            symbols_with_price = sum(
                1 for metadata in symbols_dict.values()
                if metadata.get('market_open_price') is not None
            )
            self.logger.info(
                f"📊 Total unique symbols to monitor: {len(symbols_dict)} "
                f"({symbols_with_price} with market open price) - "
                f"{symbols_list[:10]}{'...' if len(symbols_list) > 10 else ''}")
        else:
            self.logger.warning("⚠️ No symbols found in any CSV files")

        return symbols_dict

    def _load_manual_symbols(self) -> Dict[str, Dict]:
        """
        Load manually added symbols from JSON file.

        Returns:
            Dictionary mapping symbols to their metadata
        """
        manual_symbols_dict = {}

        if not self.manual_symbols_file.exists():
            self.logger.debug(f"📝 Manual symbols file not found: {self.manual_symbols_file}")
            return manual_symbols_dict

        try:
            with open(self.manual_symbols_file, 'r') as f:
                data = json.load(f)

            symbols = data.get('symbols', [])

            if not symbols:
                self.logger.debug("📝 No manual symbols found in file")
                return manual_symbols_dict

            # Add each manual symbol with appropriate metadata
            for symbol in symbols:
                symbol = symbol.strip().upper()
                if symbol and not symbol.endswith('W'):
                    manual_symbols_dict[symbol] = {
                        'source': 'manual',
                        'market_open_price': None,  # Will be fetched if needed
                        'from_gainers': False,
                        'from_volume_surge': False,
                        'oracle': False,
                        'manual': True
                    }

            self.logger.info(f"📝 Loaded {len(manual_symbols_dict)} manual symbols for monitoring")

            # Update last check timestamp
            self.last_manual_symbols_check = datetime.now(self.et_tz)

        except Exception as e:
            self.logger.error(f"❌ Error loading manual symbols file {self.manual_symbols_file}: {e}")

        return manual_symbols_dict

    def _merge_manual_symbols(self, symbols_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Merge manual symbols into the existing symbols dictionary.

        Args:
            symbols_dict: Existing dictionary of symbols

        Returns:
            Updated dictionary with manual symbols merged in
        """
        manual_symbols = self._load_manual_symbols()

        if not manual_symbols:
            return symbols_dict

        # Merge manual symbols into existing dict
        for symbol, metadata in manual_symbols.items():
            if symbol in symbols_dict:
                # Symbol already exists - just add the manual flag
                symbols_dict[symbol]['manual'] = True
                self.logger.debug(f"📝 Symbol {symbol} already monitored, marking as manual")
            else:
                # Add new manual symbol
                symbols_dict[symbol] = metadata
                self.logger.debug(f"📝 Added manual symbol {symbol} to monitoring")

        return symbols_dict

    def get_current_symbol_list(self) -> List[Dict]:
        """
        Get the current monitored symbol list with metadata for web interface.

        Returns:
            List of dictionaries containing symbol information with fields:
            - symbol: Stock symbol
            - oracle: Boolean - from Oracle data source
            - manual: Boolean - manually added
            - top_gainers: Boolean - from top gainers list
            - surge: Boolean - from volume surge list
        """
        symbol_list = []

        for symbol, metadata in self.monitored_symbols.items():
            symbol_info = {
                'symbol': symbol,
                'oracle': metadata.get('oracle', False),
                'manual': metadata.get('manual', False),
                'top_gainers': metadata.get('from_gainers', False),
                'surge': metadata.get('from_volume_surge', False),
                'premarket': metadata.get('from_premarket', False)
            }
            symbol_list.append(symbol_info)

        # Sort by symbol name
        symbol_list.sort(key=lambda x: x['symbol'])

        return symbol_list

    def _monitor_csv_file(self):
        """Monitor the CSV file for creation/updates."""
        # Check if we should reload symbols
        should_reload = False

        if self.csv_file_path.exists():
            file_mtime = datetime.fromtimestamp(self.csv_file_path.stat().st_mtime, self.et_tz)

            if self.last_csv_check is None or file_mtime > self.last_csv_check:
                should_reload = True
                self.last_csv_check = file_mtime
                self.logger.info(f"📁 CSV file detected/updated: {file_mtime.strftime('%H:%M:%S ET')}")

        if should_reload:
            new_symbols_dict = self._load_csv_symbols()

            # Merge manual symbols
            new_symbols_dict = self._merge_manual_symbols(new_symbols_dict)

            new_symbols = set(new_symbols_dict.keys())
            current_symbols = set(self.monitored_symbols.keys())

            added_symbols = new_symbols - current_symbols
            removed_symbols = current_symbols - new_symbols

            if added_symbols:
                self.logger.info(f"➕ Added symbols: {sorted(added_symbols)}")
            if removed_symbols:
                self.logger.info(f"➖ Removed symbols: {sorted(removed_symbols)}")

            self.monitored_symbols = new_symbols_dict

    def _monitor_manual_symbols_file(self):
        """Monitor the manual symbols JSON file for updates."""
        # Check if we should reload manual symbols
        should_reload = False

        if self.manual_symbols_file.exists():
            file_mtime = datetime.fromtimestamp(self.manual_symbols_file.stat().st_mtime, self.et_tz)

            if self.last_manual_symbols_check is None or file_mtime > self.last_manual_symbols_check:
                should_reload = True
                self.logger.info(f"📝 Manual symbols file detected/updated: {file_mtime.strftime('%H:%M:%S ET')}")

        if should_reload:
            # Reload all symbols (CSV + manual)
            csv_symbols = self._load_csv_symbols()
            new_symbols_dict = self._merge_manual_symbols(csv_symbols)

            new_symbols = set(new_symbols_dict.keys())
            current_symbols = set(self.monitored_symbols.keys())

            added_symbols = new_symbols - current_symbols
            removed_symbols = current_symbols - new_symbols

            if added_symbols:
                self.logger.info(f"📝 ➕ Added manual symbols: {sorted(added_symbols)}")
            if removed_symbols:
                self.logger.info(f"📝 ➖ Removed manual symbols: {sorted(removed_symbols)}")

            self.monitored_symbols = new_symbols_dict

    async def _collect_stock_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Collect 30 minutes of 1-minute candlesticks for symbols.

        Args:
            symbols: List of symbols to collect data for

        Returns:
            Dictionary mapping symbols to their DataFrame data
        """
        if not symbols or not self.historical_client:
            return {}

        # Calculate time range (last 30 minutes)
        end_time = datetime.now(self.et_tz)
        start_time = end_time - timedelta(minutes=30)

        try:
            data_dict = {}

            # Collect data for each symbol individually (legacy API pattern)
            for symbol in symbols:
                try:
                    # Use legacy alpaca_trade_api pattern
                    bars = self.historical_client.get_bars(
                        symbol,
                        tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Minute),  # 1-minute bars
                        start=start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        end=end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        limit=1000,
                        feed='sip'  # Use SIP feed
                    )

                    if bars and len(bars) > 0:
                        # Convert to DataFrame format
                        bar_data = []
                        for bar in bars:
                            bar_dict = {
                                'o': float(bar.o),    # Use single letter attributes
                                'h': float(bar.h),
                                'l': float(bar.l),
                                'c': float(bar.c),
                                'v': int(bar.v),
                                'timestamp': bar.t,
                                'vwap': getattr(bar, 'vw', None)  # Use VWAP from stock data only
                            }
                            bar_data.append(bar_dict)

                        if bar_data:
                            df = pd.DataFrame(bar_data)
                            df.set_index('timestamp', inplace=True)

                            # Ensure timezone-aware index and convert to ET
                            if df.index.tz is None:
                                df.index = df.index.tz_localize('UTC')
                            df.index = df.index.tz_convert(self.et_tz)

                            # Check if VWAP data is available from stock data
                            if 'vwap' not in df.columns or df['vwap'].isna().all():
                                self.logger.debug(f"⚠️ {symbol}: VWAP not available in stock data, skipping")
                                continue

                            # CRITICAL: Verify the latest bar has valid VWAP from stock data
                            latest_vwap = df.iloc[-1]['vwap']
                            if pd.isna(latest_vwap) or latest_vwap is None:
                                self.logger.debug(f"⚠️ {symbol}: Latest bar missing VWAP from stock data, skipping")
                                continue

                            self.logger.debug(f"✅ {symbol}: Using VWAP from stock data (vw attribute): ${latest_vwap:.2f}")

                            data_dict[symbol] = df

                            self.logger.debug(f"📊 Collected {len(df)} bars for {symbol}")

                except Exception as symbol_error:
                    self.logger.debug(f"⚠️ Error collecting data for {symbol}: {symbol_error}")
                    continue

            if data_dict:
                self.logger.debug(f"📊 Collected data for {len(data_dict)} symbols")

            return data_dict

        except Exception as e:
            self.logger.error(f"❌ Error collecting stock data: {e}")
            return {}

    async def _collect_hourly_volume_data(self, symbols: List[str]) -> Dict[str, int]:
        """
        Collect 1-hour candlesticks from 04:00 ET to now and sum volume for float rotation.

        Args:
            symbols: List of symbols to collect hourly volume for

        Returns:
            Dictionary mapping symbols to their total volume since 04:00 ET
        """
        if not symbols or not self.historical_client:
            return {}

        try:
            volume_dict = {}

            # Calculate time range (04:00 ET today to now)
            current_et = datetime.now(self.et_tz)

            # Start at 04:00 ET today
            start_time = current_et.replace(hour=4, minute=0, second=0, microsecond=0)

            # If current time is before 04:00 ET, use yesterday's 04:00 ET
            if current_et.hour < 4:
                start_time = start_time - timedelta(days=1)

            end_time = current_et

            self.logger.debug(f"📊 Fetching hourly volume from {start_time.strftime('%H:%M ET')} to {end_time.strftime('%H:%M ET')}")

            # Collect hourly data for each symbol
            for symbol in symbols:
                try:
                    # Fetch 1-hour bars
                    bars = self.historical_client.get_bars(
                        symbol,
                        tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Hour),  # 1-hour bars
                        start=start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        end=end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        limit=100,  # Up to 100 hours
                        feed='sip'
                    )

                    if bars and len(bars) > 0:
                        # Sum volume from all hourly bars
                        total_volume = 0
                        bar_count = 0

                        for bar in bars:
                            total_volume += int(bar.v)
                            bar_count += 1

                        volume_dict[symbol] = total_volume

                        self.logger.debug(
                            f"📊 {symbol}: Summed {bar_count} hourly bars "
                            f"(04:00 ET to now) = {total_volume:,} volume")

                except Exception as symbol_error:
                    self.logger.debug(f"⚠️ Error collecting hourly volume for {symbol}: {symbol_error}")
                    continue

            if volume_dict:
                self.logger.debug(f"📊 Collected hourly volume for {len(volume_dict)} symbols")

            return volume_dict

        except Exception as e:
            self.logger.error(f"❌ Error collecting hourly volume data: {e}")
            return {}

    def _check_momentum_criteria(self, symbol: str, data: pd.DataFrame, symbol_metadata: Optional[Dict] = None, hourly_volume: Optional[int] = None) -> Optional[Dict]:
        """
        Check if a stock meets momentum alert criteria.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            symbol_metadata: Symbol metadata dict containing market_open_price and boolean source fields
            hourly_volume: Total volume from 1-hour bars (04:00 ET to now) for float rotation

        Returns:
            Alert data dictionary if criteria met, None otherwise
        """
        # Extract metadata fields
        if symbol_metadata is None:
            symbol_metadata = {}

        market_open_price = symbol_metadata.get('market_open_price')
        from_gainers = symbol_metadata.get('from_gainers', False)
        from_volume_surge = symbol_metadata.get('from_volume_surge', False)
        oracle = symbol_metadata.get('oracle', False)
        volume_surge_detected = symbol_metadata.get('volume_surge_detected', False)
        volume_surge_ratio = symbol_metadata.get('volume_surge_ratio', None)
        shares_outstanding = symbol_metadata.get('shares_outstanding')
        float_shares = symbol_metadata.get('float_shares')
        market_cap = symbol_metadata.get('market_cap')
        fundamental_source = symbol_metadata.get('fundamental_source', 'none')
        if data.empty or len(data) < 9:  # Need at least 9 bars for EMA9
            return None

        try:
            # Get latest bar
            latest_bar = data.iloc[-1]
            current_price = float(latest_bar['c'])  # Use single letter attribute

            # CRITICAL: Validate VWAP from stock data before using
            if pd.isna(latest_bar['vwap']) or latest_bar['vwap'] is None:
                self.logger.debug(f"❌ {symbol}: VWAP from stock data is invalid")
                return None

            current_vwap = float(latest_bar['vwap'])  # VWAP from stock data (bar.vw attribute)
            current_volume = int(latest_bar['v'])  # Get current volume

            # Calculate float rotation using hourly volume data (04:00 ET to now)
            # Float Rotation = Total Volume (hourly bars from 04:00 ET) / Float Shares
            total_volume_since_0400 = None
            float_rotation = None
            float_rotation_percent = None

            if float_shares and float_shares > 0 and hourly_volume is not None:
                # Use hourly volume sum from 04:00 ET
                total_volume_since_0400 = hourly_volume

                # Calculate float rotation as a ratio
                float_rotation = total_volume_since_0400 / float_shares
                float_rotation_percent = float_rotation * 100

                self.logger.debug(
                    f"📊 {symbol}: Volume since 04:00 ET: {total_volume_since_0400:,} | "
                    f"Float: {float_shares:,} | "
                    f"Float Rotation: {float_rotation:.4f}x ({float_rotation_percent:.2f}%)")

            self.logger.debug(f"📊 {symbol}: Using VWAP from stock data: ${current_vwap:.2f}")

            # Filter out low volume alerts
            if current_volume < self.momentum_config.volume_low_threshold:
                self.logger.debug(f"❌ {symbol}: Volume {current_volume:,} below threshold {self.momentum_config.volume_low_threshold:,}")
                return None

            # Check VWAP criteria
            if current_price < current_vwap:
                self.logger.debug(f"❌ {symbol}: Price ${current_price:.2f} below VWAP ${current_vwap:.2f}")
                return None

            # Calculate technical indicators using breakout detector
            # Convert single-letter columns to full names for the indicator calculation
            data_for_indicators = data.copy()
            if 'o' in data_for_indicators.columns:
                data_for_indicators = data_for_indicators.rename(columns={
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                })

            indicators = self.breakout_detector.calculate_technical_indicators(data_for_indicators)

            ema_9 = indicators.get('ema_9')
            if ema_9 is None:
                self.logger.debug(f"❌ {symbol}: EMA9 calculation failed")
                return None

            # Check EMA9 criteria
            if current_price < ema_9:
                self.logger.debug(f"❌ {symbol}: Price ${current_price:.2f} below EMA9 ${ema_9:.2f}")
                return None

            # Calculate time-normalized momentum values (momentum per minute)
            momentum = 0
            momentum_short = 0

            # Initialize raw momentum values for tracking
            raw_momentum_20 = 0
            raw_momentum_5 = 0

            # Calculate time-based momentum (configurable period in minutes)
            momentum_period_minutes = self.momentum_config.momentum_period
            if len(data) >= 2:  # Need at least 2 data points
                # Get current timestamp (latest data point)
                current_timestamp = data.index[-1]

                # Calculate target timestamp (N minutes ago)
                target_timestamp = current_timestamp - timedelta(
                    minutes=momentum_period_minutes)

                # Find the data point closest to target timestamp
                # Use absolute difference to find closest match
                time_diffs = abs(data.index - target_timestamp)
                closest_idx = time_diffs.argmin()

                # Get price from closest timestamp to N minutes ago
                price_period_ago = float(data.iloc[closest_idx]['c'])
                actual_time_diff = ((current_timestamp - data.index[closest_idx])
                                    .total_seconds() / 60)

                if price_period_ago > 0 and actual_time_diff > 0:
                    raw_momentum_20 = ((current_price - price_period_ago) /
                                       price_period_ago) * 100
                    # Normalize by actual time diff (handles gaps)
                    momentum = raw_momentum_20 / actual_time_diff

            # Calculate time-based short momentum (configurable period)
            momentum_short_period_minutes = (
                self.momentum_config.momentum_short_period)
            if len(data) >= 2:  # Need at least 2 data points
                # Get current timestamp (latest data point)
                current_timestamp = data.index[-1]

                # Calculate target timestamp (N minutes ago)
                target_timestamp_short = current_timestamp - timedelta(
                    minutes=momentum_short_period_minutes)

                # Find the data point closest to target timestamp
                # Use absolute difference to find closest match
                time_diffs_short = abs(data.index - target_timestamp_short)
                closest_idx_short = time_diffs_short.argmin()

                # Get price from closest timestamp to N minutes ago
                price_short_period_ago = float(data.iloc[closest_idx_short]['c'])
                actual_time_diff_short = ((current_timestamp -
                                           data.index[closest_idx_short])
                                          .total_seconds() / 60)

                if price_short_period_ago > 0 and actual_time_diff_short > 0:
                    raw_momentum_5 = ((current_price - price_short_period_ago) /
                                      price_short_period_ago) * 100
                    # Normalize by actual time diff (handles gaps)
                    momentum_short = raw_momentum_5 / actual_time_diff_short

            # Calculate squeeze momentum (configurable period)
            momentum_squeeze = 0
            raw_momentum_squeeze = 0
            actual_time_diff_squeeze = 0
            squeeze_period_minutes = self.momentum_config.squeeze_duration
            if len(data) >= 2:  # Need at least 2 data points
                # Get current timestamp (latest data point)
                current_timestamp = data.index[-1]

                # Calculate target timestamp (N minutes ago)
                target_timestamp_squeeze = current_timestamp - timedelta(
                    minutes=squeeze_period_minutes)

                # Find the data point closest to target timestamp
                # Use absolute difference to find closest match
                time_diffs_squeeze = abs(data.index - target_timestamp_squeeze)
                closest_idx_squeeze = time_diffs_squeeze.argmin()

                # Get price from closest timestamp to N minutes ago
                price_squeeze_period_ago = float(data.iloc[closest_idx_squeeze]['c'])
                actual_time_diff_squeeze = ((current_timestamp -
                                             data.index[closest_idx_squeeze])
                                            .total_seconds() / 60)

                if price_squeeze_period_ago > 0 and actual_time_diff_squeeze > 0:
                    raw_momentum_squeeze = ((current_price - price_squeeze_period_ago) /
                                            price_squeeze_period_ago) * 100
                    # Normalize by actual time diff (handles gaps)
                    momentum_squeeze = raw_momentum_squeeze / actual_time_diff_squeeze

            # Get momentum signal light icons from momentum config
            momentum_emoji = get_momentum_standard_color_emoji(momentum)
            momentum_short_emoji = get_momentum_short_color_emoji(momentum_short)
            squeeze_emoji = get_squeeze_emoji(momentum_squeeze)

            # Check halt status
            is_halted = is_stock_halted(data, symbol, self.logger)
            halt_emoji = get_halt_status_emoji(is_halted)

            # Get volume color emoji
            volume_emoji = get_volume_color_emoji(current_volume)

            # Check urgency level using dual momentum
            urgency = get_urgency_level_dual(momentum, momentum_short)

            if urgency == 'filtered':
                self.logger.debug(f"❌ {symbol}: Filtered by urgency level "
                                  f"(momentum: {momentum:.2f}/min {momentum_emoji}, "
                                  f"momentum_short: {momentum_short:.2f}/min {momentum_short_emoji})")
                return None

            # Calculate percent gain since market open (if market_open_price is available)
            percent_gain_since_market_open = None
            if market_open_price is not None and market_open_price > 0:
                percent_gain_since_market_open = (
                    (current_price - market_open_price) / market_open_price) * 100
                self.logger.debug(
                    f"📊 {symbol}: Market open: ${market_open_price:.2f}, "
                    f"Gain since open: {percent_gain_since_market_open:+.2f}%")

            # If we get here, all criteria are met
            alert_data = {
                'symbol': symbol,
                'current_price': current_price,
                'market_open_price': market_open_price,
                'percent_gain_since_market_open': percent_gain_since_market_open,
                'vwap': current_vwap,
                'ema_9': ema_9,
                'momentum': momentum,
                'momentum_short': momentum_short,
                'momentum_squeeze': momentum_squeeze,
                'raw_momentum_20': raw_momentum_20,
                'raw_momentum_5': raw_momentum_5,
                'raw_momentum_squeeze': raw_momentum_squeeze,
                'actual_time_diff': actual_time_diff if 'actual_time_diff' in locals() else 0,
                'actual_time_diff_short': actual_time_diff_short if 'actual_time_diff_short' in locals() else 0,
                'actual_time_diff_squeeze': actual_time_diff_squeeze,
                'momentum_emoji': momentum_emoji,
                'momentum_short_emoji': momentum_short_emoji,
                'squeeze_emoji': squeeze_emoji,
                'is_halted': is_halted,
                'halt_emoji': halt_emoji,
                'current_volume': current_volume,
                'volume_emoji': volume_emoji,
                'urgency': urgency,
                'timestamp': datetime.now(self.et_tz),
                'indicators': indicators,
                'from_gainers': from_gainers,
                'from_volume_surge': from_volume_surge,
                'oracle': oracle,
                'volume_surge_detected': volume_surge_detected,
                'volume_surge_ratio': volume_surge_ratio,
                'shares_outstanding': shares_outstanding,
                'float_shares': float_shares,
                'market_cap': market_cap,
                'fundamental_source': fundamental_source,
                'total_volume_since_0400': total_volume_since_0400,
                'float_rotation': float_rotation,
                'float_rotation_percent': float_rotation_percent
            }

            self.logger.info(f"✅ {symbol}: Momentum alert criteria met!")
            self.logger.info(f"   Price: ${current_price:.2f} | VWAP: ${current_vwap:.2f} (from stock data) | EMA9: ${ema_9:.2f}")

            # Show actual time periods used (handles halts/gaps)
            time_diff = actual_time_diff if 'actual_time_diff' in locals() else 0
            time_diff_short = actual_time_diff_short if 'actual_time_diff_short' in locals() else 0
            self.logger.info(f"   Momentum: {momentum:.2f}/min {momentum_emoji} | Raw: {raw_momentum_20:.2f}% ({time_diff:.1f}min)")
            self.logger.info(f"   Momentum Short: {momentum_short:.2f}/min {momentum_short_emoji} | Raw: {raw_momentum_5:.2f}% ({time_diff_short:.1f}min)")
            self.logger.info(f"   Squeezing: {momentum_squeeze:.2f}/min {squeeze_emoji} | Raw: {raw_momentum_squeeze:.2f}% ({actual_time_diff_squeeze:.1f}min)")
            self.logger.info(f"   Volume: {current_volume:,} {volume_emoji} | Halt Status: {halt_emoji} | Urgency: {urgency}")

            return alert_data

        except Exception as e:
            self.logger.error(f"❌ Error checking momentum criteria for {symbol}: {e}")
            return None

    async def _send_momentum_alert(self, alert_data: Dict):
        """
        Send momentum alert to all users with momentum_alerts=true via Telegram.

        Args:
            alert_data: Alert information dictionary
        """
        try:
            symbol = alert_data['symbol']
            current_price = alert_data['current_price']
            market_open_price = alert_data.get('market_open_price')
            percent_gain_since_market_open = alert_data.get('percent_gain_since_market_open')
            vwap = alert_data['vwap']
            ema_9 = alert_data['ema_9']
            momentum = alert_data['momentum']
            momentum_short = alert_data['momentum_short']
            momentum_squeeze = alert_data['momentum_squeeze']
            momentum_emoji = alert_data['momentum_emoji']
            momentum_short_emoji = alert_data['momentum_short_emoji']
            squeeze_emoji = alert_data['squeeze_emoji']
            halt_emoji = alert_data['halt_emoji']
            current_volume = alert_data['current_volume']
            volume_emoji = alert_data['volume_emoji']
            urgency = alert_data['urgency']
            timestamp = alert_data['timestamp']
            volume_surge_detected = alert_data.get('volume_surge_detected', False)
            volume_surge_ratio = alert_data.get('volume_surge_ratio', None)
            shares_outstanding = alert_data.get('shares_outstanding')
            float_shares = alert_data.get('float_shares')
            market_cap = alert_data.get('market_cap')
            fundamental_source = alert_data.get('fundamental_source', 'none')
            total_volume_since_0400 = alert_data.get('total_volume_since_0400')
            float_rotation = alert_data.get('float_rotation')
            float_rotation_percent = alert_data.get('float_rotation_percent')

            # Create alert message (VWAP is from stock data, not calculated)
            message_parts = [
                f"🚀 **MOMENTUM ALERT - {symbol}**",
                "",
                f"📅 **Date:** {timestamp.strftime('%Y-%m-%d')}",
                f"⏰ **Time:** {timestamp.strftime('%H:%M:%S ET')}",
                "",
                f"💰 **Price:** ${current_price:.2f}",
            ]

            # Add market open price and gain if available
            if market_open_price is not None and percent_gain_since_market_open is not None:
                message_parts.extend([
                    f"🌅 **Market Open:** ${market_open_price:.2f}",
                    f"📈 **Gain Since Open:** {percent_gain_since_market_open:+.2f}%",
                ])

            # Add the rest of the alert info
            message_parts.extend([
                f"📊 **VWAP (Stock Data):** ${vwap:.2f} ✅",
                f"📈 **EMA9:** ${ema_9:.2f} ✅",
                f"⚡ **Momentum:** {momentum:.2f}%/min {momentum_emoji}",
                f"⚡ **Momentum Short:** {momentum_short:.2f}%/min {momentum_short_emoji}",
                f"🔥 **Squeezing:** {momentum_squeeze:.2f}%/min {squeeze_emoji}",
                f"📈 **Volume:** {current_volume:,} {volume_emoji}",
                f"🚦 **Halt Status:** {halt_emoji}",
                f"🎯 **Urgency:** {urgency.upper()}",
                "",
            ])

            # Add Volume section with surge data
            message_parts.append("**📊 Volume:**")
            surge_detected_text = "✅ Yes" if volume_surge_detected else "❌ No"
            message_parts.append(f"   • **Surge Detected:** {surge_detected_text}")
            if volume_surge_ratio is not None:
                message_parts.append(f"   • **Surge Ratio:** {volume_surge_ratio:.2f}x")
            else:
                message_parts.append(f"   • **Surge Ratio:** N/A")

            # Add float rotation data (calculated from hourly bars since 04:00 ET)
            if float_rotation is not None and total_volume_since_0400 is not None:
                # Determine emoji based on float rotation value
                if float_rotation > 1:
                    float_emoji = "🟢"  # Green if > 1
                elif float_rotation < 0.8:
                    float_emoji = "🔴"  # Red if < 0.8
                else:
                    float_emoji = "🟡"  # Yellow otherwise

                message_parts.append(f"   • **Volume (since 04:00 ET):** {total_volume_since_0400:,}")
                message_parts.append(f"   • **Float Rotation:** {float_rotation:.2f}x {float_emoji}")
            else:
                message_parts.append(f"   • **Float Rotation:** N/A")

            # Add Fundamentals section
            message_parts.extend([
                "",
                "**📈 Fundamentals:**"
            ])

            # Format shares outstanding
            if shares_outstanding is not None:
                if shares_outstanding >= 1_000_000_000:
                    shares_str = f"{shares_outstanding / 1_000_000_000:.2f}B"
                elif shares_outstanding >= 1_000_000:
                    shares_str = f"{shares_outstanding / 1_000_000:.2f}M"
                else:
                    shares_str = f"{shares_outstanding:,.0f}"
                message_parts.append(f"   • **Shares Outstanding:** {shares_str}")
            else:
                message_parts.append(f"   • **Shares Outstanding:** N/A")

            # Format float shares
            if float_shares is not None:
                if float_shares >= 1_000_000_000:
                    float_str = f"{float_shares / 1_000_000_000:.2f}B"
                elif float_shares >= 1_000_000:
                    float_str = f"{float_shares / 1_000_000:.2f}M"
                else:
                    float_str = f"{float_shares:,.0f}"
                message_parts.append(f"   • **Float Shares:** {float_str}")
            else:
                message_parts.append(f"   • **Float Shares:** N/A")

            # Format market cap
            if market_cap is not None:
                if market_cap >= 1_000_000_000:
                    cap_str = f"${market_cap / 1_000_000_000:.2f}B"
                elif market_cap >= 1_000_000:
                    cap_str = f"${market_cap / 1_000_000:.2f}M"
                else:
                    cap_str = f"${market_cap:,.0f}"
                message_parts.append(f"   • **Market Cap:** {cap_str}")
            else:
                message_parts.append(f"   • **Market Cap:** N/A")

            # Add Sources section with green/red light indicators
            from_gainers = alert_data.get('from_gainers', False)
            from_volume_surge = alert_data.get('from_volume_surge', False)
            oracle = alert_data.get('oracle', False)

            message_parts.extend([
                "",
                "**🔍 Sources:**"
            ])

            # Gainers source indicator
            gainers_indicator = "🟢" if from_gainers else "🔴"
            message_parts.append(f"   • **Top Gainers:** {gainers_indicator}")

            # Volume surge source indicator
            volume_indicator = "🟢" if from_volume_surge else "🔴"
            message_parts.append(f"   • **Volume Surge:** {volume_indicator}")

            # Oracle source indicator
            oracle_indicator = "🟢" if oracle else "🔴"
            message_parts.append(f"   • **Oracle:** {oracle_indicator}")

            message = "\n".join(message_parts)

            # Save momentum alert to historical data
            self._save_momentum_alert(alert_data, message)

            if self.test_mode:
                self.logger.info(f"[TEST MODE] {message}")
            else:
                # Get all users with momentum_alerts=true
                momentum_users = self.user_manager.get_momentum_alert_users()

                if not momentum_users:
                    self.logger.warning(
                        f"⚠️ No users with momentum_alerts=true "
                        f"found for {symbol}")
                    return

                # Send to all momentum alert users
                sent_count = 0
                failed_count = 0
                sent_to_users = []

                for user in momentum_users:
                    username = user.get('username', 'Unknown')
                    result = self.telegram_poster.send_message_to_user(
                        message, username, urgent=False)

                    if result['success']:
                        sent_count += 1
                        sent_to_users.append(username)
                        self.logger.info(
                            f"✅ Momentum alert sent to {username} "
                            f"for {symbol}")
                    else:
                        failed_count += 1
                        errors = result.get('errors', ['Unknown error'])
                        error_msg = ', '.join(errors) if isinstance(
                            errors, list) else str(errors)
                        self.logger.error(
                            f"❌ Failed to send momentum alert to "
                            f"{username} for {symbol}: {error_msg}")

                # Log summary
                if sent_count > 0:
                    self.logger.info(
                        f"📨 Momentum alert sent to {sent_count} user(s) "
                        f"for {symbol}: {', '.join(sent_to_users)}")
                    # Save sent alert to historical data with all recipients
                    self._save_momentum_alert_sent(
                        alert_data, message, sent_to_users)

                if failed_count > 0:
                    self.logger.warning(
                        f"⚠️ Failed to send to {failed_count} user(s) "
                        f"for {symbol}")

        except Exception as e:
            self.logger.error(f"❌ Error sending momentum alert: {e}")

    def _serialize_indicators(self, indicators: Dict) -> Dict:
        """
        Serialize indicators dictionary to JSON-compatible format.

        Args:
            indicators: Dictionary containing indicator values

        Returns:
            JSON-serializable dictionary
        """
        serialized = {}
        for k, v in indicators.items():
            if v is None:
                continue
            elif hasattr(v, 'dtype'):  # numpy types
                if 'bool' in str(v.dtype):
                    serialized[k] = bool(v)
                elif 'int' in str(v.dtype) or 'float' in str(v.dtype):
                    serialized[k] = float(v)
                else:
                    serialized[k] = str(v)
            elif isinstance(v, (int, float)):
                serialized[k] = float(v)
            elif isinstance(v, bool):
                serialized[k] = bool(v)
            else:
                serialized[k] = str(v)
        return serialized

    def _save_momentum_alert(self, alert_data: Dict, message: str) -> None:
        """
        Save momentum alert to historical data structure.

        Args:
            alert_data: Alert data dictionary
            message: Formatted alert message
        """
        try:
            symbol = alert_data['symbol']
            timestamp = alert_data['timestamp']

            # Create filename with timestamp
            filename = f"alert_{symbol}_{timestamp.strftime('%Y-%m-%d_%H%M%S')}.json"
            filepath = self.momentum_alerts_dir / filename

            # Convert alert data to serializable format
            market_open = alert_data.get('market_open_price')
            percent_gain = alert_data.get('percent_gain_since_market_open')

            alert_json = {
                'symbol': str(symbol),
                'current_price': float(alert_data['current_price']),
                'market_open_price': float(market_open) if market_open is not None else None,
                'percent_gain_since_market_open': float(percent_gain) if percent_gain is not None else None,
                'vwap': float(alert_data['vwap']),
                'ema_9': float(alert_data['ema_9']),
                'momentum': float(alert_data['momentum']),
                'momentum_short': float(alert_data['momentum_short']),
                'momentum_squeeze': float(alert_data['momentum_squeeze']),
                'raw_momentum_20': float(alert_data['raw_momentum_20']),
                'raw_momentum_5': float(alert_data['raw_momentum_5']),
                'raw_momentum_squeeze': float(alert_data['raw_momentum_squeeze']),
                'actual_time_diff': float(alert_data['actual_time_diff']),
                'actual_time_diff_short': float(alert_data['actual_time_diff_short']),
                'actual_time_diff_squeeze': float(alert_data['actual_time_diff_squeeze']),
                'momentum_emoji': str(alert_data['momentum_emoji']),
                'momentum_short_emoji': str(alert_data['momentum_short_emoji']),
                'squeeze_emoji': str(alert_data['squeeze_emoji']),
                'is_halted': bool(alert_data['is_halted']),
                'halt_emoji': str(alert_data['halt_emoji']),
                'current_volume': int(alert_data['current_volume']),
                'volume_emoji': str(alert_data['volume_emoji']),
                'urgency': str(alert_data['urgency']),
                'timestamp': timestamp.isoformat(),
                'message': str(message),
                'indicators': self._serialize_indicators(alert_data['indicators']),
                'from_gainers': bool(alert_data.get('from_gainers', False)),
                'from_volume_surge': bool(alert_data.get('from_volume_surge', False)),
                'oracle': bool(alert_data.get('oracle', False)),
                'volume_surge_detected': bool(alert_data.get('volume_surge_detected', False)),
                'volume_surge_ratio': float(alert_data['volume_surge_ratio']) if alert_data.get('volume_surge_ratio') is not None else None,
                'shares_outstanding': float(alert_data['shares_outstanding']) if alert_data.get('shares_outstanding') is not None else None,
                'float_shares': float(alert_data['float_shares']) if alert_data.get('float_shares') is not None else None,
                'market_cap': float(alert_data['market_cap']) if alert_data.get('market_cap') is not None else None,
                'fundamental_source': str(alert_data.get('fundamental_source', 'none')),
                'total_volume_since_0400': int(alert_data['total_volume_since_0400']) if alert_data.get('total_volume_since_0400') is not None else None,
                'float_rotation': float(alert_data['float_rotation']) if alert_data.get('float_rotation') is not None else None,
                'float_rotation_percent': float(alert_data['float_rotation_percent']) if alert_data.get('float_rotation_percent') is not None else None
            }

            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(alert_json, f, indent=2)

            self.logger.debug(f"📝 Saved momentum alert for {symbol} to {filename}")

        except Exception as e:
            self.logger.error(f"❌ Error saving momentum alert: {e}")

    def _save_momentum_alert_sent(
            self, alert_data: Dict, message: str,
            sent_to_users: List[str]) -> None:
        """
        Save sent momentum alert to historical data structure.

        Args:
            alert_data: Alert data dictionary
            message: Formatted alert message
            sent_to_users: List of usernames that received the alert
        """
        try:
            symbol = alert_data['symbol']
            timestamp = alert_data['timestamp']

            # Create filename with timestamp
            filename = f"alert_{symbol}_{timestamp.strftime('%Y-%m-%d_%H%M%S')}.json"
            filepath = self.momentum_alerts_sent_dir / filename

            # Convert alert data to serializable format (same as save alert)
            market_open = alert_data.get('market_open_price')
            percent_gain = alert_data.get('percent_gain_since_market_open')

            alert_json = {
                'symbol': str(symbol),
                'current_price': float(alert_data['current_price']),
                'market_open_price': float(market_open) if market_open is not None else None,
                'percent_gain_since_market_open': float(percent_gain) if percent_gain is not None else None,
                'vwap': float(alert_data['vwap']),
                'ema_9': float(alert_data['ema_9']),
                'momentum': float(alert_data['momentum']),
                'momentum_short': float(alert_data['momentum_short']),
                'momentum_squeeze': float(alert_data['momentum_squeeze']),
                'raw_momentum_20': float(alert_data['raw_momentum_20']),
                'raw_momentum_5': float(alert_data['raw_momentum_5']),
                'raw_momentum_squeeze': float(alert_data['raw_momentum_squeeze']),
                'actual_time_diff': float(alert_data['actual_time_diff']),
                'actual_time_diff_short': float(alert_data['actual_time_diff_short']),
                'actual_time_diff_squeeze': float(alert_data['actual_time_diff_squeeze']),
                'momentum_emoji': str(alert_data['momentum_emoji']),
                'momentum_short_emoji': str(alert_data['momentum_short_emoji']),
                'squeeze_emoji': str(alert_data['squeeze_emoji']),
                'is_halted': bool(alert_data['is_halted']),
                'halt_emoji': str(alert_data['halt_emoji']),
                'current_volume': int(alert_data['current_volume']),
                'volume_emoji': str(alert_data['volume_emoji']),
                'urgency': str(alert_data['urgency']),
                'timestamp': timestamp.isoformat(),
                'message': str(message),
                'sent_to': sent_to_users,
                'sent_at': datetime.now(self.et_tz).isoformat(),
                'indicators': self._serialize_indicators(alert_data['indicators']),
                'from_gainers': bool(alert_data.get('from_gainers', False)),
                'from_volume_surge': bool(alert_data.get('from_volume_surge', False)),
                'oracle': bool(alert_data.get('oracle', False)),
                'volume_surge_detected': bool(alert_data.get('volume_surge_detected', False)),
                'volume_surge_ratio': float(alert_data['volume_surge_ratio']) if alert_data.get('volume_surge_ratio') is not None else None,
                'shares_outstanding': float(alert_data['shares_outstanding']) if alert_data.get('shares_outstanding') is not None else None,
                'float_shares': float(alert_data['float_shares']) if alert_data.get('float_shares') is not None else None,
                'market_cap': float(alert_data['market_cap']) if alert_data.get('market_cap') is not None else None,
                'fundamental_source': str(alert_data.get('fundamental_source', 'none')),
                'total_volume_since_0400': int(alert_data['total_volume_since_0400']) if alert_data.get('total_volume_since_0400') is not None else None,
                'float_rotation': float(alert_data['float_rotation']) if alert_data.get('float_rotation') is not None else None,
                'float_rotation_percent': float(alert_data['float_rotation_percent']) if alert_data.get('float_rotation_percent') is not None else None
            }

            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(alert_json, f, indent=2)

            self.logger.debug(f"📝 Saved sent momentum alert for {symbol} to {filename}")

        except Exception as e:
            self.logger.error(f"❌ Error saving sent momentum alert: {e}")

    async def _send_debug_alert(self):
        """
        Send a random debug alert for testing purposes.
        Only active when debug_mode is True.
        """
        import random

        # Pick a random symbol
        symbol = random.choice(self.debug_symbols)

        # Generate random but realistic alert data
        current_price = round(random.uniform(50.0, 500.0), 2)
        market_open_price = round(current_price * random.uniform(0.85, 0.98), 2)
        percent_gain = ((current_price - market_open_price) / market_open_price) * 100

        vwap = round(current_price * random.uniform(0.95, 1.00), 2)
        ema_9 = round(current_price * random.uniform(0.90, 0.98), 2)

        momentum = round(random.uniform(0.5, 3.0), 2)
        momentum_short = round(random.uniform(0.5, 4.0), 2)
        momentum_squeeze = round(random.uniform(0.3, 2.5), 2)

        # Pick random emojis for momentum
        momentum_emojis = ['🟢', '🟡', '🔴']
        momentum_emoji = random.choice(momentum_emojis)
        momentum_short_emoji = random.choice(momentum_emojis)
        squeeze_emoji = random.choice(momentum_emojis)

        volume = random.randint(100000, 5000000)
        volume_emojis = ['🟢', '🟡', '🔴']
        volume_emoji = random.choice(volume_emojis)

        halt_emojis = ['✅', '⏸️']
        halt_emoji = random.choice(halt_emojis)

        urgencies = ['low', 'medium', 'high', 'critical']
        urgency = random.choice(urgencies)

        timestamp = datetime.now(self.et_tz)

        # Create debug alert data
        alert_data = {
            'symbol': symbol,
            'current_price': current_price,
            'market_open_price': market_open_price,
            'percent_gain_since_market_open': percent_gain,
            'vwap': vwap,
            'ema_9': ema_9,
            'momentum': momentum,
            'momentum_short': momentum_short,
            'momentum_squeeze': momentum_squeeze,
            'raw_momentum_20': momentum * 20,
            'raw_momentum_5': momentum_short * 5,
            'raw_momentum_squeeze': momentum_squeeze * 3,
            'actual_time_diff': 20.0,
            'actual_time_diff_short': 5.0,
            'actual_time_diff_squeeze': 3.0,
            'momentum_emoji': momentum_emoji,
            'momentum_short_emoji': momentum_short_emoji,
            'squeeze_emoji': squeeze_emoji,
            'is_halted': halt_emoji == '⏸️',
            'halt_emoji': halt_emoji,
            'current_volume': volume,
            'volume_emoji': volume_emoji,
            'urgency': urgency,
            'timestamp': timestamp,
            'indicators': {},
            'from_gainers': random.choice([True, False]),
            'from_volume_surge': random.choice([True, False]),
            'oracle': random.choice([True, False]),
            'volume_surge_detected': random.choice([True, False]),
            'volume_surge_ratio': round(random.uniform(1.0, 10.0), 2) if random.choice([True, False]) else None,
            'shares_outstanding': random.randint(10000000, 1000000000),
            'float_shares': random.randint(5000000, 500000000),
            'market_cap': random.randint(1000000000, 100000000000),
            'fundamental_source': 'debug',
            'total_volume_since_0400': random.randint(500000, 10000000),
            'float_rotation': round(random.uniform(0.1, 5.0), 2),
            'float_rotation_percent': round(random.uniform(10.0, 500.0), 2)
        }

        self.logger.info(f"🧪 DEBUG: Sending random test alert for {symbol}")
        await self._send_momentum_alert(alert_data)

    async def _stock_monitoring_loop(self):
        """Main stock monitoring loop - runs every minute."""
        while self.running:
            try:
                # Debug mode: Send random alert every minute
                if self.debug_mode:
                    current_time = datetime.now(self.et_tz)

                    # Check if a minute has passed since last debug alert
                    if (self.last_debug_alert_time is None or
                        (current_time - self.last_debug_alert_time).total_seconds() >= 60):

                        self.logger.debug(f"🧪 DEBUG: Triggering test alert (1 minute elapsed)")
                        await self._send_debug_alert()
                        self.last_debug_alert_time = current_time

                # Normal monitoring logic
                if self.monitored_symbols:
                    self.logger.debug(f"🔍 Monitoring {len(self.monitored_symbols)} symbols for momentum alerts")

                    # Collect stock data for all monitored symbols
                    symbols_list = list(self.monitored_symbols.keys())
                    stock_data = await self._collect_stock_data(symbols_list)

                    # Collect hourly volume data (04:00 ET to now) for float rotation
                    hourly_volume_data = await self._collect_hourly_volume_data(symbols_list)

                    # Check each symbol for momentum alerts
                    for symbol in symbols_list:
                        if symbol in stock_data:
                            # Get symbol metadata (includes market_open_price and boolean source fields)
                            symbol_metadata = self.monitored_symbols.get(symbol, {})

                            # Get hourly volume for this symbol (if available)
                            hourly_volume = hourly_volume_data.get(symbol)

                            alert_data = self._check_momentum_criteria(symbol, stock_data[symbol], symbol_metadata, hourly_volume)
                            if alert_data:
                                await self._send_momentum_alert(alert_data)

                # Wait 60 seconds before next check
                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error(f"❌ Error in stock monitoring loop: {e}")
                await asyncio.sleep(60)  # Still wait before retrying

    async def start(self):
        """Start the momentum alerts system."""
        self.logger.info("🚀 Starting Momentum Alerts System...")

        self.running = True

        try:
            # Load existing symbols from CSV files immediately at startup
            self.logger.info("📂 Loading existing symbols from CSV files at startup...")
            if self.csv_file_path.exists() or (self.volume_surge_dir / "relative_volume_nasdaq_amex.csv").exists():
                existing_symbols = self._load_csv_symbols()

                # Merge manual symbols
                existing_symbols = self._merge_manual_symbols(existing_symbols)

                if existing_symbols:
                    self.monitored_symbols = existing_symbols
                    self.logger.info(f"✅ Loaded {len(existing_symbols)} symbols from existing CSV files")
                else:
                    self.logger.info("📋 No existing symbols found, will collect fresh data")
            else:
                self.logger.info("📋 No existing CSV files, will generate fresh data")
                # Still load manual symbols even if no CSV files exist
                manual_symbols = self._load_manual_symbols()
                if manual_symbols:
                    self.monitored_symbols = manual_symbols
                    self.logger.info(f"✅ Loaded {len(manual_symbols)} manual symbols")

            # Run startup script immediately to collect fresh symbols from all sources
            self.logger.info("📊 Running startup script immediately to collect fresh data...")
            await self._run_startup_script()

            # Run premarket script immediately to collect premarket data
            self.logger.info("🌅 Running premarket script immediately to collect fresh data...")
            await self._run_premarket_script()

            # Schedule next startup script runs (will start scheduling from next 20-min interval)
            self._schedule_startup_runs()

            # Schedule premarket script runs (04:00-20:00 ET, every 20 minutes)
            self._schedule_premarket_runs()

            # Schedule volume surge scanner runs (hourly at :45 from 06:45-12:45 ET, weekdays only)
            self._schedule_volume_surge_runs()

            # Start the main monitoring loop
            monitoring_task = asyncio.create_task(self._stock_monitoring_loop())

            # Main control loop
            while self.running:
                try:
                    # Check startup schedule
                    self._check_startup_schedule()

                    # Check premarket schedule
                    self._check_premarket_schedule()

                    # Check volume surge schedule
                    self._check_volume_surge_schedule()

                    # Check startup processes
                    await self._check_startup_processes()

                    # Check premarket processes
                    await self._check_premarket_processes()

                    # Monitor CSV file
                    self._monitor_csv_file()

                    # Monitor manual symbols file
                    self._monitor_manual_symbols_file()

                    # Sleep for 30 seconds before next check
                    await asyncio.sleep(30)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"❌ Error in main control loop: {e}")
                    await asyncio.sleep(60)

            # Clean up
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            self.logger.error(f"❌ Error in momentum alerts system: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the momentum alerts system."""
        self.logger.info("🛑 Stopping Momentum Alerts System...")

        self.running = False

        # Stop any running startup processes
        for process_id, process_info in self.startup_processes.items():
            try:
                process = process_info['process']
                if process.poll() is None:
                    self.logger.info(f"🛑 Terminating startup process {process_id}")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            except Exception as e:
                self.logger.error(f"❌ Error stopping process {process_id}: {e}")

        # Stop any running premarket processes
        for process_id, process_info in self.premarket_processes.items():
            try:
                process = process_info['process']
                if process.poll() is None:
                    self.logger.info(f"🛑 Terminating premarket process {process_id}")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            except Exception as e:
                self.logger.error(f"❌ Error stopping premarket process {process_id}: {e}")

        self.logger.info("✅ Momentum Alerts System stopped")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Momentum Alerts System")

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (no actual alerts sent)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        system = MomentumAlertsSystem(
            test_mode=args.test,
            verbose=args.verbose
        )

        if args.test:
            print("🧪 Running in test mode - alerts will be logged but not sent")

        await system.start()

    except KeyboardInterrupt:
        print("\n👋 Momentum alerts system stopped by user")
    except Exception as e:
        print(f"❌ Failed to start momentum alerts system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
