#!/usr/bin/env python3
"""
Momentum Alerts System

This system monitors stocks from the market open top gainers CSV and generates momentum alerts
based on VWAP and EMA9 criteria. It follows the specification in specs/momentum_alert.md.

Process:
1. Startup: Run market_open_top_gainers.py every hour for 4 hours
2. Monitor: Watch for CSV file creation in ./historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv
3. Stock monitoring: Every minute, collect 30 minutes of 1-minute candlesticks for each stock
4. Momentum alerts: Check stocks above VWAP, above EMA9, and pass urgency filter
5. Integration: Send alerts to Bruce via Telegram

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
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import pytz

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_trade_api as tradeapi
from atoms.api.init_alpaca_client import init_alpaca_client
from atoms.api.stock_halt_detector import is_stock_halted, get_halt_status_emoji
from atoms.alerts.breakout_detector import BreakoutDetector
from atoms.telegram.telegram_post import TelegramPoster
from code.momentum_alerts_config import (
    get_momentum_alerts_config, get_volume_color_emoji,
    get_momentum_standard_color_emoji, get_momentum_short_color_emoji,
    get_urgency_level_dual
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

        # Create momentum alert directories
        self.momentum_alerts_dir.mkdir(parents=True, exist_ok=True)
        self.momentum_alerts_sent_dir.mkdir(parents=True, exist_ok=True)

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
        self.momentum_config = get_momentum_alerts_config()

        # Tracking
        self.monitored_symbols: Set[str] = set()
        self.last_csv_check = None
        self.startup_runs_completed = 0
        self.startup_schedule = []  # List of scheduled startup times

        # State
        self.running = False
        self.startup_processes = {}  # Track running startup scripts

        self.logger.info(f"🔧 Momentum Alerts System initialized in {'TEST' if test_mode else 'LIVE'} mode")
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
        """Schedule the startup script to run every hour for 4 hours."""
        current_time = datetime.now(self.et_tz)

        # Schedule runs starting from next hour, then every hour for 4 hours
        for i in range(4):
            # Start from next hour
            scheduled_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i+1)
            self.startup_schedule.append(scheduled_time)

        self.logger.info(f"📅 Scheduled {len(self.startup_schedule)} startup script runs:")
        for i, scheduled_time in enumerate(self.startup_schedule, 1):
            self.logger.info(f"   Run {i}: {scheduled_time.strftime('%H:%M:%S ET')}")

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
            "--max-price", "40.00",
            "--min-volume", "50000",
            "--top-gainers", "20",
            "--export-csv", "gainers_nasdaq_amex.csv",
            "--verbose"
        ]

        try:
            # Expand the tilde in the Python path
            cmd[0] = os.path.expanduser(cmd[0])

            # Create process with logging
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path.cwd())
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

    def _check_startup_schedule(self):
        """Check if it's time to run a startup script."""
        current_time = datetime.now(self.et_tz)

        # Check if we have any scheduled runs left
        while (self.startup_schedule and
               self.startup_runs_completed < len(self.startup_schedule) and
               current_time >= self.startup_schedule[self.startup_runs_completed]):

            asyncio.create_task(self._run_startup_script())
            self.startup_runs_completed += 1

            if self.startup_runs_completed < len(self.startup_schedule):
                next_run = self.startup_schedule[self.startup_runs_completed]
                self.logger.info(f"⏰ Next startup script run scheduled for: {next_run.strftime('%H:%M:%S ET')}")

    def _check_startup_processes(self):
        """Check the status of running startup processes."""
        completed_processes = []

        for process_id, process_info in self.startup_processes.items():
            process = process_info['process']

            if process.poll() is not None:  # Process has completed
                return_code = process.returncode
                runtime = datetime.now(self.et_tz) - process_info['start_time']

                if return_code == 0:
                    self.logger.info(f"✅ Startup script completed successfully (Runtime: {runtime})")
                else:
                    self.logger.error(f"❌ Startup script failed with return code {return_code} (Runtime: {runtime})")

                # Log any output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stdout:
                        self.logger.debug(f"Startup script output: {stdout[-500:]}")  # Last 500 chars
                except Exception:
                    pass

                completed_processes.append(process_id)

        # Remove completed processes
        for process_id in completed_processes:
            del self.startup_processes[process_id]

    def _load_csv_symbols(self) -> List[str]:
        """
        Load symbols from the gainers CSV file and additional data/{YYYYMMDD}.csv file.

        Returns:
            List of unique symbols to monitor
        """
        symbols = set()  # Use set to automatically handle uniqueness

        # Load from gainers CSV file
        if self.csv_file_path.exists():
            try:
                with open(self.csv_file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbol = row.get('symbol', '').strip().upper()
                        if symbol and not (len(symbol) == 5 and symbol.endswith('W')):  # Filter out 5-char warrants ending in W
                            symbols.add(symbol)

                self.logger.info(f"📊 Loaded {len(symbols)} symbols from gainers CSV")

            except Exception as e:
                self.logger.error(f"❌ Error loading gainers CSV file: {e}")

        # Load from additional data/{YYYYMMDD}.csv file if it exists
        compact_date = datetime.now(self.et_tz).strftime('%Y%m%d')  # YYYYMMDD format
        data_csv_path = Path("data") / f"{compact_date}.csv"

        if data_csv_path.exists():
            try:
                additional_count = 0
                with open(data_csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        symbol = row.get('symbol', '').strip().upper()
                        if symbol and not (len(symbol) == 5 and symbol.endswith('W')) and symbol not in symbols:  # Filter out 5-char warrants ending in W
                            symbols.add(symbol)
                            additional_count += 1

                self.logger.info(f"📊 Added {additional_count} unique symbols from data CSV: {data_csv_path}")

            except Exception as e:
                self.logger.error(f"❌ Error loading data CSV file {data_csv_path}: {e}")
        else:
            self.logger.debug(f"📄 Data CSV file not found: {data_csv_path}")

        # Convert set back to sorted list
        symbols_list = sorted(list(symbols))

        if symbols_list:
            self.logger.info(f"📊 Total unique symbols to monitor: {len(symbols_list)} - {symbols_list[:10]}{'...' if len(symbols_list) > 10 else ''}")
        else:
            self.logger.warning("⚠️ No symbols found in any CSV files")

        return symbols_list

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
            new_symbols = set(self._load_csv_symbols())
            added_symbols = new_symbols - self.monitored_symbols
            removed_symbols = self.monitored_symbols - new_symbols

            if added_symbols:
                self.logger.info(f"➕ Added symbols: {sorted(added_symbols)}")
            if removed_symbols:
                self.logger.info(f"➖ Removed symbols: {sorted(removed_symbols)}")

            self.monitored_symbols = new_symbols

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

    def _check_momentum_criteria(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Check if a stock meets momentum alert criteria.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data

        Returns:
            Alert data dictionary if criteria met, None otherwise
        """
        if data.empty or len(data) < 9:  # Need at least 9 bars for EMA9
            return None

        try:
            # Get latest bar
            latest_bar = data.iloc[-1]
            current_price = float(latest_bar['c'])  # Use single letter attribute
            current_vwap = float(latest_bar['vwap'])  # VWAP from stock data only
            current_volume = int(latest_bar['v'])  # Get current volume

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

            # Get momentum signal light icons from momentum config
            momentum_emoji = get_momentum_standard_color_emoji(momentum)
            momentum_short_emoji = get_momentum_short_color_emoji(momentum_short)

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

            # If we get here, all criteria are met
            alert_data = {
                'symbol': symbol,
                'current_price': current_price,
                'vwap': current_vwap,
                'ema_9': ema_9,
                'momentum': momentum,
                'momentum_short': momentum_short,
                'raw_momentum_20': raw_momentum_20,
                'raw_momentum_5': raw_momentum_5,
                'actual_time_diff': actual_time_diff if 'actual_time_diff' in locals() else 0,
                'actual_time_diff_short': actual_time_diff_short if 'actual_time_diff_short' in locals() else 0,
                'momentum_emoji': momentum_emoji,
                'momentum_short_emoji': momentum_short_emoji,
                'is_halted': is_halted,
                'halt_emoji': halt_emoji,
                'current_volume': current_volume,
                'volume_emoji': volume_emoji,
                'urgency': urgency,
                'timestamp': datetime.now(self.et_tz),
                'indicators': indicators
            }

            self.logger.info(f"✅ {symbol}: Momentum alert criteria met!")
            self.logger.info(f"   Price: ${current_price:.2f} | VWAP: ${current_vwap:.2f} | EMA9: ${ema_9:.2f}")

            # Show actual time periods used (handles halts/gaps)
            time_diff = actual_time_diff if 'actual_time_diff' in locals() else 0
            time_diff_short = actual_time_diff_short if 'actual_time_diff_short' in locals() else 0
            self.logger.info(f"   Momentum: {momentum:.2f}/min {momentum_emoji} | Raw: {raw_momentum_20:.2f}% ({time_diff:.1f}min)")
            self.logger.info(f"   Momentum Short: {momentum_short:.2f}/min {momentum_short_emoji} | Raw: {raw_momentum_5:.2f}% ({time_diff_short:.1f}min)")
            self.logger.info(f"   Volume: {current_volume:,} {volume_emoji} | Halt Status: {halt_emoji} | Urgency: {urgency}")

            return alert_data

        except Exception as e:
            self.logger.error(f"❌ Error checking momentum criteria for {symbol}: {e}")
            return None

    async def _send_momentum_alert(self, alert_data: Dict):
        """
        Send momentum alert to Bruce via Telegram.

        Args:
            alert_data: Alert information dictionary
        """
        try:
            symbol = alert_data['symbol']
            current_price = alert_data['current_price']
            vwap = alert_data['vwap']
            ema_9 = alert_data['ema_9']
            momentum = alert_data['momentum']
            momentum_short = alert_data['momentum_short']
            momentum_emoji = alert_data['momentum_emoji']
            momentum_short_emoji = alert_data['momentum_short_emoji']
            is_halted = alert_data['is_halted']
            halt_emoji = alert_data['halt_emoji']
            current_volume = alert_data['current_volume']
            volume_emoji = alert_data['volume_emoji']
            urgency = alert_data['urgency']
            timestamp = alert_data['timestamp']

            # Create alert message
            message_parts = [
                f"🚀 **MOMENTUM ALERT - {symbol}**",
                "",
                f"💰 **Price:** ${current_price:.2f}",
                f"📊 **VWAP:** ${vwap:.2f} ✅",
                f"📈 **EMA9:** ${ema_9:.2f} ✅",
                f"⚡ **Momentum:** {momentum:.2f}% {momentum_emoji}",
                f"⚡ **Momentum Short:** {momentum_short:.2f}% {momentum_short_emoji}",
                f"📈 **Volume:** {current_volume:,} {volume_emoji}",
                f"🚦 **Halt Status:** {halt_emoji}",
                f"🎯 **Urgency:** {urgency.upper()}",
                "",
                f"⏰ **Time:** {timestamp.strftime('%H:%M:%S ET')}",
                f"📅 **Date:** {timestamp.strftime('%Y-%m-%d')}"
            ]

            message = "\n".join(message_parts)

            # Save momentum alert to historical data
            self._save_momentum_alert(alert_data, message)

            if self.test_mode:
                self.logger.info(f"[TEST MODE] {message}")
            else:
                # Send to Bruce
                result = self.telegram_poster.send_message_to_user(message, "bruce", urgent=False)

                if result['success']:
                    self.logger.info(f"✅ Momentum alert sent to Bruce for {symbol}")
                    # Save sent alert to historical data
                    self._save_momentum_alert_sent(alert_data, message)
                else:
                    errors = result.get('errors', ['Unknown error'])
                    error_msg = ', '.join(errors) if isinstance(errors, list) else str(errors)
                    self.logger.error(f"❌ Failed to send momentum alert to Bruce for {symbol}: {error_msg}")

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
            alert_json = {
                'symbol': str(symbol),
                'current_price': float(alert_data['current_price']),
                'vwap': float(alert_data['vwap']),
                'ema_9': float(alert_data['ema_9']),
                'momentum': float(alert_data['momentum']),
                'momentum_short': float(alert_data['momentum_short']),
                'raw_momentum_20': float(alert_data['raw_momentum_20']),
                'raw_momentum_5': float(alert_data['raw_momentum_5']),
                'actual_time_diff': float(alert_data['actual_time_diff']),
                'actual_time_diff_short': float(alert_data['actual_time_diff_short']),
                'momentum_emoji': str(alert_data['momentum_emoji']),
                'momentum_short_emoji': str(alert_data['momentum_short_emoji']),
                'is_halted': bool(alert_data['is_halted']),
                'halt_emoji': str(alert_data['halt_emoji']),
                'current_volume': int(alert_data['current_volume']),
                'volume_emoji': str(alert_data['volume_emoji']),
                'urgency': str(alert_data['urgency']),
                'timestamp': timestamp.isoformat(),
                'message': str(message),
                'indicators': self._serialize_indicators(alert_data['indicators'])
            }

            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(alert_json, f, indent=2)

            self.logger.debug(f"📝 Saved momentum alert for {symbol} to {filename}")

        except Exception as e:
            self.logger.error(f"❌ Error saving momentum alert: {e}")

    def _save_momentum_alert_sent(self, alert_data: Dict, message: str) -> None:
        """
        Save sent momentum alert to historical data structure.

        Args:
            alert_data: Alert data dictionary
            message: Formatted alert message
        """
        try:
            symbol = alert_data['symbol']
            timestamp = alert_data['timestamp']

            # Create filename with timestamp
            filename = f"alert_{symbol}_{timestamp.strftime('%Y-%m-%d_%H%M%S')}.json"
            filepath = self.momentum_alerts_sent_dir / filename

            # Convert alert data to serializable format (same as save alert)
            alert_json = {
                'symbol': str(symbol),
                'current_price': float(alert_data['current_price']),
                'vwap': float(alert_data['vwap']),
                'ema_9': float(alert_data['ema_9']),
                'momentum': float(alert_data['momentum']),
                'momentum_short': float(alert_data['momentum_short']),
                'raw_momentum_20': float(alert_data['raw_momentum_20']),
                'raw_momentum_5': float(alert_data['raw_momentum_5']),
                'actual_time_diff': float(alert_data['actual_time_diff']),
                'actual_time_diff_short': float(alert_data['actual_time_diff_short']),
                'momentum_emoji': str(alert_data['momentum_emoji']),
                'momentum_short_emoji': str(alert_data['momentum_short_emoji']),
                'is_halted': bool(alert_data['is_halted']),
                'halt_emoji': str(alert_data['halt_emoji']),
                'current_volume': int(alert_data['current_volume']),
                'volume_emoji': str(alert_data['volume_emoji']),
                'urgency': str(alert_data['urgency']),
                'timestamp': timestamp.isoformat(),
                'message': str(message),
                'sent_to': 'bruce',
                'sent_at': datetime.now(self.et_tz).isoformat(),
                'indicators': self._serialize_indicators(alert_data['indicators'])
            }

            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(alert_json, f, indent=2)

            self.logger.debug(f"📝 Saved sent momentum alert for {symbol} to {filename}")

        except Exception as e:
            self.logger.error(f"❌ Error saving sent momentum alert: {e}")

    async def _stock_monitoring_loop(self):
        """Main stock monitoring loop - runs every minute."""
        while self.running:
            try:
                if self.monitored_symbols:
                    self.logger.debug(f"🔍 Monitoring {len(self.monitored_symbols)} symbols for momentum alerts")

                    # Collect stock data for all monitored symbols
                    symbols_list = list(self.monitored_symbols)
                    stock_data = await self._collect_stock_data(symbols_list)

                    # Check each symbol for momentum alerts
                    for symbol in symbols_list:
                        if symbol in stock_data:
                            alert_data = self._check_momentum_criteria(symbol, stock_data[symbol])
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
            # Schedule startup script runs
            self._schedule_startup_runs()

            # Start the main monitoring loop
            monitoring_task = asyncio.create_task(self._stock_monitoring_loop())

            # Main control loop
            while self.running:
                try:
                    # Check startup schedule
                    self._check_startup_schedule()

                    # Check startup processes
                    self._check_startup_processes()

                    # Monitor CSV file
                    self._monitor_csv_file()

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
