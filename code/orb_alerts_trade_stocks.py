"""
ORB Trade Stocks Monitor - Automated Trading System

This system monitors the superduper_alerts_sent/bullish/green directory for new superduper alerts
with green momentum indicators and executes automated trades through the TradeGenerator atom.
Only processes alerts with green light (🟢) momentum indicators for high-quality trading signals.

Usage:
    python3 code/orb_alerts_trade_stocks.py                           # Monitor current date superduper alerts
    python3 code/orb_alerts_trade_stocks.py --date 2025-08-01         # Monitor specific date superduper alerts
    python3 code/orb_alerts_trade_stocks.py --test                    # Run in test mode
    python3 code/orb_alerts_trade_stocks.py --no-telegram             # Disable telegram notifications
"""

import asyncio
import argparse
import logging
import sys
import json
import os
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
import pytz
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.alerts.trade_generator import TradeGenerator  # noqa: E402
from atoms.alerts.config import get_historical_root_dir, get_logs_root_dir  # noqa: E402
from atoms.telegram.orb_alerts import send_orb_alert  # noqa: E402
from atoms.telegram.telegram_post import TelegramPoster  # noqa: E402


class SuperduperAlertFileHandler(FileSystemEventHandler):
    """Handles new superduper alert files in the bullish/green superduper alerts directory."""

    def __init__(self, monitor, loop):
        self.monitor = monitor
        self.loop = loop

    def on_created(self, event):
        """Called when a new superduper alert file is created."""
        if not event.is_directory and event.src_path.endswith('.json'):
            # Schedule the coroutine in the main event loop from this thread
            asyncio.run_coroutine_threadsafe(
                self.monitor._process_new_superduper_alert_file(event.src_path),
                self.loop
            )


class ORBTradeStocksMonitor:
    """Main ORB Trade Stocks Monitor that watches for superduper alerts and executes trades."""

    def __init__(self, test_mode: bool = False, post_only_urgent: bool = False,
                 no_telegram: bool = False, date: Optional[str] = None,
                 backtesting_mode: bool = False):
        """
        Initialize ORB Trade Stocks Monitor.

        Args:
            test_mode: Run in test mode (no actual trades)
            post_only_urgent: Only send urgent telegram notifications
            no_telegram: Disable telegram notifications
            date: Date in YYYY-MM-DD format (default: current date)
            backtesting_mode: Run in backtesting mode (monitors runs/current)
        """
        # Setup logging
        self.logger = self._setup_logging()

        # Initialize trade generation atom
        self.trade_generator = None  # Will be initialized when directories are set up
        self.test_mode = test_mode
        self.post_only_urgent = post_only_urgent
        self.no_telegram = no_telegram
        self.backtesting_mode = backtesting_mode

        # Alert monitoring setup
        if date:
            # Validate date format
            try:
                datetime.strptime(date, '%Y-%m-%d')
                target_date = date
            except ValueError:
                raise ValueError(f"Invalid date format: {date}. Expected YYYY-MM-DD format.")
        else:
            # Use current date
            et_tz = pytz.timezone('US/Eastern')
            target_date = datetime.now(et_tz).strftime('%Y-%m-%d')

        self.target_date = target_date
        
        if self.backtesting_mode:
            # In backtesting mode, monitor superduper_alerts in runs/current and move to _sent
            from pathlib import Path
            current_run_dir = Path("runs/current/historical_data") / target_date
            self.source_alerts_dir = current_run_dir / "superduper_alerts/bullish"
            self.superduper_alerts_dir = current_run_dir / "superduper_alerts_sent/bullish/green"
            self.trades_dir = current_run_dir / "trades"
            
            self.logger.info(f"ORB Trade Stocks Monitor initialized in BACKTESTING mode")
            self.logger.info(f"Source alerts: {self.source_alerts_dir}")
            self.logger.info(f"Monitoring superduper alerts in: {self.source_alerts_dir}")
        else:
            # Normal mode - monitor superduper_alerts_sent directory
            historical_root = get_historical_root_dir()
            self.source_alerts_dir = historical_root.get_superduper_alerts_dir(target_date) / "bullish"
            self.superduper_alerts_dir = historical_root.get_superduper_alerts_sent_dir(target_date)
            self.trades_dir = historical_root.get_trades_dir(target_date)
            
            self.logger.info(f"ORB Trade Stocks Monitor initialized in LIVE mode")

        # Ensure directories exist
        self.superduper_alerts_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trade generator now that directory is set up
        self.trade_generator = TradeGenerator(self.trades_dir, test_mode)

        # File system watcher
        self.observer = Observer()
        self.file_handler = None  # Will be set when event loop is available

        # Telegram integration for targeted notifications
        self.telegram_poster = TelegramPoster() if not no_telegram else None

        # Processed alerts tracking
        self.processed_superduper_alerts = set()
        self.filtered_superduper_alerts = set()  # Track filtered superduper alerts
        self.executed_trades = set()  # Track executed trades

        self.logger.info(f"ORB Trade Stocks Monitor initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Target date: {target_date}")
        self.logger.info(f"Monitoring superduper alerts in: {self.superduper_alerts_dir}")
        self.logger.info(f"Trade results will be saved to: {self.trades_dir}")
        if no_telegram:
            self.logger.info("📵 Telegram notifications disabled")
        else:
            self.logger.info("📱 Telegram notifications enabled for trade execution")

    def _setup_logging(self) -> logging.Logger:
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
            
            # Setup file handler using centralized logs config
            try:
                logs_config = get_logs_root_dir()
                log_dir = logs_config.get_component_logs_dir("orb_trades")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                et_tz = pytz.timezone('US/Eastern')
                log_filename = f"orb_trades_{datetime.now(et_tz).strftime('%Y%m%d_%H%M%S')}.log"
                log_file_path = log_dir / log_filename
                
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            except Exception as e:
                # If file logging fails, continue with console logging only
                logger.warning(f"Could not setup file logging: {e}")
            
            logger.setLevel(logging.INFO)

        return logger

    def _validate_time_of_day_signal(self, superduper_alert_data: Dict) -> bool:
        """
        Validate both time of day emoji signal AND historical alert timestamp market hours.
        
        Requires BOTH conditions:
        1. Alert message contains green (🟢) or yellow (🟡) time signal
        2. Alert's historical timestamp is within market hours (M-F 9:30-16:00 ET)
        
        Args:
            superduper_alert_data: Superduper alert JSON data
            
        Returns:
            True if both emoji signal is green/yellow AND alert was during market hours, False otherwise
        """
        try:
            # FIRST: Check emoji-based time signal from alert message
            alert_message = superduper_alert_data.get('alert_message', '')
            
            if not alert_message:
                self.logger.warning(f"No alert message found in superduper alert data")
                return False
            
            # Look for time of day signal in the alert message
            # Expected format: "• Time of Day: 🟢 **MORNING POWER** (10:30 ET)"
            if "• Time of Day:" not in alert_message:
                self.logger.warning(f"No time of day signal found in alert message")
                return False
            
            # Extract the time signal emoji and period
            lines = alert_message.split('\n')
            time_line = None
            for line in lines:
                if "• Time of Day:" in line:
                    time_line = line.strip()
                    break
            
            if not time_line:
                self.logger.warning(f"Could not parse time of day line from alert message")
                return False
            
            # Check for red signal (🔴) which indicates CAUTION PERIOD
            if "🔴" in time_line:
                self.logger.info(f"Time of day emoji signal is RED (🔴) - rejecting trade")
                return False
            
            # IMPORTANT FIX: Don't trust the emoji in the message during backtesting!
            # The emoji was generated using backtesting runtime, not historical alert time.
            # We'll determine the correct emoji based on the historical timestamp below.
            
            self.logger.info(f"Found time signal in message: {time_line}")
            self.logger.info(f"⚠️  NOTE: During backtesting, ignoring emoji in message (may be wrong) - will validate using historical timestamp")
            
            # SECOND: Use historical alert timestamp instead of current system time
            et_tz = pytz.timezone('US/Eastern')
            
            # Get the historical timestamp from the alert (try multiple sources)
            historical_timestamp = None
            if 'latest_super_alert' in superduper_alert_data:
                timestamp_str = superduper_alert_data['latest_super_alert'].get('timestamp')
                if timestamp_str:
                    try:
                        historical_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except ValueError:
                        pass
            
            # Fallback to original_alert timestamp if available
            if not historical_timestamp and 'latest_super_alert' in superduper_alert_data:
                original_alert = superduper_alert_data['latest_super_alert'].get('original_alert', {})
                timestamp_str = original_alert.get('timestamp')
                if timestamp_str:
                    try:
                        historical_timestamp = datetime.fromisoformat(timestamp_str + '-04:00' if '+' not in timestamp_str else timestamp_str)
                    except ValueError:
                        pass
            
            if not historical_timestamp:
                self.logger.error(f"Could not extract historical timestamp from superduper alert - using current time as fallback")
                historical_et = datetime.now(et_tz)
            else:
                # Convert to ET timezone
                if historical_timestamp.tzinfo is None:
                    historical_et = et_tz.localize(historical_timestamp)
                else:
                    historical_et = historical_timestamp.astimezone(et_tz)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if historical_et.weekday() > 4:  # Saturday=5, Sunday=6
                self.logger.info(f"Market closed: Weekend (day {historical_et.weekday()}) - rejecting trade despite good emoji signal")
                return False
            
            # Get historical time as hour and minute
            historical_time = historical_et.time()
            
            # Market open: 9:30 AM ET
            market_open = historical_et.replace(hour=9, minute=30, second=0, microsecond=0).time()
            
            # Market close: 4:00 PM ET
            market_close = historical_et.replace(hour=16, minute=0, second=0, microsecond=0).time()
            
            # Check if historical alert time was within market hours
            is_open = market_open <= historical_time <= market_close
            
            if not is_open:
                self.logger.info(f"❌ Alert was outside market hours: {historical_et.strftime('%Y-%m-%d %H:%M:%S %Z')} (Hours: 9:30-16:00 ET) - rejecting trade")
                return False
            
            # Now determine the correct emoji signal based on historical timestamp
            hour_minute = historical_et.hour + historical_et.minute / 60.0
            
            if 9.5 <= hour_minute < 12.0:
                correct_emoji = "🟢"
                correct_period = "MORNING POWER"
            elif 12.0 <= hour_minute < 14.0:
                correct_emoji = "🟡" 
                correct_period = "LUNCH HOUR"
            elif 14.0 <= hour_minute <= 16.0:
                correct_emoji = "🟢"
                correct_period = "AFTERNOON POWER"
            else:
                # Should not reach here since we already checked market hours
                self.logger.warning(f"Historical time {hour_minute} doesn't match expected market periods")
                correct_emoji = "⚫"
                correct_period = "UNKNOWN"
            
            # Only allow green or yellow signals
            if correct_emoji in ["🟢", "🟡"]:
                self.logger.info(f"✅ BOTH conditions met: Historical time shows {correct_emoji} {correct_period} signal AND alert was during market hours ({historical_et.strftime('%Y-%m-%d %H:%M:%S %Z')}) - allowing trade")
                return True
            else:
                self.logger.info(f"❌ Historical time shows {correct_emoji} signal - rejecting trade")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating time of day signal: {e}")
            # Fail safe: don't trade if we can't determine both conditions
            return False

    def _get_telegram_user_for_account(self, account_name: str) -> Optional[str]:
        """
        Get telegram username that matches the account name directly.
        
        Args:
            account_name: Account name from alpaca config (e.g., "Bruce", "Dale", "Primary")
            
        Returns:
            Account name as telegram username (direct match), or None if not found
        """
        # Account names should match telegram usernames directly
        if account_name:
            self.logger.debug(f"Using account name '{account_name}' as telegram username")
            return account_name
        else:
            self.logger.warning(f"Empty account name provided")
            return None

    async def _process_new_superduper_alert_file(self, file_path: str) -> None:
        """Process a new superduper alert file and execute trade if conditions are met."""
        try:
            # Avoid processing the same file multiple times
            if file_path in self.processed_superduper_alerts:
                return

            self.processed_superduper_alerts.add(file_path)

            # Wait a moment for file to be fully written
            await asyncio.sleep(0.1)

            with open(file_path, 'r') as f:
                superduper_alert_data = json.load(f)

            symbol = superduper_alert_data.get('symbol', 'UNKNOWN')

            # Check time of day signal filtering before executing trade
            if not self._validate_time_of_day_signal(superduper_alert_data):
                self.filtered_superduper_alerts.add(file_path)
                self.logger.info(f"🔴 Trade rejected for {symbol}: Market is closed (outside trading hours)")
                return

            # In backtesting mode, move validated alert to superduper_alerts_sent directory
            if self.backtesting_mode:
                import shutil
                from pathlib import Path
                
                source_path = Path(file_path)
                dest_path = self.superduper_alerts_dir / source_path.name
                
                try:
                    # Copy (don't move) the file to preserve original in source directory for analysis
                    shutil.copy2(source_path, dest_path)
                    self.logger.info(f"📋 Copied validated alert to sent directory: {dest_path.name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to copy alert to sent directory: {e}")

            # Use TradeGenerator to create and execute trades on all accounts
            trade_filenames = self.trade_generator.create_and_execute_trade(superduper_alert_data)

            if trade_filenames and len(trade_filenames) > 0:
                self.logger.info(f"✅ {len(trade_filenames)} trades executed for {symbol}")
                
                # Process each executed trade
                for trade_filename in trade_filenames:
                    self.executed_trades.add(trade_filename)
                    self.logger.info(f"✅ Trade executed and saved: {trade_filename}")

                    # Send Telegram notification (if enabled)
                    if not self.no_telegram:
                        try:
                            # Extract trade execution details for notification
                            trade_file_path = self._find_trade_file(trade_filename)

                            if trade_file_path and trade_file_path.exists():
                                with open(trade_file_path, 'r') as f:
                                    trade_data = json.load(f)

                                execution_status = trade_data.get('execution_status', {})
                                success = execution_status.get('success', 'no')
                                amount = trade_data.get('auto_amount', 0)
                                account_name = trade_data.get('account_name', 'Unknown')
                                account_type = trade_data.get('account_type', 'Unknown')
                                account = f"{account_name}/{account_type}"

                                # Create trade notification message with trading parameters
                                trailing_percent = trade_data.get('trailing_percent', 0)
                                take_profit_percent = trade_data.get('take_profit_percent', 0)
                                reason = execution_status.get('reason', 'No reason provided')
                                trade_message = self._create_trade_notification_message(
                                    symbol, success, amount, account, execution_status, trailing_percent, take_profit_percent, reason)

                                # Determine urgency - all executed trades are considered urgent
                                is_urgent = True

                                # Send targeted notification to specific user based on account
                                result = await self._send_trade_notification(trade_message, is_urgent, account_name)

                                if result['success']:
                                    if result.get('skipped'):
                                        reason = result.get('reason', 'Non-urgent filtered')
                                        target_user = result.get('target_user', 'unknown')
                                        self.logger.info(f"⏭️ Telegram trade notification skipped for {target_user}: {reason}")
                                    else:
                                        emoji = "💰"
                                        target_user = result.get('target_user', 'unknown')
                                        account_name = result.get('account_name', 'unknown')
                                        msg = f"📤 {emoji} Telegram trade notification sent to {target_user} for {account_name} account"
                                        self.logger.info(msg)
                                else:
                                    target_user = result.get('target_user', 'unknown')
                                    account_name = result.get('account_name', 'unknown')
                                    errors = result.get('errors', ['Unknown error'])
                                    error_msg = ', '.join(errors) if isinstance(errors, list) else str(errors)
                                    self.logger.warning(f"❌ Telegram trade notification failed for {target_user} ({account_name}): {error_msg}")

                        except Exception as e:
                            self.logger.error(f"❌ Error sending Telegram trade notification for {trade_filename}: {e}")
                    else:
                        self.logger.info(f"💰 Trade executed for {symbol} on account (Telegram disabled): {trade_filename}")
            else:
                # Alert was filtered out (no green momentum or other reasons)
                self.filtered_superduper_alerts.add(file_path)
                self.logger.info(f"🚫 Trade skipped for {symbol}: Failed momentum or time signal filtering criteria")

        except Exception as e:
            self.logger.error(f"Error processing superduper alert file {file_path}: {e}")

    def _find_trade_file(self, trade_filename: str) -> Optional[Path]:
        """Find the full path to the trade file."""
        # Search through account directories for the trade file
        for account_dir in self.trades_dir.iterdir():
            if account_dir.is_dir():
                for account_type_dir in account_dir.iterdir():
                    if account_type_dir.is_dir():
                        trade_file = account_type_dir / trade_filename
                        if trade_file.exists():
                            return trade_file
        return None

    def _create_trade_notification_message(self, symbol: str, success: str, amount: int,
                                           account: str, execution_status: Dict, 
                                           trailing_percent: float = 0, take_profit_percent: float = 0, reason: str = "No reason provided") -> str:
        """Create a formatted trade notification message."""
        success_emoji = "✅" if success == "yes" else "🔄"  # ✅ for real success, 🔄 for dry run
        dry_run_text = " (DRY RUN)" if execution_status.get('dry_run_executed', False) else ""

        # Color-code the status
        if success.upper() == "YES":
            status_text = "🟢 **YES**"
        else:
            status_text = "🔴 **NO**"

        message_parts = [
            "💰💰 **TRADE EXECUTED** 💰💰",
            "",
            f"🎯 **{symbol}** {success_emoji} **${amount}**{dry_run_text}",
            f"📊 **Account:** {account}",
            "",
            "📈 **Trade Details:**",
            "• Command: Buy Market + Trailing Sell + Take Profit",
            f"• Amount: ${amount}",
            f"• Trailing Stop: {trailing_percent}%",
            f"• Take Profit: {take_profit_percent}%",
            f"• Status: {status_text}",
            f"• Reason: {reason}",
            "• Trigger: Green Momentum Superduper Alert",
        ]
        
        # Add retry information if available
        retry_attempts = execution_status.get('retry_attempts', [])
        successful_attempt = execution_status.get('successful_attempt')
        max_retries_reached = execution_status.get('max_retries_reached', False)
        
        if retry_attempts or successful_attempt:
            message_parts.append("")
            message_parts.append("🔄 **Retry Information:**")
            
            if successful_attempt and successful_attempt > 1:
                message_parts.append(f"• Succeeded on attempt {successful_attempt}/3")
                message_parts.append(f"• Previous failures: {len(retry_attempts)}")
            elif max_retries_reached:
                message_parts.append("• Failed after 3 retry attempts")
                message_parts.append(f"• All attempts exhausted")
            
            # Show details of retry attempts if there were any
            if retry_attempts:
                for attempt in retry_attempts[:2]:  # Show max 2 retry details to keep message concise
                    attempt_num = attempt.get('attempt', 'N/A')
                    attempt_reason = attempt.get('reason', 'Unknown error')[:50] + "..."  # Truncate long reasons
                    message_parts.append(f"• Attempt {attempt_num}: {attempt_reason}")
        
        message_parts.extend([
            "",
            f"⏰ **Executed:** {datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M:%S ET')}",
        ])

        if execution_status.get('dry_run_executed', False):
            message_parts.insert(-1, "🧪 **DRY RUN MODE** - No actual trade placed")

        return "\n".join(message_parts)

    async def _send_trade_notification(self, message: str, is_urgent: bool, account_name: str) -> Dict:
        """Send targeted trade notification via Telegram to specific user based on account."""
        try:
            if not self.telegram_poster:
                return {
                    'success': True,
                    'sent_count': 0,
                    'failed_count': 0,
                    'errors': ['Telegram notifications disabled'],
                    'target_user': 'N/A'
                }

            # Get telegram username (direct match with account name)
            target_username = self._get_telegram_user_for_account(account_name)
            
            if not target_username:
                self.logger.warning(f"No telegram user found for account '{account_name}', skipping notification")
                return {
                    'success': False,
                    'sent_count': 0,
                    'failed_count': 1,
                    'errors': [f'No telegram user mapping for account: {account_name}'],
                    'target_user': account_name,
                    'account_name': account_name
                }

            # Check if we should send based on urgency filter
            if self.post_only_urgent and not is_urgent:
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'Non-urgent message filtered by post_only_urgent setting',
                    'sent_count': 0,
                    'target_user': target_username,
                    'account_name': account_name
                }

            # Send targeted message to specific user
            result = self.telegram_poster.send_message_to_user(message, target_username, urgent=is_urgent)
            
            # Add account information to the result
            result['account_name'] = account_name
            
            return result

        except Exception as e:
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 1,
                'errors': [f'Exception occurred: {str(e)}'],
                'target_user': account_name,
                'account_name': account_name
            }

    async def _scan_existing_superduper_alerts(self) -> None:
        """Scan existing superduper alert files on startup."""
        try:
            if not self.superduper_alerts_dir.exists():
                self.logger.info("No existing superduper alerts directory found")
                return

            superduper_alert_files = list(self.superduper_alerts_dir.glob("superduper_alert_*.json"))
            self.logger.info(f"Scanning {len(superduper_alert_files)} existing superduper alert files...")

            processed_count = 0
            for superduper_alert_file in superduper_alert_files:
                await self._process_new_superduper_alert_file(str(superduper_alert_file))
                processed_count += 1

            self.logger.info(f"Processed {processed_count} existing superduper alerts")

        except Exception as e:
            self.logger.error(f"Error scanning existing superduper alerts: {e}")

    async def start(self) -> None:
        """Start the ORB Trade Stocks Monitor."""
        self.logger.info("Starting ORB Trade Stocks Monitor...")

        try:
            # Initialize file handler with current event loop
            current_loop = asyncio.get_running_loop()
            self.file_handler = SuperduperAlertFileHandler(self, current_loop)

            # Skip processing existing superduper alerts on startup to avoid executing trades on stale alerts
            # await self._scan_existing_superduper_alerts()

            # Start file system monitoring
            if self.backtesting_mode:
                # In backtesting mode, monitor the source alerts directory
                monitor_dir = self.source_alerts_dir
                if monitor_dir.exists():
                    self.observer.schedule(self.file_handler, str(monitor_dir), recursive=False)
                    self.observer.start()
                    self.logger.info(f"Started monitoring {monitor_dir}")
                else:
                    self.logger.warning(f"Source superduper alerts directory does not exist: {monitor_dir}")
                    # Create the directory structure if it doesn't exist
                    monitor_dir.mkdir(parents=True, exist_ok=True)
                    self.observer.schedule(self.file_handler, str(monitor_dir), recursive=False)
                    self.observer.start()
                    self.logger.info(f"Created and started monitoring {monitor_dir}")
            else:
                # Normal mode - monitor superduper_alerts_sent
                if self.superduper_alerts_dir.exists():
                    self.observer.schedule(self.file_handler, str(self.superduper_alerts_dir), recursive=False)
                    self.observer.start()
                    self.logger.info(f"Started monitoring {self.superduper_alerts_dir}")
                else:
                    self.logger.warning(f"Superduper alerts directory does not exist: {self.superduper_alerts_dir}")

            # Print status
            print("\n" + "="*80)
            print("💰 ORB TRADE STOCKS MONITOR ACTIVE")
            print(f"📅 Target date: {self.target_date}")
            print(f"📁 Monitoring: {self.superduper_alerts_dir}")
            print(f"💾 Trade results: {self.trades_dir}")
            print("✅ Filtering: Green momentum (🟢) AND green/yellow time signals (🟢🟡) AND actual market hours (M-F 9:30-16:00 ET)")
            if self.no_telegram:
                print("📵 Telegram: Notifications disabled")
            else:
                print("📱 Telegram: Targeted trade notifications enabled (account-specific users)")
                print("🎯 Direct matching: Account name = Telegram username")
            if self.test_mode:
                print("🧪 TEST MODE: Trades will be marked as [TEST MODE]")
            if self.post_only_urgent:
                print("⚡ URGENT ONLY: Only urgent trade notifications will be sent via Telegram")
            print(f"💡 Trade Limits: Per-account limits configured ({'TEST' if self.test_mode else 'LIVE'} mode)")
            print("="*80 + "\n")

            # Keep running
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in trade stocks monitor: {e}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the ORB Trade Stocks Monitor."""
        self.logger.info("Stopping ORB Trade Stocks Monitor...")

        try:
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            self.logger.info("ORB Trade Stocks Monitor stopped")
        except Exception as e:
            self.logger.error(f"Error stopping monitor: {e}")

    def get_statistics(self) -> dict:
        """Get monitoring statistics."""
        return {
            'trades_executed': len(self.executed_trades),
            'superduper_alerts_processed': len(self.processed_superduper_alerts),
            'superduper_alerts_filtered': len(self.filtered_superduper_alerts),
            'monitoring_directory': str(self.superduper_alerts_dir),
            'trades_directory': str(self.trades_dir),
            'test_mode': self.test_mode,
            'post_only_urgent': self.post_only_urgent,
            'no_telegram': self.no_telegram
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Trade Stocks Monitor - Automated Trading System")

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (dry run)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--post-only-urgent",
        action="store_true",
        help="Only send telegram notifications for urgent trade executions"
    )

    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable telegram notifications"
    )

    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYY-MM-DD format (default: current date)"
    )

    parser.add_argument(
        "--backtesting",
        action="store_true",
        help="Run in backtesting mode (monitors runs/current instead of historical_data)"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and start monitor
    try:
        monitor = ORBTradeStocksMonitor(
            test_mode=args.test,
            post_only_urgent=args.post_only_urgent,
            no_telegram=args.no_telegram,
            date=args.date,
            backtesting_mode=args.backtesting
        )

        if args.test:
            print("Running in test mode - trades will be marked as [TEST MODE]")

        if args.post_only_urgent:
            print("Urgent only mode - only urgent trade notifications will be sent via Telegram")

        if args.no_telegram:
            print("Telegram disabled - no telegram notifications will be sent")

        await monitor.start()

    except Exception as e:
        logging.error(f"Failed to start ORB Trade Stocks Monitor: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Install required dependency
    try:
        import watchdog
    except ImportError:
        print("Installing required dependency: watchdog")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
        import watchdog

    asyncio.run(main())
