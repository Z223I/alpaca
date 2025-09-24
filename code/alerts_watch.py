#!/usr/bin/env python3
"""
Alerts Watch - Automated Market Hours Alert System

This system manages ORB alert processes during market hours (9:30 AM - 4:00 PM ET, Mon-Fri):
1. Starts 5 alert processes at market open
2. Monitors and restarts them if they fail
3. Stops them at market close
4. Runs post-market analysis and summary
5. Sends daily summary to Bruce via Telegram
6. Automatically shuts down after all EOD tasks complete

Managed Processes:
- ORB Alerts (Basic alert generation)
- ORB Alerts Monitor (Super alerts)
- ORB Superduper Alerts
- ORB Trade Execution
- VWAP Bounce Alerts (Bounce pattern detection)

Post-Market Tasks:
- ORB Alerts Summary
- ORB Analysis with Charts
- Telegram summary to Bruce
- End-of-day shutdown notification
- Automatic watchdog termination

Usage: Run once per day - will automatically shutdown after market close and EOD tasks.
"""

import sys
import os
import time
import signal
import subprocess
import threading
import schedule
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Optional, List
import pytz

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.telegram.telegram_post import send_message

class AlertsWatchdog:
    """Watchdog for managing ORB alert processes during market hours."""

    def __init__(self):
        # Get project root directory
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"

        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(exist_ok=True)

        # Create subdirectories for each program
        self.watchdog_logs_dir = self.logs_dir / "alerts_watchdog"
        self.watchdog_logs_dir.mkdir(exist_ok=True)

        # Setup main watchdog logging
        et_now = datetime.now(pytz.timezone('US/Eastern'))
        timestamp = et_now.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.watchdog_logs_dir / f"alerts_watch_{timestamp}.log"

        # Market hours configuration (Eastern Time)
        self.et_tz = pytz.timezone('US/Eastern')
        self.market_open_time = dt_time(9, 30)  # 9:30 AM ET
        self.market_close_time = dt_time(16, 0)  # 4:00 PM ET

        # Process management
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        self.market_open = False

        # Alert process configurations
        self.alert_processes = {
            'orb_alerts': {
                'script': 'code/orb_alerts.py',
                'args': ['--verbose'],
                'python_cmd': 'python3',
                'log_dir': 'orb_alerts'
            },
            'orb_monitor': {
                'script': 'code/orb_alerts_monitor.py',
                'args': ['--no-telegram', '--verbose'],
                'python_cmd': 'python3',
                'log_dir': 'orb_monitor'
            },
            'orb_superduper': {
                'script': 'code/orb_alerts_monitor_superduper.py',
                'args': ['--verbose'],
                'python_cmd': 'python',
                'log_dir': 'orb_superduper'
            },
            'orb_trades': {
                'script': 'code/orb_alerts_trade_stocks.py',
                'args': ['--verbose'],
                'python_cmd': 'python',
                'log_dir': 'orb_trades'
            },
            'vwap_bounce_alerts': {
                'script': 'code/vwap_bounce_alerts.py',
                'args': ['--verbose'],
                'python_cmd': 'python3',
                'log_dir': 'vwap_bounce_alerts'
            }
        }

        # Create log subdirectories for each process
        for process_name, config in self.alert_processes.items():
            process_log_dir = self.logs_dir / config['log_dir']
            process_log_dir.mkdir(exist_ok=True)

        # Post-market analysis scripts
        self.post_market_scripts = [
            {
                'script': 'code/orb_alerts_summary.py',
                'python_cmd': 'python',
                'args': [],
                'log_dir': 'orb_alerts_summary'
            },
            {
                'script': 'code/orb.py',
                'python_cmd': 'python3',
                'args': [],
                'log_dir': 'orb_analysis'
            }
        ]

        # PNL reports will be generated dynamically for each account

        # Create log subdirectories for post-market scripts
        for script_config in self.post_market_scripts:
            script_log_dir = self.logs_dir / script_config['log_dir']
            script_log_dir.mkdir(exist_ok=True)

        # Bruce's chat ID for Telegram notifications
        self.bruce_chat_id = None  # Will be loaded from user manager

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._log("🔧 Alerts Watchdog initialized")
        self._log(f"📁 Project root: {self.project_root}")
        self._log(f"📁 Logs directory: {self.logs_dir}")
        self._log(f"📋 Watchdog log: {self.log_file}")
        self._log(f"📁 Process logs will be saved to: {self.logs_dir}/[program_name]/")
        self._log(f"🕘 Market hours: {self.market_open_time} - {self.market_close_time} ET")

        # Create PNL logs directory
        pnl_log_dir = self.logs_dir / "pnl_reports"
        pnl_log_dir.mkdir(exist_ok=True)

    def _get_today_date(self) -> str:
        """Get today's date in YYYY-MM-DD format (Eastern Time)."""
        et_now = datetime.now(self.et_tz)
        return et_now.strftime('%Y-%m-%d')

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._log(f"🛑 Received signal {signum}, shutting down alerts watchdog...")
        self.running = False
        self._stop_all_processes()

    def _log(self, message: str, level: str = "INFO"):
        """Log message with timestamp to both console and file."""
        et_now = datetime.now(self.et_tz)
        timestamp = et_now.strftime("%Y-%m-%d %H:%M:%S ET")
        log_entry = f"[{timestamp}] {level}: {message}"

        # Print to console
        print(log_entry)
        sys.stdout.flush()

        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\n")
                f.flush()
        except Exception as e:
            et_now = datetime.now(self.et_tz)
            timestamp = et_now.strftime("%Y-%m-%d %H:%M:%S ET")
            print(f"[{timestamp}] ERROR: Failed to write to log file: {e}")

    def start_watchdog(self):
        """Start the alerts watchdog service."""
        try:
            self._log("🚀 Starting Alerts Watchdog")
            self._log("📅 Monitoring market hours Monday-Friday")
            self._log("⏰ Market Open: 9:30 AM ET - Starting alert processes")
            self._log("⏰ Market Close: 4:00 PM ET - Stopping processes and running summary")
            self._log("💡 Send CTRL+C to stop watchdog")

            self.running = True

            # Setup schedule for market hours
            self._setup_market_schedule()

            # Check if we're currently in market hours
            self._check_initial_market_status()

            # Main monitoring loop
            while self.running:
                try:
                    # Run scheduled tasks
                    schedule.run_pending()

                    # Monitor running processes if market is open
                    if self.market_open:
                        self._monitor_alert_processes()

                    # Sleep for a bit
                    time.sleep(30)  # Check every 30 seconds

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self._log(f"❌ Watchdog error: {e}", "ERROR")
                    time.sleep(60)  # Wait longer on error

            self._log("🛑 Alerts watchdog stopped")

        except Exception as e:
            self._log(f"❌ Failed to start alerts watchdog: {e}", "ERROR")
            return False

        return True

    def _get_trading_account_combinations(self):
        """
        Get unique account-name/account combinations where auto_trade is enabled.

        Returns:
            List of tuples: [(account_name, account_type), ...]
        """
        try:
            # Add the code directory to Python path to import alpaca_config
            import sys
            code_dir = self.project_root / "code"
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))

            from alpaca_config import get_current_config
            config = get_current_config()

            account_combinations = []

            # Loop through all accounts and account types to find auto_trade enabled ones
            for account_name, account_config in config.providers["alpaca"].accounts.items():
                for account_type in ["paper", "live", "cash"]:
                    env_config = getattr(account_config, account_type)
                    if env_config.auto_trade == "yes":
                        account_combinations.append((account_name, account_type))

            self._log(f"Found {len(account_combinations)} trading account combinations: {account_combinations}")
            return account_combinations

        except ImportError as e:
            self._log(f"❌ Could not import alpaca_config: {e}", "ERROR")
            return []
        except Exception as e:
            self._log(f"❌ Error loading trading accounts: {e}", "ERROR")
            return []

    def _generate_pnl_reports(self):
        """Generate PNL reports for all trading accounts and send to Bruce via Telegram."""
        self._log("💰 Generating PNL reports for all trading accounts...")

        # Get all trading account combinations
        account_combinations = self._get_trading_account_combinations()

        if not account_combinations:
            self._log("⚠️ No trading accounts found with auto_trade enabled", "WARN")
            return

        pnl_results = {}

        for account_name, account_type in account_combinations:
            self._log(f"📊 Generating PNL report for {account_name}/{account_type}...")

            try:
                # Execute alpaca.py --PNL for this account
                python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')
                alpaca_script = self.project_root / "code" / "alpaca.py"

                cmd = [
                    python_path,
                    str(alpaca_script),
                    "--account-name", account_name,
                    "--account", account_type,
                    "--PNL"
                ]

                # Setup PNL-specific log file
                et_now = datetime.now(self.et_tz)
                timestamp = et_now.strftime("%Y%m%d_%H%M%S")
                pnl_log_dir = self.logs_dir / "pnl_reports"
                pnl_log_file = pnl_log_dir / f"pnl_{account_name}_{account_type}_{timestamp}.log"

                self._log(f"📝 PNL logs: {pnl_log_file}")

                # Execute PNL command with output saved to log file
                with open(pnl_log_file, 'w') as log_file:
                    # Write header
                    et_now = datetime.now(self.et_tz)
                    log_file.write(f"# PNL Report for {account_name}/{account_type}\n")
                    log_file.write(f"# Generated: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
                    log_file.write(f"# Command: {' '.join(cmd)}\n")
                    log_file.write("# " + "="*50 + "\n\n")
                    log_file.flush()

                    result = subprocess.run(
                        cmd,
                        cwd=str(self.project_root),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=60  # 1 minute timeout for PNL report
                    )

                    # Write completion info
                    et_now = datetime.now(self.et_tz)
                    log_file.write(f"\n\n# Completed: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
                    log_file.write(f"# Return code: {result.returncode}\n")

                # Read the PNL output for Telegram notification
                try:
                    with open(pnl_log_file, 'r') as f:
                        log_content = f.read()
                        # Extract just the actual PNL output (skip headers)
                        output_lines = []
                        in_output = False
                        for line in log_content.split('\n'):
                            if line.startswith('# ========'):
                                in_output = True
                                continue
                            elif in_output and not line.startswith('# Completed:') and not line.startswith('# Return code:'):
                                output_lines.append(line)
                        pnl_output = '\n'.join(output_lines).strip()
                except Exception:
                    pnl_output = "PNL output saved to log file"

                pnl_results[f"{account_name}/{account_type}"] = {
                    'success': result.returncode == 0,
                    'output': pnl_output,
                    'log_file': str(pnl_log_file),
                    'error': None if result.returncode == 0 else f"PNL command failed with return code {result.returncode}"
                }

                if result.returncode == 0:
                    self._log(f"✅ PNL report generated successfully for {account_name}/{account_type}")
                else:
                    self._log(f"❌ PNL report failed for {account_name}/{account_type}: return code {result.returncode}", "ERROR")

                # Send individual Telegram notification to Bruce for this account's PNL
                self._send_pnl_notification_to_bruce(account_name, account_type, pnl_results[f"{account_name}/{account_type}"])

            except subprocess.TimeoutExpired:
                error_msg = f"PNL command timed out after 60 seconds for {account_name}/{account_type}"
                self._log(f"❌ {error_msg}", "ERROR")
                pnl_results[f"{account_name}/{account_type}"] = {
                    'success': False,
                    'output': '',
                    'log_file': str(pnl_log_file) if 'pnl_log_file' in locals() else None,
                    'error': error_msg
                }
            except Exception as e:
                error_msg = f"Error generating PNL report for {account_name}/{account_type}: {e}"
                self._log(f"❌ {error_msg}", "ERROR")
                pnl_results[f"{account_name}/{account_type}"] = {
                    'success': False,
                    'output': '',
                    'log_file': None,
                    'error': error_msg
                }

        self._log(f"💰 PNL report generation completed for {len(account_combinations)} accounts")
        return pnl_results

    def _send_pnl_notification_to_bruce(self, account_name: str, account_type: str, pnl_result: dict):
        """Send individual PNL report to Bruce via Telegram."""
        try:
            account_key = f"{account_name}/{account_type}"

            # Create PNL notification message
            if pnl_result['success']:
                # Extract key PNL information from output
                pnl_output = pnl_result['output']

                message_parts = [
                    f"💰 **PNL Report - {account_key}**",
                    "",
                    "📊 **Daily Profit/Loss Summary:**"
                ]

                # Add the PNL output (truncate if too long for Telegram)
                if pnl_output:
                    # Telegram messages have a 4096 character limit
                    if len(pnl_output) > 3500:
                        pnl_output = pnl_output[:3500] + "\n...\n[Output truncated - see log file for full report]"

                    # Format the PNL output with proper markdown
                    formatted_output = pnl_output.replace('$', '\\$')  # Escape dollar signs for Telegram
                    message_parts.extend([
                        "```",
                        formatted_output,
                        "```"
                    ])
                else:
                    message_parts.append("• No PNL data available")

                message_parts.extend([
                    "",
                    f"📋 **Log File:** {pnl_result['log_file']}",
                    f"⏰ **Generated:** {datetime.now(self.et_tz).strftime('%H:%M:%S ET')}"
                ])
            else:
                message_parts = [
                    f"❌ **PNL Report Failed - {account_key}**",
                    "",
                    f"**Error:** {pnl_result['error']}",
                    f"📋 **Log File:** {pnl_result.get('log_file', 'N/A')}",
                    f"⏰ **Generated:** {datetime.now(self.et_tz).strftime('%H:%M:%S ET')}"
                ]

            pnl_message = "\n".join(message_parts)

            # Send to Bruce
            try:
                from atoms.telegram.user_manager import UserManager
                user_manager = UserManager()
                active_users = user_manager.get_active_users()

                # Look for Bruce specifically
                bruce_users = [u for u in active_users if 'bruce' in u.get('username', '').lower()]

                if bruce_users:
                    from atoms.telegram.telegram_post import TelegramPoster
                    telegram_poster = TelegramPoster()

                    for user in bruce_users:
                        result = telegram_poster.send_message_to_user(pnl_message, user['username'])
                        if result['success']:
                            self._log(f"✅ PNL notification sent to Bruce for {account_key}")
                        else:
                            errors = result.get('errors', ['Unknown error'])
                            error_msg = ', '.join(errors) if isinstance(errors, list) else str(errors)
                            self._log(f"❌ Failed to send PNL notification to Bruce for {account_key}: {error_msg}", "ERROR")
                else:
                    self._log(f"⚠️ No Bruce user found for PNL notification ({account_key})", "WARN")

            except Exception as e:
                self._log(f"❌ Error sending PNL notification to Bruce for {account_key}: {e}", "ERROR")

        except Exception as e:
            self._log(f"❌ Error creating PNL notification for {account_name}/{account_type}: {e}", "ERROR")

    def _setup_market_schedule(self):
        """Setup scheduled tasks for market open/close."""
        # Convert ET market hours to local time for the schedule library
        # Since schedule library uses local system time, we need to convert ET to local

        # Create ET datetime objects for today
        et_now = datetime.now(self.et_tz)
        et_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        et_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Convert to local timezone
        local_tz = pytz.timezone('US/Central')  # Assuming system is in Central Time
        try:
            # Try to detect system timezone automatically
            import time
            local_tz_name = time.tzname[0]
            if 'CST' in local_tz_name or 'CDT' in local_tz_name:
                local_tz = pytz.timezone('US/Central')
            elif 'MST' in local_tz_name or 'MDT' in local_tz_name:
                local_tz = pytz.timezone('US/Mountain')
            elif 'PST' in local_tz_name or 'PDT' in local_tz_name:
                local_tz = pytz.timezone('US/Pacific')
            else:
                local_tz = pytz.timezone('US/Eastern')  # Default to ET if system is already ET
        except:
            # Fallback to detecting offset
            local_now = datetime.now()
            et_equiv = datetime.now(self.et_tz).replace(tzinfo=None)
            offset_hours = int((local_now - et_equiv).total_seconds() / 3600)

            if offset_hours == -1:  # 1 hour behind ET = Central
                local_tz = pytz.timezone('US/Central')
            elif offset_hours == -2:  # 2 hours behind ET = Mountain
                local_tz = pytz.timezone('US/Mountain')
            elif offset_hours == -3:  # 3 hours behind ET = Pacific
                local_tz = pytz.timezone('US/Pacific')
            else:
                local_tz = self.et_tz  # Same as ET

        local_open = et_open.astimezone(local_tz)
        local_close = et_close.astimezone(local_tz)

        open_time_str = local_open.strftime("%H:%M")
        close_time_str = local_close.strftime("%H:%M")

        self._log(f"🕘 Converting ET market hours to local time:")
        self._log(f"   Market Open: 09:30 ET = {open_time_str} local")
        self._log(f"   Market Close: 16:00 ET = {close_time_str} local")

        # Market open - Monday to Friday at converted local time
        schedule.every().monday.at(open_time_str).do(self._market_open_handler)
        schedule.every().tuesday.at(open_time_str).do(self._market_open_handler)
        schedule.every().wednesday.at(open_time_str).do(self._market_open_handler)
        schedule.every().thursday.at(open_time_str).do(self._market_open_handler)
        schedule.every().friday.at(open_time_str).do(self._market_open_handler)

        # Market close - Monday to Friday at converted local time
        schedule.every().monday.at(close_time_str).do(self._market_close_handler)
        schedule.every().tuesday.at(close_time_str).do(self._market_close_handler)
        schedule.every().wednesday.at(close_time_str).do(self._market_close_handler)
        schedule.every().thursday.at(close_time_str).do(self._market_close_handler)
        schedule.every().friday.at(close_time_str).do(self._market_close_handler)

        self._log("📅 Market schedule configured with timezone conversion")

    def _check_initial_market_status(self):
        """Check if market is currently open and start processes if needed."""
        et_now = datetime.now(self.et_tz)
        current_time = et_now.time()
        is_weekday = et_now.weekday() < 5  # Monday = 0, Friday = 4

        if (is_weekday and
            self.market_open_time <= current_time <= self.market_close_time):
            self._log("📈 Market is currently open, starting alert processes...")
            self._market_open_handler()
        else:
            self._log("📉 Market is currently closed")

    def _market_open_handler(self):
        """Handle market open - start all alert processes."""
        et_now = datetime.now(self.et_tz)
        if et_now.weekday() >= 5:  # Skip weekends
            return

        self._log("🔔 MARKET OPEN - Starting alert processes")
        self.market_open = True

        for process_name, config in self.alert_processes.items():
            self._start_alert_process(process_name, config)

        self._log("✅ All alert processes started for market session")

    def _market_close_handler(self):
        """Handle market close - stop processes, run summaries, and shutdown."""
        et_now = datetime.now(self.et_tz)
        if et_now.weekday() >= 5:  # Skip weekends
            return

        self._log("🔔 MARKET CLOSE - Stopping alert processes")
        self.market_open = False

        # Stop all alert processes
        self._stop_all_processes()

        # Wait a moment for processes to fully stop
        time.sleep(5)

        # Run post-market analysis
        self._run_post_market_analysis()

        # Shutdown after all EOD chores are complete
        self._shutdown_after_eod()

    def _start_alert_process(self, process_name: str, config: Dict):
        """Start a specific alert process with dedicated logging."""
        try:
            if process_name in self.processes:
                # Process already running
                if self._is_process_running(process_name):
                    self._log(f"⚠️ Process {process_name} already running")
                    return
                else:
                    # Clean up dead process
                    del self.processes[process_name]

            # Build command
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')
            if config['python_cmd'] == 'python3':
                python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            script_path = self.project_root / config['script']
            cmd = [python_path, str(script_path)] + config['args']

            # Setup process-specific log file
            et_now = datetime.now(self.et_tz)
            timestamp = et_now.strftime("%Y%m%d_%H%M%S")
            process_log_dir = self.logs_dir / config['log_dir']
            process_log_file = process_log_dir / f"{process_name}_{timestamp}.log"

            self._log(f"🚀 Starting {process_name}: {' '.join(cmd)}")
            self._log(f"📝 Process logs: {process_log_file}")

            # Start process with output redirected to log file
            with open(process_log_file, 'w') as log_file:
                # Write header to log file
                et_now = datetime.now(self.et_tz)
                log_file.write(f"# {process_name} Log - Started: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
                log_file.write(f"# Command: {' '.join(cmd)}\n")
                log_file.write(f"# Working Directory: {self.project_root}\n")
                log_file.write("# " + "="*50 + "\n\n")
                log_file.flush()

                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

            # Store process info including log file path
            self.processes[process_name] = {
                'process': process,
                'log_file': process_log_file,
                'config': config
            }

            # Give it a moment to start
            time.sleep(2)

            if self._is_process_running(process_name):
                self._log(f"✅ {process_name} started (PID: {process.pid})")
                # Start log monitoring thread
                self._start_log_monitor(process_name)
            else:
                self._log(f"❌ Failed to start {process_name}", "ERROR")

        except Exception as e:
            self._log(f"❌ Error starting {process_name}: {e}", "ERROR")

    def _stop_all_processes(self):
        """Stop all running alert processes."""
        for process_name in list(self.processes.keys()):
            self._stop_alert_process(process_name)

    def _stop_alert_process(self, process_name: str):
        """Stop a specific alert process."""
        if process_name not in self.processes:
            return

        try:
            process_info = self.processes[process_name]
            if isinstance(process_info, dict):
                process = process_info['process']
                log_file = process_info['log_file']
            else:
                # Legacy format
                process = process_info
                log_file = None

            self._log(f"🛑 Stopping {process_name} (PID: {process.pid})")

            # Try graceful shutdown
            process.terminate()

            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                self._log(f"✅ {process_name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self._log(f"⚠️ Forcibly killing {process_name}", "WARN")
                process.kill()
                process.wait()

            # Add shutdown timestamp to log file
            if log_file and log_file.exists():
                try:
                    with open(log_file, 'a') as f:
                        et_now = datetime.now(self.et_tz)
                        f.write(f"\n# Process stopped: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
                except Exception:
                    pass

        except Exception as e:
            self._log(f"❌ Error stopping {process_name}: {e}", "ERROR")
        finally:
            if process_name in self.processes:
                del self.processes[process_name]

    def _is_process_running(self, process_name: str) -> bool:
        """Check if a specific process is still running."""
        if process_name not in self.processes:
            return False

        try:
            process_info = self.processes[process_name]
            if isinstance(process_info, dict):
                return process_info['process'].poll() is None
            else:
                # Legacy format for backward compatibility
                return process_info.poll() is None
        except Exception:
            return False

    def _monitor_alert_processes(self):
        """Monitor all alert processes and restart if needed."""
        for process_name, config in self.alert_processes.items():
            if not self._is_process_running(process_name):
                if process_name in self.processes:
                    self._log(f"⚠️ Process {process_name} died, restarting...", "WARN")
                    del self.processes[process_name]

                # Restart the process
                self._start_alert_process(process_name, config)

    def _start_log_monitor(self, process_name: str):
        """Start a thread to monitor process logs and report critical errors."""
        def monitor_logs():
            try:
                process_info = self.processes.get(process_name)
                if not process_info or not isinstance(process_info, dict):
                    return

                log_file = process_info['log_file']
                if not log_file.exists():
                    time.sleep(1)  # Wait for log file to be created

                # Monitor the log file for critical errors
                # Note: Avoid 'CRITICAL' alone as superduper alerts use "Urgency: CRITICAL" in normal output
                error_keywords = ['ERROR', 'FATAL', 'Exception', 'Traceback', '- CRITICAL -', 'CRITICAL:']

                with open(log_file, 'r') as f:
                    f.seek(0, 2)  # Go to end of file

                    while self._is_process_running(process_name):
                        line = f.readline()
                        if line:
                            # Check for critical errors
                            if any(keyword in line for keyword in error_keywords):
                                self._log(f"⚠️ {process_name} error: {line.strip()}", "WARN")
                        else:
                            time.sleep(1)  # No new content, wait

            except Exception as e:
                self._log(f"❌ Error monitoring logs for {process_name}: {e}", "ERROR")

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_logs, daemon=True)
        monitor_thread.start()

    def _run_post_market_analysis(self):
        """Run post-market analysis scripts and send summary."""
        self._log("📊 Starting post-market analysis...")

        analysis_results = {}

        # Run each post-market script
        for script_config in self.post_market_scripts:
            script_name = Path(script_config['script']).stem
            self._log(f"🔧 Running {script_name}...")

            try:
                result = self._execute_script(script_config)
                analysis_results[script_name] = result

                if result['success']:
                    self._log(f"✅ {script_name} completed successfully")
                else:
                    self._log(f"❌ {script_name} failed: {result['error']}", "ERROR")

            except Exception as e:
                self._log(f"❌ Error running {script_name}: {e}", "ERROR")
                analysis_results[script_name] = {
                    'success': False,
                    'error': str(e),
                    'output': ''
                }

        # Generate PNL reports for all trading accounts
        self._log("💰 Generating PNL reports for all trading accounts...")
        try:
            pnl_results = self._generate_pnl_reports()
            if pnl_results:
                analysis_results['pnl_reports'] = {
                    'success': True,
                    'output': f"Generated PNL reports for {len(pnl_results)} accounts",
                    'pnl_data': pnl_results
                }
                self._log(f"✅ PNL reports completed for {len(pnl_results)} accounts")
            else:
                analysis_results['pnl_reports'] = {
                    'success': False,
                    'error': 'No trading accounts found or all PNL reports failed',
                    'output': ''
                }
                self._log("⚠️ No PNL reports generated", "WARN")
        except Exception as e:
            self._log(f"❌ Error generating PNL reports: {e}", "ERROR")
            analysis_results['pnl_reports'] = {
                'success': False,
                'error': str(e),
                'output': ''
            }

        # Send summary to Bruce
        self._send_daily_summary(analysis_results)

    def _execute_script(self, script_config: Dict) -> Dict:
        """Execute a post-market script and return results with logging."""
        try:
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')
            if script_config['python_cmd'] == 'python3':
                python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            script_path = self.project_root / script_config['script']
            cmd = [python_path, str(script_path)] + script_config['args']

            # Set environment variable for ORB script auto-selection
            env = os.environ.copy()
            if 'orb.py' in script_config['script']:
                env['AUTO_SELECT_DEFAULT'] = '1'

            # Setup script-specific log file
            et_now = datetime.now(self.et_tz)
            timestamp = et_now.strftime("%Y%m%d_%H%M%S")
            script_log_dir = self.logs_dir / script_config['log_dir']
            script_name = Path(script_config['script']).stem
            script_log_file = script_log_dir / f"{script_name}_{timestamp}.log"

            self._log(f"📝 Script logs: {script_log_file}")

            # Execute with output saved to log file
            with open(script_log_file, 'w') as log_file:
                # Write header
                et_now = datetime.now(self.et_tz)
                log_file.write(f"# {script_name} Post-Market Analysis Log\n")
                log_file.write(f"# Started: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
                log_file.write(f"# Command: {' '.join(cmd)}\n")
                log_file.write(f"# Working Directory: {self.project_root}\n")
                log_file.write("# " + "="*50 + "\n\n")
                log_file.flush()

                result = subprocess.run(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    env=env
                )

                # Write completion info
                et_now = datetime.now(self.et_tz)
                log_file.write(f"\n\n# Completed: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
                log_file.write(f"# Return code: {result.returncode}\n")

            # Read the log file to get output for summary
            try:
                with open(script_log_file, 'r') as f:
                    log_content = f.read()
                    # Extract just the actual output (skip headers)
                    output_lines = []
                    in_output = False
                    for line in log_content.split('\n'):
                        if line.startswith('# ========'):
                            in_output = True
                            continue
                        elif in_output and not line.startswith('# Completed:') and not line.startswith('# Return code:'):
                            output_lines.append(line)
                    output = '\n'.join(output_lines).strip()
            except Exception:
                output = "Output saved to log file"

            return {
                'success': result.returncode == 0,
                'output': output,
                'error': None if result.returncode == 0 else f"Script failed with return code {result.returncode}",
                'returncode': result.returncode,
                'log_file': str(script_log_file)
            }

        except subprocess.TimeoutExpired:
            # Write timeout info to log file if it exists
            try:
                with open(script_log_file, 'a') as f:
                    et_now = datetime.now(self.et_tz)
                    f.write(f"\n\n# TIMEOUT: Script timed out after 5 minutes - {et_now.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
            except Exception:
                pass

            return {
                'success': False,
                'output': '',
                'error': 'Script timed out after 5 minutes',
                'returncode': -1,
                'log_file': str(script_log_file) if 'script_log_file' in locals() else None
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'returncode': -1,
                'log_file': None
            }

    def _send_daily_summary(self, analysis_results: Dict):
        """Send daily summary to Bruce via Telegram."""
        try:
            # Get today's date in ET
            et_now = datetime.now(self.et_tz)
            today = et_now.strftime("%Y-%m-%d")

            # Build summary message
            summary_lines = [
                f"📊 **Daily Trading Summary - {today}**",
                "",
                "🎯 **Alert Processes Status:**"
            ]

            # Add process runtime info
            for process_name in self.alert_processes.keys():
                summary_lines.append(f"✅ {process_name.replace('_', ' ').title()}: Completed")

            summary_lines.extend([
                "",
                "📈 **Post-Market Analysis:**"
            ])

            # Add analysis results
            for script_name, result in analysis_results.items():
                if script_name == 'pnl_reports':
                    # Special handling for PNL reports
                    status = "✅" if result['success'] else "❌"
                    summary_lines.append(f"{status} PNL Reports")

                    if result['success'] and result.get('pnl_data'):
                        pnl_data = result['pnl_data']
                        successful_count = sum(1 for pnl_result in pnl_data.values() if pnl_result['success'])
                        total_count = len(pnl_data)
                        summary_lines.append(f"   Accounts: {successful_count}/{total_count} successful")

                        # Show account names that were processed
                        account_names = list(pnl_data.keys())
                        if account_names:
                            summary_lines.append(f"   Processed: {', '.join(account_names)}")

                    if not result['success'] and result.get('error'):
                        summary_lines.append(f"   Error: {result['error']}")
                else:
                    # Normal script handling
                    status = "✅" if result['success'] else "❌"
                    summary_lines.append(f"{status} {script_name.replace('_', ' ').title()}")

                    if result.get('log_file'):
                        summary_lines.append(f"   Log: {result['log_file']}")

                    if not result['success'] and result['error']:
                        summary_lines.append(f"   Error: {result['error']}")

            # Add log directory information
            summary_lines.extend([
                "",
                "🔍 **Key Highlights:**",
                "• Superduper alerts monitoring completed",
                "• Trade execution monitoring completed",
                "• VWAP bounce alerts monitoring completed",
                "• ORB analysis charts generated",
                "• Alert summary processed",
                "• PNL reports generated and sent individually",
                "",
                "📋 **Log Files Available:**",
                f"• Watchdog logs: logs/alerts_watchdog/",
                f"• ORB Monitor: logs/orb_monitor/",
                f"• Superduper Alerts: logs/orb_superduper/",
                f"• Trade Execution: logs/orb_trades/",
                f"• VWAP Bounce Alerts: logs/vwap_bounce_alerts/",
                f"• Post-Market Analysis: logs/orb_alerts_summary/ & logs/orb_analysis/",
                f"• PNL Reports: logs/pnl_reports/"
            ])

            summary_message = "\n".join(summary_lines)

            # Send to Bruce (we'll try to find his chat ID)
            self._log("📤 Sending daily summary to Bruce...")

            # For now, send to all active users since we don't have Bruce's specific chat ID
            try:
                from atoms.telegram.user_manager import UserManager
                user_manager = UserManager()
                active_users = user_manager.get_active_users()

                # Look for Bruce specifically
                bruce_users = [u for u in active_users if 'bruce' in u.get('username', '').lower()]

                if bruce_users:
                    for user in bruce_users:
                        from atoms.telegram.telegram_post import TelegramPoster
                        telegram_poster = TelegramPoster()
                        telegram_poster.send_message_to_user(summary_message, user['username'])
                        self._log(f"✅ Summary sent to Bruce ({user['chat_id']})")
                else:
                    # Fallback: send to first active user (assuming it's Bruce)
                    if active_users:
                        send_message(active_users[0]['chat_id'], summary_message)
                        self._log(f"✅ Summary sent to first active user ({active_users[0]['chat_id']})")
                    else:
                        self._log("⚠️ No active Telegram users found", "WARN")

            except Exception as e:
                self._log(f"❌ Failed to send Telegram summary: {e}", "ERROR")

        except Exception as e:
            self._log(f"❌ Error creating daily summary: {e}", "ERROR")

    def _shutdown_after_eod(self):
        """Shutdown the watchdog after all end-of-day chores are completed."""
        try:
            self._log("🏁 END OF DAY - All tasks completed")
            self._log("📊 Daily summary sent")
            self._log("🔄 All processes stopped")
            self._log("📈 Post-market analysis finished")
            self._log("")
            self._log("🌙 SHUTTING DOWN WATCHDOG - EOD tasks complete")
            self._log("💡 Restart tomorrow for next trading session")

            # Send shutdown notification to Bruce
            try:
                et_now = datetime.now(self.et_tz)
                today = et_now.strftime("%Y-%m-%d")
                shutdown_message = f"""🌙 **EOD Shutdown Complete - {today}**

✅ **All Tasks Completed:**
• Alert processes stopped
• Post-market analysis finished
• Daily summary sent
• Logs saved and organized

📋 **Log Directory:** logs/alerts_watchdog/
⏰ **Shutdown Time:** {et_now.strftime('%H:%M:%S ET')}"""

                from atoms.telegram.user_manager import UserManager
                user_manager = UserManager()
                active_users = user_manager.get_active_users()
                bruce_users = [u for u in active_users if 'bruce' in u.get('username', '').lower()]

                if bruce_users:
                    for user in bruce_users:
                        from atoms.telegram.telegram_post import TelegramPoster
                        telegram_poster = TelegramPoster()
                        telegram_poster.send_message_to_user(shutdown_message, user['username'])
                        self._log(f"✅ Shutdown notification sent to Bruce ({user['chat_id']})")

            except Exception as e:
                self._log(f"⚠️ Failed to send shutdown notification: {e}", "WARN")

            # Wait a moment for any final operations
            time.sleep(2)

            # Set running flag to False to exit main loop
            self.running = False
            self._log("👋 Watchdog shutdown initiated - goodbye!")

        except Exception as e:
            self._log(f"❌ Error during EOD shutdown: {e}", "ERROR")
            self.running = False

def main():
    """Main entry point."""
    print("📊 ALERTS WATCHDOG - Market Hours Alert System")
    print("=" * 60)

    try:
        watchdog = AlertsWatchdog()
        watchdog.start_watchdog()
    except KeyboardInterrupt:
        print("\n👋 Alerts watchdog stopped by user")
    except Exception as e:
        print(f"❌ Watchdog error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())