#!/usr/bin/env python3
"""
Alerts Watch - Automated Market Hours Alert System

This system manages ORB alert processes during market hours (9:30 AM - 4:00 PM ET, Mon-Fri):
1. Starts 3 alert processes at market open
2. Monitors and restarts them if they fail
3. Stops them at market close
4. Runs post-market analysis and summary
5. Sends daily summary to Bruce via Telegram

Managed Processes:
- ORB Alerts Monitor (Super alerts)
- ORB Superduper Alerts
- ORB Trade Execution

Post-Market Tasks:
- ORB Alerts Summary
- ORB Analysis with Charts
- Telegram summary to Bruce
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
        """Handle market close - stop processes and run summaries."""
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
                error_keywords = ['ERROR', 'CRITICAL', 'FATAL', 'Exception', 'Traceback']
                
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
                    timeout=300  # 5 minute timeout
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
                "• ORB analysis charts generated",
                "• Alert summary processed",
                "",
                "📋 **Log Files Available:**",
                f"• Watchdog logs: logs/alerts_watchdog/",
                f"• ORB Monitor: logs/orb_monitor/",
                f"• Superduper Alerts: logs/orb_superduper/",
                f"• Trade Execution: logs/orb_trades/",
                f"• Post-Market Analysis: logs/orb_alerts_summary/ & logs/orb_analysis/"
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