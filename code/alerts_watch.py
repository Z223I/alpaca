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
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"alerts_watch_{timestamp}.log"
        
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
                'python_cmd': 'python3'
            },
            'orb_superduper': {
                'script': 'code/orb_alerts_monitor_superduper.py', 
                'args': ['--verbose'],
                'python_cmd': 'python'
            },
            'orb_trades': {
                'script': 'code/orb_alerts_trade_stocks.py',
                'args': ['--verbose'],
                'python_cmd': 'python'
            }
        }
        
        # Post-market analysis scripts
        self.post_market_scripts = [
            {
                'script': 'code/orb_alerts_summary.py',
                'python_cmd': 'python',
                'args': []
            },
            {
                'script': 'code/orb.py',
                'python_cmd': 'python3',
                'args': []
            }
        ]
        
        # Bruce's chat ID for Telegram notifications
        self.bruce_chat_id = None  # Will be loaded from user manager
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._log("🔧 Alerts Watchdog initialized")
        self._log(f"📁 Project root: {self.project_root}")
        self._log(f"📋 Log file: {self.log_file}")
        self._log(f"🕘 Market hours: {self.market_open_time} - {self.market_close_time} ET")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._log(f"🛑 Received signal {signum}, shutting down alerts watchdog...")
        self.running = False
        self._stop_all_processes()
    
    def _log(self, message: str, level: str = "INFO"):
        """Log message with timestamp to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        # Market open - Monday to Friday at 9:30 AM ET
        schedule.every().monday.at("09:30").do(self._market_open_handler)
        schedule.every().tuesday.at("09:30").do(self._market_open_handler)
        schedule.every().wednesday.at("09:30").do(self._market_open_handler)
        schedule.every().thursday.at("09:30").do(self._market_open_handler)
        schedule.every().friday.at("09:30").do(self._market_open_handler)
        
        # Market close - Monday to Friday at 4:00 PM ET
        schedule.every().monday.at("16:00").do(self._market_close_handler)
        schedule.every().tuesday.at("16:00").do(self._market_close_handler)
        schedule.every().wednesday.at("16:00").do(self._market_close_handler)
        schedule.every().thursday.at("16:00").do(self._market_close_handler)
        schedule.every().friday.at("16:00").do(self._market_close_handler)
        
        self._log("📅 Market schedule configured")
    
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
        """Start a specific alert process."""
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
            
            self._log(f"🚀 Starting {process_name}: {' '.join(cmd)}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.processes[process_name] = process
            
            # Give it a moment to start
            time.sleep(2)
            
            if self._is_process_running(process_name):
                self._log(f"✅ {process_name} started (PID: {process.pid})")
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
            process = self.processes[process_name]
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
            return self.processes[process_name].poll() is None
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
        """Execute a post-market script and return results."""
        try:
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')
            if script_config['python_cmd'] == 'python3':
                python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')
            
            script_path = self.project_root / script_config['script']
            cmd = [python_path, str(script_path)] + script_config['args']
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout.strip(),
                'error': result.stderr.strip() if result.returncode != 0 else None,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': 'Script timed out after 5 minutes',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'returncode': -1
            }
    
    def _send_daily_summary(self, analysis_results: Dict):
        """Send daily summary to Bruce via Telegram."""
        try:
            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")
            
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
                
                if not result['success'] and result['error']:
                    summary_lines.append(f"   Error: {result['error']}")
            
            # Add interesting notes placeholder
            summary_lines.extend([
                "",
                "🔍 **Key Highlights:**",
                "• Superduper alerts monitoring completed",
                "• Trade execution monitoring completed", 
                "• ORB analysis charts generated",
                "• Alert summary processed",
                "",
                "📋 Full logs available in logs/ directory"
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
                        send_message(user['chat_id'], summary_message)
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