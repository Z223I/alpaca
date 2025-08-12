#!/usr/bin/env python3
"""
Telegram Polling Watchdog Service

This watchdog ensures that molecules/telegram_polling.py is always running.
It monitors the process, starts it if it's not running, and restarts it
if errors are detected in the logs.

Key Features:
- Monitors telegram_polling.py process
- Automatically starts/restarts the service
- Detects error patterns in logs and triggers restarts
- Creates timestamped log files in logs/ directory
- 30-second restart delay after errors
"""

import sys
import os
import time
import signal
import subprocess
import threading
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List

class TelegramPollingWatchdog:
    """Watchdog service for monitoring and managing telegram_polling.py"""
    
    def __init__(self):
        # Get project root directory
        self.project_root = Path(__file__).parent.parent
        self.polling_script = self.project_root / "molecules" / "telegram_polling.py"
        self.logs_dir = self.project_root / "logs"
        
        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"telegram_polling_{timestamp}.log"
        
        # Process monitoring
        self.polling_process: Optional[subprocess.Popen] = None
        self.running = False
        self.restart_delay = 30  # seconds
        self.check_interval = 5  # seconds
        self.log_monitor_thread: Optional[threading.Thread] = None
        
        # Error patterns to watch for
        self.error_patterns = [
            "ERROR: Error polling updates:",
            "ERROR:",  # General error pattern
            "CRITICAL:",
            "❌ Polling error:",
            "❌ Failed to start polling service:"
        ]
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._log("🔧 Telegram Polling Watchdog initialized")
        self._log(f"📁 Project root: {self.project_root}")
        self._log(f"📄 Polling script: {self.polling_script}")
        self._log(f"📋 Log file: {self.log_file}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._log(f"🛑 Received signal {signum}, shutting down watchdog...")
        self.running = False
        if self.polling_process:
            self._stop_polling_service()
    
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
        """Start the watchdog service."""
        try:
            self._log("🚀 Starting Telegram Polling Watchdog")
            self._log(f"⏰ Check interval: {self.check_interval} seconds")
            self._log(f"⏳ Restart delay on error: {self.restart_delay} seconds")
            self._log("💡 Send CTRL+C to stop watchdog")
            
            self.running = True
            
            # Start the monitoring loop
            while self.running:
                try:
                    self._check_and_manage_polling_service()
                    time.sleep(self.check_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self._log(f"❌ Watchdog error: {e}", "ERROR")
                    time.sleep(self.check_interval * 2)  # Wait longer on error
            
            self._log("🛑 Watchdog service stopped")
            
        except Exception as e:
            self._log(f"❌ Failed to start watchdog: {e}", "ERROR")
            return False
        
        return True
    
    def _check_and_manage_polling_service(self):
        """Check if polling service is running and manage it."""
        is_running = self._is_polling_service_running()
        
        if not is_running:
            if self.polling_process:
                self._log("⚠️ Polling service process died, restarting...", "WARN")
            else:
                self._log("🚀 Starting polling service...")
            
            self._start_polling_service()
        else:
            # Service is running, check if we need to monitor its logs
            if not self.log_monitor_thread or not self.log_monitor_thread.is_alive():
                self._start_log_monitoring()
    
    def _is_polling_service_running(self) -> bool:
        """Check if the polling service process is still running."""
        if not self.polling_process:
            return False
        
        try:
            # Check if process is still alive
            return self.polling_process.poll() is None
        except Exception:
            return False
    
    def _start_polling_service(self):
        """Start the telegram polling service."""
        try:
            # Stop existing process if any
            if self.polling_process:
                self._stop_polling_service()
            
            # Use conda environment python
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')
            
            # Start the polling service
            cmd = [python_path, str(self.polling_script)]
            
            self._log(f"🔧 Executing: {' '.join(cmd)}")
            
            self.polling_process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            if self._is_polling_service_running():
                self._log(f"✅ Polling service started (PID: {self.polling_process.pid})")
                self._start_log_monitoring()
            else:
                self._log("❌ Failed to start polling service", "ERROR")
                self.polling_process = None
                
        except Exception as e:
            self._log(f"❌ Error starting polling service: {e}", "ERROR")
            self.polling_process = None
    
    def _stop_polling_service(self):
        """Stop the polling service gracefully."""
        if not self.polling_process:
            return
        
        try:
            self._log(f"🛑 Stopping polling service (PID: {self.polling_process.pid})")
            
            # Try graceful shutdown first
            self.polling_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.polling_process.wait(timeout=10)
                self._log("✅ Polling service stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self._log("⚠️ Forcibly killing polling service", "WARN")
                self.polling_process.kill()
                self.polling_process.wait()
                
        except Exception as e:
            self._log(f"❌ Error stopping polling service: {e}", "ERROR")
        finally:
            self.polling_process = None
    
    def _start_log_monitoring(self):
        """Start monitoring the polling service logs for errors."""
        if self.log_monitor_thread and self.log_monitor_thread.is_alive():
            return
        
        self.log_monitor_thread = threading.Thread(
            target=self._monitor_polling_logs,
            daemon=True
        )
        self.log_monitor_thread.start()
        self._log("👁️ Started log monitoring thread")
    
    def _monitor_polling_logs(self):
        """Monitor polling service output for error patterns."""
        if not self.polling_process or not self.polling_process.stdout:
            return
        
        try:
            while self.running and self._is_polling_service_running():
                line = self.polling_process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                if line:
                    # Log the output from polling service
                    self._log(f"📤 POLLING: {line}")
                    
                    # Check for error patterns
                    for pattern in self.error_patterns:
                        if pattern in line:
                            self._log(f"🚨 Error detected: {pattern}", "ERROR")
                            self._handle_polling_error(line)
                            return
                            
        except Exception as e:
            self._log(f"❌ Error monitoring logs: {e}", "ERROR")
    
    def _handle_polling_error(self, error_line: str):
        """Handle detected error in polling service."""
        self._log(f"🚨 Handling polling error: {error_line}", "ERROR")
        
        # Stop the current service
        self._stop_polling_service()
        
        # Wait before restarting
        self._log(f"⏳ Waiting {self.restart_delay} seconds before restart...")
        for i in range(self.restart_delay, 0, -5):
            if not self.running:
                return
            self._log(f"⏳ Restarting in {i} seconds...")
            time.sleep(min(5, i))
        
        if self.running:
            self._log("🔄 Restarting polling service after error...")
            self._start_polling_service()
    
    def get_status(self) -> dict:
        """Get current status of watchdog and polling service."""
        return {
            'watchdog_running': self.running,
            'polling_service_running': self._is_polling_service_running(),
            'polling_process_pid': self.polling_process.pid if self.polling_process else None,
            'log_file': str(self.log_file),
            'restart_delay': self.restart_delay,
            'check_interval': self.check_interval
        }

def main():
    """Main entry point."""
    print("🐕 TELEGRAM POLLING WATCHDOG")
    print("=" * 50)
    
    try:
        watchdog = TelegramPollingWatchdog()
        watchdog.start_watchdog()
    except KeyboardInterrupt:
        print("\n👋 Watchdog stopped by user")
    except Exception as e:
        print(f"❌ Watchdog error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())