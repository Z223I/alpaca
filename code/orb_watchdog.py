#!/usr/bin/env python3
"""
ORB Alerts Watchdog
Monitors and manages the orb_alerts.py process, ensuring it stays running.
Launches orb_alerts on startup and displays its output to the console.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path


class ORBWatchdog:
    def __init__(self):
        self.process = None
        self.should_run = True
        self.script_path = Path(__file__).parent / "orb_alerts.py"
        self.python_path = "python3"
        
        # Use conda python if available (following project instructions)
        conda_python = Path.home() / "miniconda3/envs/alpaca/bin/python"
        if conda_python.exists():
            self.python_path = str(conda_python)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n📡 Watchdog received signal {signum}, shutting down...")
        self.should_run = False
        if self.process:
            self._stop_process()
        sys.exit(0)
    
    def _start_process(self):
        """Start the orb_alerts.py process"""
        try:
            cmd = [self.python_path, str(self.script_path), "--verbose"]
            print(f"🚀 Starting ORB Alerts: {' '.join(cmd)}")
            
            # Start process with output forwarded to console
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            print(f"✅ ORB Alerts started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start ORB Alerts: {e}")
            return False
    
    def _stop_process(self):
        """Stop the orb_alerts.py process"""
        if self.process:
            try:
                print(f"🛑 Stopping ORB Alerts (PID {self.process.pid})")
                self.process.terminate()
                
                # Wait up to 10 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("⚠️  Process didn't terminate gracefully, forcing kill")
                    self.process.kill()
                    self.process.wait()
                
                print("✅ ORB Alerts stopped")
                
            except Exception as e:
                print(f"❌ Error stopping process: {e}")
            finally:
                self.process = None
    
    def _is_process_running(self):
        """Check if the orb_alerts process is still running"""
        if not self.process:
            return False
        
        # Check if process is still alive
        poll_result = self.process.poll()
        return poll_result is None
    
    def _read_output(self):
        """Read and display output from the orb_alerts process"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            # Non-blocking read of available output
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                print(f"[ORB] {line.rstrip()}")
        except Exception as e:
            print(f"⚠️  Error reading process output: {e}")
    
    def run(self):
        """Main watchdog loop"""
        print("🔍 ORB Alerts Watchdog starting...")
        print(f"📁 Monitoring script: {self.script_path}")
        print(f"🐍 Using Python: {self.python_path}")
        
        # Check if orb_alerts.py exists
        if not self.script_path.exists():
            print(f"❌ ORB Alerts script not found: {self.script_path}")
            return 1
        
        # Initial startup
        if not self._start_process():
            return 1
        
        print("👁️  Watchdog monitoring active. Press Ctrl+C to stop.")
        
        try:
            while self.should_run:
                # Read and display any new output
                self._read_output()
                
                # Check if process is still running
                if not self._is_process_running():
                    if self.process:
                        exit_code = self.process.returncode
                        print(f"⚠️  ORB Alerts process died (exit code: {exit_code})")
                    
                    if self.should_run:
                        print("🔄 Restarting ORB Alerts in 5 seconds...")
                        time.sleep(5)
                        
                        if not self._start_process():
                            print("❌ Failed to restart ORB Alerts, waiting 30 seconds...")
                            time.sleep(30)
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n📡 Watchdog interrupted by user")
        
        # Cleanup
        self._stop_process()
        print("🔍 Watchdog stopped")
        return 0


def main():
    """Main entry point"""
    watchdog = ORBWatchdog()
    return watchdog.run()


if __name__ == "__main__":
    sys.exit(main())