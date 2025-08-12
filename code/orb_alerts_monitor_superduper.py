"""
ORB Superduper Alerts Monitor - Advanced Alert Generation System

This system monitors the super_alerts directory for bullish super alerts and creates 
superduper alerts when advanced trend analysis criteria are met. It analyzes price 
movement patterns, momentum, and consolidation to identify the highest quality alerts.

Usage:
    python3 code/orb_alerts_monitor_superduper.py                           # Monitor current date super alerts
    python3 code/orb_alerts_monitor_superduper.py --date 2025-08-01         # Monitor specific date super alerts
    python3 code/orb_alerts_monitor_superduper.py --timeframe 60            # Use 60-minute analysis window
    python3 code/orb_alerts_monitor_superduper.py --test                    # Run in test mode
    python3 code/orb_alerts_monitor_superduper.py --no-telegram             # Disable telegram notifications
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
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from atoms.alerts.superduper_alert_filter import SuperduperAlertFilter
from atoms.alerts.superduper_alert_generator import SuperduperAlertGenerator
from atoms.alerts.config import get_momentum_thresholds
from atoms.telegram.orb_alerts import send_orb_alert


class SuperAlertFileHandler(FileSystemEventHandler):
    """Handles new super alert files in the bullish super alerts directory."""

    def __init__(self, monitor, loop):
        self.monitor = monitor
        self.loop = loop

    def on_created(self, event):
        """Called when a new super alert file is created."""
        if not event.is_directory and event.src_path.endswith('.json'):
            # Schedule the coroutine in the main event loop from this thread
            asyncio.run_coroutine_threadsafe(
                self.monitor._process_new_super_alert_file(event.src_path), 
                self.loop
            )


class ORBSuperduperAlertMonitor:
    """Main ORB Superduper Alert Monitor that watches for super alerts and creates superduper alerts."""

    def __init__(self, timeframe_minutes: int = 30, test_mode: bool = False, post_only_urgent: bool = False, no_telegram: bool = False, date: Optional[str] = None):
        """
        Initialize ORB Superduper Alert Monitor.

        Args:
            timeframe_minutes: Time window for trend analysis (default 30)
            test_mode: Run in test mode (no actual alerts)
            post_only_urgent: Only send urgent telegram alerts
            no_telegram: Disable telegram notifications
            date: Date in YYYY-MM-DD format (default: current date)
        """
        # Setup logging
        self.logger = self._setup_logging()

        # Initialize momentum configuration
        self.momentum_thresholds = get_momentum_thresholds()

        # Initialize filtering and generation atoms
        self.superduper_alert_filter = SuperduperAlertFilter(timeframe_minutes)
        self.superduper_alert_generator = None  # Will be initialized when directories are set up
        self.test_mode = test_mode
        self.post_only_urgent = post_only_urgent
        self.no_telegram = no_telegram
        self.timeframe_minutes = timeframe_minutes

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
        self.super_alerts_dir = Path(f"historical_data/{target_date}/super_alerts/bullish")
        self.superduper_alerts_dir = Path(f"historical_data/{target_date}/superduper_alerts/bullish")

        # Ensure directories exist
        self.super_alerts_dir.mkdir(parents=True, exist_ok=True)
        self.superduper_alerts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize superduper alert generator now that directory is set up
        self.superduper_alert_generator = SuperduperAlertGenerator(self.superduper_alerts_dir, test_mode)

        # File system watcher
        self.observer = Observer()
        self.file_handler = None  # Will be set when event loop is available

        # Processed alerts tracking
        self.processed_super_alerts = set()
        self.filtered_super_alerts = set()  # Track filtered super alerts
        self.created_superduper_alerts = set()  # Track created superduper alerts

        self.logger.info(f"ORB Superduper Alert Monitor initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Target date: {target_date}")
        self.logger.info(f"Monitoring super alerts in: {self.super_alerts_dir}")
        self.logger.info(f"Superduper alerts will be saved to: {self.superduper_alerts_dir}")
        self.logger.info(f"Trend analysis timeframe: {timeframe_minutes} minutes")
        if no_telegram:
            self.logger.info("📵 Telegram notifications disabled")
        else:
            self.logger.info("📱 Telegram notifications enabled for superduper alerts")

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
            handler = logging.StreamHandler()
            formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    async def _process_new_super_alert_file(self, file_path: str) -> None:
        """Process a new super alert file."""
        try:
            # Avoid processing the same file multiple times
            if file_path in self.processed_super_alerts:
                return

            self.processed_super_alerts.add(file_path)

            # Wait a moment for file to be fully written
            await asyncio.sleep(0.1)

            with open(file_path, 'r') as f:
                super_alert_data = json.load(f)

            symbol = super_alert_data.get('symbol', 'UNKNOWN')

            # Use SuperduperAlertFilter to determine if we should create a superduper alert
            should_create, filter_reason, analysis_data = self.superduper_alert_filter.should_create_superduper_alert(file_path)

            if not should_create:
                # Log filtered super alerts
                self.filtered_super_alerts.add(file_path)
                if filter_reason.startswith("Insufficient"):
                    self.logger.debug(f"Skipping superduper alert for {symbol}: {filter_reason}")
                else:
                    self.logger.info(f"🚫 Filtered superduper alert for {symbol}: {filter_reason}")
                return

            # Extract trend analysis results
            trend_type = None
            trend_strength = 0.0

            if analysis_data:
                # Re-run trend analysis to get the classification
                symbol_data = self.superduper_alert_filter.symbol_data.get(symbol)
                if symbol_data:
                    trend_type, trend_strength, _ = symbol_data.analyze_trend(self.timeframe_minutes)

            if not trend_type or trend_type in ['insufficient_data', 'declining']:
                self.logger.warning(f"⚠️ Invalid trend analysis for {symbol}: {trend_type}")
                return

            # Create and save superduper alert
            filename = self.superduper_alert_generator.create_and_save_superduper_alert(
                super_alert_data, analysis_data, trend_type, trend_strength
            )

            if filename:
                self.created_superduper_alerts.add(filename)
                self.logger.info(f"✅ Superduper alert created and saved: {filename}")

                # Send Telegram notification (if enabled)
                if not self.no_telegram:
                    try:
                        # Determine urgency based on momentum color thresholds
                        urgency_level = self._determine_urgency(trend_type, trend_strength, analysis_data)
                        momentum = abs(analysis_data.get('price_momentum', 0))

                        self.logger.info(f"📊 Superduper analysis: {trend_type.upper()} trend, strength {trend_strength:.2f}, momentum {momentum:.4f} ({urgency_level.upper()})")

                        # Filter out red and yellow momentum alerts
                        if urgency_level == 'filtered':
                            green_threshold = self.momentum_thresholds.green_threshold
                            self.logger.info(f"🚫 Telegram superduper alert filtered (momentum < {green_threshold}): {symbol}")
                        else:
                            # Only green momentum alerts are sent (urgent)
                            is_urgent = (urgency_level == 'urgent')

                            file_path = self.superduper_alerts_dir / filename
                            result = send_orb_alert(str(file_path), urgent=is_urgent, post_only_urgent=self.post_only_urgent)

                            urgency_type = urgency_level
                            if result['success']:
                                if result.get('skipped'):
                                    self.logger.info(f"⏭️ Telegram superduper alert skipped ({urgency_type}): {result.get('reason', 'Non-urgent filtered')}")
                                else:
                                    emoji = "🟢" if is_urgent else "🟡"
                                    self.logger.info(f"📤 {emoji} Telegram superduper alert sent ({urgency_type}): {result['sent_count']} users notified")
                            else:
                                self.logger.warning(f"❌ Telegram superduper alert failed ({urgency_type}): {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        self.logger.error(f"❌ Error sending Telegram superduper alert: {e}")
                else:
                    momentum = abs(analysis_data.get('price_momentum', 0))
                    urgency_level = self._determine_urgency(trend_type, trend_strength, analysis_data)
                    self.logger.info(f"📊 Superduper analysis: {trend_type.upper()} trend, strength {trend_strength:.2f}, momentum {momentum:.4f} ({urgency_level.upper()}) (Telegram disabled)")
            else:
                self.logger.warning(f"⚠️ Failed to create superduper alert for {symbol} - no Telegram notification sent")

        except Exception as e:
            self.logger.error(f"Error processing super alert file {file_path}: {e}")

    def _determine_urgency(self, trend_type: str, trend_strength: float, analysis_data: Dict) -> str:
        """
        Determine alert priority based on centralized momentum thresholds.

        Returns:
            'filtered' - Red and yellow momentum, don't send to Telegram
            'urgent' - Green momentum, urgent Telegram notification
        """
        momentum = abs(analysis_data.get('price_momentum', 0))
        return self.momentum_thresholds.get_urgency_level(momentum)

    async def _scan_existing_super_alerts(self) -> None:
        """Scan existing super alert files on startup."""
        try:
            if not self.super_alerts_dir.exists():
                self.logger.info("No existing super alerts directory found")
                return

            super_alert_files = list(self.super_alerts_dir.glob("super_alert_*.json"))
            self.logger.info(f"Scanning {len(super_alert_files)} existing super alert files...")

            processed_count = 0
            for super_alert_file in super_alert_files:
                await self._process_new_super_alert_file(str(super_alert_file))
                processed_count += 1

            self.logger.info(f"Processed {processed_count} existing super alerts")

        except Exception as e:
            self.logger.error(f"Error scanning existing super alerts: {e}")

    async def start(self) -> None:
        """Start the ORB Superduper Alert Monitor."""
        self.logger.info("Starting ORB Superduper Alert Monitor...")

        try:
            # Initialize file handler with current event loop
            current_loop = asyncio.get_running_loop()
            self.file_handler = SuperAlertFileHandler(self, current_loop)

            # Start file system monitoring (real-time alerts only)
            if self.super_alerts_dir.exists():
                self.observer.schedule(self.file_handler, str(self.super_alerts_dir), recursive=False)
                self.observer.start()
                self.logger.info(f"Started monitoring {self.super_alerts_dir}")
            else:
                self.logger.warning(f"Super alerts directory does not exist: {self.super_alerts_dir}")

            # Print status
            print("\n" + "="*80)
            print("🎯 ORB SUPERDUPER ALERTS MONITOR ACTIVE")
            print(f"📅 Target date: {self.target_date}")
            print(f"📁 Monitoring: {self.super_alerts_dir}")
            print(f"💾 Superduper alerts: {self.superduper_alerts_dir}")
            print(f"⏱️ Analysis timeframe: {self.timeframe_minutes} minutes")
            print("✅ Filtering: Rising trends & high-quality consolidation patterns")
            if self.no_telegram:
                print("📵 Telegram: Notifications disabled")
            else:
                print("📱 Telegram: Enhanced notifications for superduper alerts")
            if self.test_mode:
                print("🧪 TEST MODE: Superduper alerts will be marked as [TEST MODE]")
            if self.post_only_urgent:
                print("⚡ URGENT ONLY: Only urgent superduper alerts will be sent via Telegram")
            print("="*80 + "\n")

            # Keep running
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in superduper alert monitor: {e}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the ORB Superduper Alert Monitor."""
        self.logger.info("Stopping ORB Superduper Alert Monitor...")

        try:
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            self.logger.info("ORB Superduper Alert Monitor stopped")
        except Exception as e:
            self.logger.error(f"Error stopping monitor: {e}")

    def get_statistics(self) -> dict:
        """Get monitoring statistics."""
        return {
            'timeframe_minutes': self.timeframe_minutes,
            'superduper_alerts_generated': len(self.created_superduper_alerts),
            'super_alerts_processed': len(self.processed_super_alerts),
            'super_alerts_filtered': len(self.filtered_super_alerts),
            'monitoring_directory': str(self.super_alerts_dir),
            'superduper_alerts_directory': str(self.superduper_alerts_dir),
            'test_mode': self.test_mode,
            'post_only_urgent': self.post_only_urgent,
            'no_telegram': self.no_telegram
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Superduper Alerts Monitor - Advanced Alert Generation System")

    parser.add_argument(
        "--timeframe",
        type=int,
        default=30,
        help="Time window in minutes for trend analysis (default: 30)"
    )

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
        help="Only send telegram notifications for urgent superduper alerts"
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

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and start monitor
    try:
        monitor = ORBSuperduperAlertMonitor(
            timeframe_minutes=args.timeframe,
            test_mode=args.test,
            post_only_urgent=args.post_only_urgent,
            no_telegram=args.no_telegram,
            date=args.date
        )

        if args.test:
            print("Running in test mode - superduper alerts will be marked as [TEST MODE]")

        if args.post_only_urgent:
            print("Urgent only mode - only urgent superduper alerts will be sent via Telegram")

        if args.no_telegram:
            print("Telegram disabled - no telegram notifications will be sent")

        await monitor.start()

    except Exception as e:
        logging.error(f"Failed to start ORB Superduper Alert Monitor: {e}")
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