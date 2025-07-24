"""
ORB Alerts Monitor - Super Alert Generation System

This system monitors the alerts directory for bullish ORB alerts and creates super alerts
when the current price reaches the Signal price from the CSV file. It calculates penetration
into the Signal-to-Resistance range and range percentage.

Usage:
    python3 code/orb_alerts_monitor.py                    # Monitor current date alerts
    python3 code/orb_alerts_monitor.py --symbols-file data/YYYYMMDD.csv  # Use specific symbols file
    python3 code/orb_alerts_monitor.py --test             # Run in test mode
"""

import asyncio
import argparse
import logging
import sys
import json
import csv
import os
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
import pytz
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from atoms.config.alert_config import config
from atoms.config.symbol_manager import SymbolManager
from atoms.config.symbol_data_loader import SymbolDataLoader
from atoms.alerts.super_alert_filter import SuperAlertFilter
from atoms.alerts.super_alert_generator import SuperAlertGenerator
from atoms.telegram.orb_alerts import send_orb_alert

# Alpaca API imports for real-time price checking
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    try:
        from alpaca_trade_api.rest import REST
        ALPACA_AVAILABLE = "legacy"
    except ImportError:
        ALPACA_AVAILABLE = False


# SuperAlertData is now imported from atoms.alerts.super_alert_filter


class AlertFileHandler(FileSystemEventHandler):
    """Handles new alert files in the bullish alerts directory."""
    
    def __init__(self, monitor, loop):
        self.monitor = monitor
        self.loop = loop
        
    def on_created(self, event):
        """Called when a new file is created."""
        if not event.is_directory and event.src_path.endswith('.json'):
            # Schedule the coroutine in the main event loop from this thread
            asyncio.run_coroutine_threadsafe(
                self.monitor._process_new_alert_file(event.src_path), 
                self.loop
            )


class ORBAlertMonitor:
    """Main ORB Alert Monitor that watches for bullish alerts and creates super alerts."""
    
    def __init__(self, symbols_file: Optional[str] = None, test_mode: bool = False):
        """
        Initialize ORB Alert Monitor.
        
        Args:
            symbols_file: Path to symbols CSV file
            test_mode: Run in test mode (no actual alerts)
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load symbol data with Signal and Resistance prices using atom
        self.symbol_loader = SymbolDataLoader(symbols_file)
        self.symbol_data = self.symbol_loader.load_symbol_data()
        
        # Initialize filtering and generation atoms
        self.super_alert_filter = SuperAlertFilter(self.symbol_data)
        self.super_alert_generator = None  # Will be initialized when directories are set up
        self.test_mode = test_mode
        
        # Alert monitoring setup
        et_tz = pytz.timezone('US/Eastern')
        current_date = datetime.now(et_tz).strftime('%Y-%m-%d')
        self.alerts_dir = Path(f"historical_data/{current_date}/alerts/bullish")
        self.super_alerts_dir = Path(f"historical_data/{current_date}/super_alerts/bullish")
        
        # Ensure directories exist
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.super_alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize super alert generator now that directory is set up
        self.super_alert_generator = SuperAlertGenerator(self.super_alerts_dir, test_mode)
        
        # Initialize price client
        self.price_client = None
        if ALPACA_AVAILABLE == True:
            try:
                self.price_client = StockHistoricalDataClient(
                    api_key=config.api_key,
                    secret_key=config.secret_key
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize price client: {e}")
        elif ALPACA_AVAILABLE == "legacy":
            try:
                self.price_client = REST(
                    key_id=config.api_key,
                    secret_key=config.secret_key,
                    base_url=config.base_url
                )
                self.logger.info("Using legacy alpaca-trade-api for price data")
            except Exception as e:
                self.logger.warning(f"Could not initialize legacy price client: {e}")
        
        # File system watcher
        self.observer = Observer()
        self.file_handler = None  # Will be set when event loop is available
        
        # Processed alerts tracking
        self.processed_alerts = set()
        self.filtered_alerts = set()  # Track filtered alerts
        
        self.logger.info(f"ORB Alert Monitor initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Monitoring alerts in: {self.alerts_dir}")
        self.logger.info(f"Super alerts will be saved to: {self.super_alerts_dir}")
        self.logger.info(f"Loaded {len(self.symbol_data)} symbols with Signal/Resistance data")
        self.logger.info("📱 Telegram notifications enabled for super alerts")
    
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
    
    # Symbol data loading is now handled by SymbolDataLoader atom
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if not self.price_client:
            self.logger.warning("Price client not available")
            return None
            
        try:
            if ALPACA_AVAILABLE == "legacy":
                # Use legacy API to get latest quote
                latest_trade = self.price_client.get_latest_trade(symbol)
                if latest_trade:
                    return float(latest_trade.price)
            else:
                # Use new API (not implemented in this example)
                pass
                
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            
        return None
    
    async def _process_new_alert_file(self, file_path: str) -> None:
        """Process a new alert file."""
        try:
            # Avoid processing the same file multiple times
            if file_path in self.processed_alerts:
                return
                
            self.processed_alerts.add(file_path)
            
            # Wait a moment for file to be fully written
            await asyncio.sleep(0.1)
            
            with open(file_path, 'r') as f:
                alert_data = json.load(f)
            
            # Use SuperAlertFilter to determine if we should create a super alert
            should_create, filter_reason = self.super_alert_filter.should_create_super_alert(alert_data)
            
            if not should_create:
                if filter_reason.startswith("Price") or filter_reason.startswith("No signal"):
                    # Log at debug level for price/signal issues
                    self.logger.debug(f"Skipping alert: {filter_reason}")
                else:
                    # Log filtered alerts
                    self.filtered_alerts.add(file_path)
                    self.logger.info(f"🚫 Filtered alert: {filter_reason}")
                return
            
            # Get symbol info and create super alert
            symbol = alert_data.get('symbol')
            symbol_info = self.super_alert_filter.get_symbol_info(symbol)
            
            if symbol_info:
                # Create and save super alert (only after all filters have been applied)
                filename = self.super_alert_generator.create_and_save_super_alert(alert_data, symbol_info)
                if filename:
                    self.logger.info(f"✅ Super alert created and saved: {filename}")
                    
                    # Send Telegram notification only after successful super alert creation
                    try:
                        # Determine urgency based on price breakout percentage
                        current_price = alert_data.get('current_price', 0)
                        orb_high = alert_data.get('orb_high', 0)
                        
                        # Calculate breakout ratio and determine urgency
                        is_urgent = False
                        if orb_high > 0:
                            breakout_ratio = current_price / orb_high
                            is_urgent = breakout_ratio >= 1.20
                            
                            self.logger.info(f"📊 Breakout analysis: {current_price:.2f} / {orb_high:.2f} = {breakout_ratio:.3f} {'(URGENT)' if is_urgent else '(REGULAR)'}")
                        
                        file_path = self.super_alerts_dir / filename
                        result = send_orb_alert(str(file_path), urgent=is_urgent)
                        
                        urgency_type = "urgent" if is_urgent else "regular"
                        if result['success']:
                            self.logger.info(f"📤 Telegram alert sent ({urgency_type}): {result['sent_count']} users notified")
                        else:
                            self.logger.warning(f"❌ Telegram alert failed ({urgency_type}): {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error sending Telegram alert: {e}")
                else:
                    self.logger.warning(f"⚠️ Failed to create super alert for {symbol} - no Telegram notification sent")
            
        except Exception as e:
            self.logger.error(f"Error processing alert file {file_path}: {e}")
    
    # _create_super_alert method is now handled by SuperAlertGenerator atom
    
    async def _scan_existing_alerts(self) -> None:
        """Scan existing alert files on startup."""
        try:
            if not self.alerts_dir.exists():
                self.logger.info("No existing alerts directory found")
                return
                
            alert_files = list(self.alerts_dir.glob("alert_*.json"))
            self.logger.info(f"Scanning {len(alert_files)} existing alert files...")
            
            processed_count = 0
            for alert_file in alert_files:
                await self._process_new_alert_file(str(alert_file))
                processed_count += 1
                
            self.logger.info(f"Processed {processed_count} existing alerts")
            
        except Exception as e:
            self.logger.error(f"Error scanning existing alerts: {e}")
    
    async def start(self) -> None:
        """Start the ORB Alert Monitor."""
        self.logger.info("Starting ORB Alert Monitor...")
        
        try:
            # Initialize file handler with current event loop
            current_loop = asyncio.get_running_loop()
            self.file_handler = AlertFileHandler(self, current_loop)
            
            # Process existing alerts first
            await self._scan_existing_alerts()
            
            # Start file system monitoring
            if self.alerts_dir.exists():
                self.observer.schedule(self.file_handler, str(self.alerts_dir), recursive=False)
                self.observer.start()
                self.logger.info(f"Started monitoring {self.alerts_dir}")
            else:
                self.logger.warning(f"Alerts directory does not exist: {self.alerts_dir}")
            
            # Print status
            print("\n" + "="*80)
            print("🔍 ORB ALERTS MONITOR ACTIVE")
            print(f"📁 Monitoring: {self.alerts_dir}")
            print(f"💾 Super alerts: {self.super_alerts_dir}")
            print(f"📊 Symbols loaded: {len(self.symbol_data)}")
            print("✅ Filtering: Only bullish alerts with candlestick low >= EMA9 will be allowed")
            print("📱 Telegram: Instant notifications enabled for super alerts")
            if self.test_mode:
                print("🧪 TEST MODE: Super alerts will be marked as [TEST MODE]")
            print("="*80 + "\n")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in alert monitor: {e}")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the ORB Alert Monitor."""
        self.logger.info("Stopping ORB Alert Monitor...")
        
        try:
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            self.logger.info("ORB Alert Monitor stopped")
        except Exception as e:
            self.logger.error(f"Error stopping monitor: {e}")
    
    def get_statistics(self) -> dict:
        """Get monitoring statistics."""
        super_alerts_count = sum(len(symbol_info.alerts_triggered) for symbol_info in self.symbol_data.values())
        
        return {
            'symbols_monitored': len(self.symbol_data),
            'super_alerts_generated': super_alerts_count,
            'alerts_processed': len(self.processed_alerts),
            'alerts_filtered': len(self.filtered_alerts),
            'monitoring_directory': str(self.alerts_dir),
            'super_alerts_directory': str(self.super_alerts_dir)
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Alerts Monitor - Super Alert Generation System")
    
    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Path to symbols CSV file (default: data/YYYYMMDD.csv for current date)"
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
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start monitor
    try:
        monitor = ORBAlertMonitor(
            symbols_file=args.symbols_file,
            test_mode=args.test
        )
        
        if args.test:
            print("Running in test mode - super alerts will be marked as [TEST MODE]")
        
        await monitor.start()
        
    except Exception as e:
        logging.error(f"Failed to start ORB Alert Monitor: {e}")
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