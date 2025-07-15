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


class SuperAlertData:
    """Data structure for super alert information."""
    
    def __init__(self, symbol: str, signal_price: float, resistance_price: float):
        self.symbol = symbol
        self.signal_price = signal_price
        self.resistance_price = resistance_price
        self.range_percent = (resistance_price / signal_price) if signal_price > 0 else 0
        self.alerts_triggered = []
        
    def calculate_penetration(self, current_price: float) -> float:
        """Calculate penetration percentage into Signal-to-Resistance range."""
        if current_price < self.signal_price:
            return 0.0
        
        range_size = self.resistance_price - self.signal_price
        if range_size <= 0:
            return 0.0
            
        penetration = (current_price - self.signal_price) / range_size
        return min(penetration * 100, 100.0)  # Cap at 100%


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
        
        # Load symbol data with Signal and Resistance prices
        self.symbol_manager = SymbolManager(symbols_file) if symbols_file else SymbolManager()
        self.symbol_data = self._load_symbol_data()
        self.test_mode = test_mode
        
        # Alert monitoring setup
        et_tz = pytz.timezone('US/Eastern')
        current_date = datetime.now(et_tz).strftime('%Y-%m-%d')
        self.alerts_dir = Path(f"historical_data/{current_date}/alerts/bullish")
        self.super_alerts_dir = Path(f"historical_data/{current_date}/super_alerts/bullish")
        
        # Ensure directories exist
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.super_alerts_dir.mkdir(parents=True, exist_ok=True)
        
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
        self.filtered_alerts = set()  # Track alerts filtered due to EMA9 below EMA20
        
        self.logger.info(f"ORB Alert Monitor initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Monitoring alerts in: {self.alerts_dir}")
        self.logger.info(f"Super alerts will be saved to: {self.super_alerts_dir}")
        self.logger.info(f"Loaded {len(self.symbol_data)} symbols with Signal/Resistance data")
    
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
    
    def _load_symbol_data(self) -> Dict[str, SuperAlertData]:
        """Load symbol data from CSV file with Signal and Resistance prices."""
        symbol_data = {}
        
        try:
            with open(self.symbol_manager.symbols_file, 'r') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    symbol = row.get('Symbol', '').strip().upper()
                    if not symbol or symbol in ['SYMBOL', 'TICKER', 'STOCK']:
                        continue
                    
                    # Parse Signal and Resistance prices
                    try:
                        signal_str = row.get('Signal', '0').strip()
                        resistance_str = row.get('Resistance', '0').strip()
                        
                        # Handle empty values
                        if not signal_str or not resistance_str:
                            continue
                            
                        signal_price = float(signal_str)
                        resistance_price = float(resistance_str)
                        
                        if signal_price > 0 and resistance_price > 0:
                            symbol_data[symbol] = SuperAlertData(symbol, signal_price, resistance_price)
                            self.logger.debug(f"Loaded {symbol}: Signal=${signal_price:.2f}, Resistance=${resistance_price:.2f}")
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Could not parse prices for {symbol}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error loading symbol data: {e}")
            
        return symbol_data
    
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
            
            symbol = alert_data.get('symbol')
            current_price = alert_data.get('current_price')
            
            if not symbol or current_price is None:
                self.logger.warning(f"Invalid alert data in {file_path}")
                return
            
            # Check if we have signal data for this symbol
            if symbol not in self.symbol_data:
                self.logger.debug(f"No signal data for {symbol}, skipping")
                return
            
            # Filter out bullish alerts where EMA9 is below EMA20
            breakout_type = alert_data.get('breakout_type', '').lower()
            if breakout_type == 'bullish_breakout':
                ema_9_below_20 = alert_data.get('ema_9_below_20')
                ema_9 = alert_data.get('ema_9')
                ema_20 = alert_data.get('ema_20')
                
                if ema_9_below_20 is True:
                    self.filtered_alerts.add(file_path)
                    ema_info = ""
                    if ema_9 is not None and ema_20 is not None:
                        ema_info = f" (EMA9: ${ema_9:.2f} < EMA20: ${ema_20:.2f})"
                    self.logger.info(f"🚫 Filtered bullish alert for {symbol}: EMA9 below EMA20{ema_info}")
                    return
                elif ema_9_below_20 is None:
                    # Handle case where EMA data is not available
                    self.logger.warning(f"No EMA9/EMA20 data available for {symbol}, allowing alert")
                else:
                    # EMA9 is above EMA20, log this for bullish alerts
                    ema_info = ""
                    if ema_9 is not None and ema_20 is not None:
                        ema_info = f" (EMA9: ${ema_9:.2f} > EMA20: ${ema_20:.2f})"
                    self.logger.debug(f"✅ Allowing bullish alert for {symbol}: EMA9 above EMA20{ema_info}")
            
            symbol_info = self.symbol_data[symbol]
            
            # Check if current price has reached signal price
            if current_price >= symbol_info.signal_price:
                await self._create_super_alert(alert_data, symbol_info)
            else:
                self.logger.debug(f"{symbol}: Price ${current_price:.2f} below Signal ${symbol_info.signal_price:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error processing alert file {file_path}: {e}")
    
    async def _create_super_alert(self, alert_data: dict, symbol_info: SuperAlertData) -> None:
        """Create a super alert when signal price is reached."""
        try:
            symbol = alert_data['symbol']
            current_price = alert_data['current_price']
            original_timestamp = alert_data['timestamp']
            
            # Create ET timestamp for when super alert is generated
            et_tz = pytz.timezone('US/Eastern')
            super_alert_time = datetime.now(et_tz)
            et_timestamp = super_alert_time.strftime('%Y-%m-%dT%H:%M:%S%z')
            
            # Calculate metrics
            penetration = symbol_info.calculate_penetration(current_price)
            range_percent = symbol_info.range_percent
            
            # Create super alert data
            super_alert = {
                "symbol": symbol,
                "timestamp": et_timestamp,
                "original_alert_timestamp": original_timestamp,
                "alert_type": "super_alert",
                "trigger_condition": "signal_price_reached",
                "original_alert": alert_data,
                "signal_analysis": {
                    "signal_price": symbol_info.signal_price,
                    "resistance_price": symbol_info.resistance_price,
                    "current_price": current_price,
                    "penetration_percent": round(penetration, 2),
                    "range_percent": round(range_percent, 2),
                    "signal_reached": True,
                    "resistance_reached": current_price >= symbol_info.resistance_price
                },
                "metrics": {
                    "signal_to_resistance_range": symbol_info.resistance_price - symbol_info.signal_price,
                    "price_above_signal": current_price - symbol_info.signal_price,
                    "distance_to_resistance": symbol_info.resistance_price - current_price
                },
                "risk_assessment": {
                    "entry_price": symbol_info.signal_price,
                    "target_price": symbol_info.resistance_price,
                    "current_risk_reward": (symbol_info.resistance_price - current_price) / (current_price - symbol_info.signal_price) if current_price > symbol_info.signal_price else 0
                }
            }
            
            # Save super alert using the same ET time
            filename = f"super_alert_{symbol}_{super_alert_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.super_alerts_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(super_alert, f, indent=2)
            
            # Log and display super alert
            message = (f"🚀 SUPER ALERT: {symbol} @ ${current_price:.2f}\n"
                      f"   Signal: ${symbol_info.signal_price:.2f} ✅ | "
                      f"Resistance: ${symbol_info.resistance_price:.2f}\n"
                      f"   Penetration: {penetration:.1f}% | "
                      f"Range %: {range_percent:.1f}%\n"
                      f"   Saved: {filename}")
            
            if self.test_mode:
                print(f"[TEST MODE] {message}")
            else:
                print(message)
                
            self.logger.info(f"Super alert created for {symbol} at ${current_price:.2f}")
            
            # Track this alert
            symbol_info.alerts_triggered.append(super_alert)
            
        except Exception as e:
            self.logger.error(f"Error creating super alert: {e}")
    
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
            print("🚫 Filtering: Bullish alerts with EMA9 < EMA20 will be filtered out")
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
            'alerts_filtered_ema': len(self.filtered_alerts),
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