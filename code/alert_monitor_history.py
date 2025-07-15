"""
Alert Monitor History - End-of-Day Super Alert Generation System

This system processes the alerts directory for bullish ORB alerts and creates super alerts
based on historical data from the trading day. It analyzes all alerts from a given date
and generates super alerts when the current price reached the Signal price from the CSV file.

Usage:
    python3 code/alert_monitor_history.py                          # Process current date alerts
    python3 code/alert_monitor_history.py --date 2025-01-15        # Process specific date
    python3 code/alert_monitor_history.py --symbols-file data/YYYYMMDD.csv  # Use specific symbols file
    python3 code/alert_monitor_history.py --test                   # Run in test mode
"""

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

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from atoms.config.alert_config import config
from atoms.config.symbol_manager import SymbolManager


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


class AlertMonitorHistory:
    """End-of-day alert history processor that creates super alerts from historical alert data."""
    
    def __init__(self, date: Optional[str] = None, symbols_file: Optional[str] = None, test_mode: bool = False):
        """
        Initialize Alert Monitor History.
        
        Args:
            date: Target date in YYYY-MM-DD format (defaults to current ET date)
            symbols_file: Path to symbols CSV file
            test_mode: Run in test mode (no actual alerts)
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Set target date
        et_tz = pytz.timezone('US/Eastern')
        if date:
            try:
                self.target_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date format: {date}. Use YYYY-MM-DD format.")
        else:
            self.target_date = datetime.now(et_tz).strftime('%Y-%m-%d')
        
        # Load symbol data with Signal and Resistance prices
        self.symbol_manager = SymbolManager(symbols_file) if symbols_file else SymbolManager()
        self.symbol_data = self._load_symbol_data()
        self.test_mode = test_mode
        
        # Directory setup
        self.alerts_dir = Path(f"historical_data/{self.target_date}/alerts/bullish")
        self.super_alerts_dir = Path(f"historical_data/{self.target_date}/super_alerts/bullish")
        
        # Ensure super alerts directory exists
        self.super_alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Processed alerts tracking
        self.processed_alerts = []
        self.super_alerts_created = []
        self.filtered_alerts = []  # Track alerts filtered due to EMA9 below EMA20
        
        self.logger.info(f"Alert Monitor History initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Processing date: {self.target_date}")
        self.logger.info(f"Reading alerts from: {self.alerts_dir}")
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
    
    def _process_alert_file(self, file_path: Path) -> None:
        """Process a single alert file."""
        try:
            with open(file_path, 'r') as f:
                alert_data = json.load(f)
            
            symbol = alert_data.get('symbol')
            current_price = alert_data.get('current_price')
            
            if not symbol or current_price is None:
                self.logger.warning(f"Invalid alert data in {file_path}")
                return
            
            self.processed_alerts.append(alert_data)
            
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
                    self.filtered_alerts.append(alert_data)
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
                self._create_super_alert(alert_data, symbol_info, file_path)
            else:
                self.logger.debug(f"{symbol}: Price ${current_price:.2f} below Signal ${symbol_info.signal_price:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error processing alert file {file_path}: {e}")
    
    def _create_super_alert(self, alert_data: dict, symbol_info: SuperAlertData, source_file: Path) -> None:
        """Create a super alert when signal price is reached."""
        try:
            symbol = alert_data['symbol']
            current_price = alert_data['current_price']
            original_timestamp = alert_data['timestamp']
            
            # Use original alert timestamp as the super alert timestamp for realism
            et_timestamp = original_timestamp
            
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
                "processing_mode": "historical",
                "source_alert_file": str(source_file),
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
            
            # Save super alert using the original alert timestamp for filename
            try:
                # Handle different timestamp formats
                if original_timestamp.endswith('Z'):
                    original_dt = datetime.fromisoformat(original_timestamp.replace('Z', '+00:00'))
                elif '+' in original_timestamp or original_timestamp.count('-') > 2:
                    original_dt = datetime.fromisoformat(original_timestamp)
                else:
                    # Assume no timezone info, parse as-is and assume ET
                    original_dt = datetime.fromisoformat(original_timestamp)
                    if original_dt.tzinfo is None:
                        et_tz = pytz.timezone('US/Eastern')
                        original_dt = et_tz.localize(original_dt)
                
                filename = f"super_alert_{symbol}_{original_dt.strftime('%Y%m%d_%H%M%S')}_historical.json"
            except Exception as e:
                # Fallback to a safe filename if timestamp parsing fails
                self.logger.warning(f"Could not parse timestamp '{original_timestamp}' for filename: {e}")
                filename = f"super_alert_{symbol}_unknown_time_historical.json"
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
            self.super_alerts_created.append(super_alert)
            
        except Exception as e:
            self.logger.error(f"Error creating super alert: {e}")
    
    def process_alerts(self) -> None:
        """Process all alert files from the target date."""
        self.logger.info("Starting historical alert processing...")
        
        try:
            if not self.alerts_dir.exists():
                self.logger.error(f"Alerts directory does not exist: {self.alerts_dir}")
                return
                
            alert_files = list(self.alerts_dir.glob("alert_*.json"))
            self.logger.info(f"Found {len(alert_files)} alert files to process...")
            
            if not alert_files:
                self.logger.info("No alert files found to process")
                return
            
            # Process each alert file
            for alert_file in sorted(alert_files):
                self._process_alert_file(alert_file)
                
            # Print summary
            print("\n" + "="*80)
            print("📊 HISTORICAL ALERT PROCESSING COMPLETE")
            print(f"📅 Date processed: {self.target_date}")
            print(f"📁 Alerts directory: {self.alerts_dir}")
            print(f"💾 Super alerts directory: {self.super_alerts_dir}")
            print(f"📈 Alerts processed: {len(self.processed_alerts)}")
            print(f"🚫 Alerts filtered (EMA9 < EMA20): {len(self.filtered_alerts)}")
            print(f"🚀 Super alerts created: {len(self.super_alerts_created)}")
            print(f"📊 Symbols monitored: {len(self.symbol_data)}")
            if self.test_mode:
                print("🧪 TEST MODE: No actual super alert files were created")
            print("🚫 Filtering: Bullish alerts with EMA9 < EMA20 are filtered out")
            print("="*80 + "\n")
            
            # Print super alert details
            if self.super_alerts_created:
                print("🚀 SUPER ALERTS GENERATED:")
                for super_alert in self.super_alerts_created:
                    symbol = super_alert['symbol']
                    price = super_alert['signal_analysis']['current_price']
                    penetration = super_alert['signal_analysis']['penetration_percent']
                    print(f"   {symbol}: ${price:.2f} ({penetration:.1f}% penetration)")
                print()
            
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")
    
    def get_statistics(self) -> dict:
        """Get processing statistics."""
        return {
            'target_date': self.target_date,
            'symbols_monitored': len(self.symbol_data),
            'alerts_processed': len(self.processed_alerts),
            'alerts_filtered_ema': len(self.filtered_alerts),
            'super_alerts_generated': len(self.super_alerts_created),
            'alerts_directory': str(self.alerts_dir),
            'super_alerts_directory': str(self.super_alerts_dir),
            'test_mode': self.test_mode
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Alert Monitor History - End-of-Day Super Alert Generation System")
    
    parser.add_argument(
        "--date",
        type=str,
        help="Target date to process in YYYY-MM-DD format (default: current ET date)"
    )
    
    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Path to symbols CSV file (default: data/YYYYMMDD.csv for target date)"
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


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run processor
    try:
        processor = AlertMonitorHistory(
            date=args.date,
            symbols_file=args.symbols_file,
            test_mode=args.test
        )
        
        if args.test:
            print("Running in test mode - no super alert files will be created")
        
        processor.process_alerts()
        
        # Print final statistics
        stats = processor.get_statistics()
        logging.info(f"Processing complete. Statistics: {stats}")
        
    except Exception as e:
        logging.error(f"Failed to process alert history: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()