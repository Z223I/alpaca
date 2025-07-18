"""
Alert Monitor History - End-of-Day Super Alert Generation System

This system processes the alerts directory for bullish ORB alerts and creates super alerts
based on historical data from the trading day. It analyzes all alerts from a given date
and generates super alerts when the current price reached the Signal price from the CSV file.

Usage:
    python3 code/orb_alerts_history.py                          # Process current date alerts
    python3 code/orb_alerts_history.py --date 2025-01-15        # Process specific date
    python3 code/orb_alerts_history.py --symbols-file data/YYYYMMDD.csv  # Use specific symbols file
    python3 code/orb_alerts_history.py --test                   # Run in test mode
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
from atoms.config.symbol_data_loader import SymbolDataLoader
from atoms.alerts.super_alert_filter import SuperAlertFilter, SuperAlertData
from atoms.alerts.super_alert_generator import SuperAlertGenerator


# SuperAlertData is now imported from atoms.alerts.super_alert_filter


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
        
        # Load symbol data with Signal and Resistance prices using atom
        self.symbol_loader = SymbolDataLoader(symbols_file)
        self.symbol_data = self.symbol_loader.load_symbol_data()
        self.test_mode = test_mode
        
        # Initialize filtering and generation atoms
        self.super_alert_filter = SuperAlertFilter(self.symbol_data)
        self.super_alert_generator = None  # Will be initialized when directories are set up
        
        # Directory setup
        self.alerts_dir = Path(f"historical_data/{self.target_date}/alerts/bullish")
        self.super_alerts_dir = Path(f"historical_data/{self.target_date}/super_alerts/bullish")
        
        # Ensure super alerts directory exists
        self.super_alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize super alert generator now that directory is set up
        self.super_alert_generator = SuperAlertGenerator(self.super_alerts_dir, test_mode)
        
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
    
    # Symbol data loading is now handled by SymbolDataLoader atom
    
    def _process_alert_file(self, file_path: Path) -> None:
        """Process a single alert file."""
        try:
            with open(file_path, 'r') as f:
                alert_data = json.load(f)
            
            self.processed_alerts.append(alert_data)
            
            # Use SuperAlertFilter to determine if we should create a super alert
            should_create, filter_reason = self.super_alert_filter.should_create_super_alert(alert_data)
            
            if not should_create:
                if filter_reason.startswith("Price") or filter_reason.startswith("No signal"):
                    # Log at debug level for price/signal issues
                    self.logger.debug(f"Skipping alert: {filter_reason}")
                else:
                    # Log filtered alerts
                    self.filtered_alerts.append(alert_data)
                    self.logger.info(f"🚫 Filtered alert: {filter_reason}")
                return
            
            # Get symbol info and create super alert
            symbol = alert_data.get('symbol')
            symbol_info = self.super_alert_filter.get_symbol_info(symbol)
            
            if symbol_info:
                filename = self.super_alert_generator.create_and_save_super_alert(
                    alert_data, symbol_info, use_original_timestamp=True)
                if filename:
                    # Create super alert for tracking
                    super_alert = self.super_alert_generator.create_super_alert(
                        alert_data, symbol_info, use_original_timestamp=True)
                    if super_alert:
                        super_alert["processing_mode"] = "historical"
                        super_alert["source_alert_file"] = str(file_path)
                        self.super_alerts_created.append(super_alert)
                        self.logger.info(f"Super alert created: {filename}")
                
        except Exception as e:
            self.logger.error(f"Error processing alert file {file_path}: {e}")
    
    # _create_super_alert_historical method is now handled by SuperAlertGenerator atom with use_original_timestamp=True
    
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
            print(f"🚫 Alerts filtered: {len(self.filtered_alerts)}")
            print(f"🚀 Super alerts created: {len(self.super_alerts_created)}")
            print(f"📊 Symbols monitored: {len(self.symbol_data)}")
            if self.test_mode:
                print("🧪 TEST MODE: No actual super alert files were created")
            print("✅ Filtering: Only bullish alerts with candlestick low >= EMA9 are allowed")
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
            'alerts_filtered': len(self.filtered_alerts),
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