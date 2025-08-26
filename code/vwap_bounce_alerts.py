"""
VWAP Bounce Alerts System

This system monitors historical data files for VWAP bounce patterns and sends Telegram alerts.
It watches for new CSV files in historical_data/YYYY-MM-DD/market_data/ directory,
analyzes the last 10 1-minute candlesticks, combines them into two 5-minute candlesticks,
and alerts when both are green and one is within 7% above VWAP.

Usage:
    python3 code/vwap_bounce_alerts.py                     # Start monitoring for current date
    python3 code/vwap_bounce_alerts.py --date 2025-07-18   # Monitor specific date
    python3 code/vwap_bounce_alerts.py --test              # Run in test mode (no alerts sent)
"""

import asyncio
import argparse
import logging
import os
import sys
import json
import pandas as pd
import pytz
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.config.alert_config import config
from atoms.telegram.telegram_post import TelegramPoster
from atoms.alerts.config import get_logs_root_dir


class VWAPBounceAlert:
    """Data class for VWAP bounce alert information."""
    
    def __init__(self, symbol: str, timestamp: datetime, 
                 first_5min: Dict, second_5min: Dict, 
                 vwap_distance: float, current_vwap: float):
        self.symbol = symbol
        self.timestamp = timestamp
        self.first_5min = first_5min
        self.second_5min = second_5min
        self.vwap_distance = vwap_distance
        self.current_vwap = current_vwap
        
    def format_alert_message(self) -> str:
        """Format the alert message for Telegram."""
        return f"""🟢 VWAP BOUNCE ALERT - {self.symbol}

📊 **Pattern Detected**: Two consecutive 5-minute green candles
⏰ **Time**: {self.timestamp.strftime('%H:%M ET')}
📈 **VWAP**: ${self.current_vwap:.4f}
🎯 **Distance to VWAP**: {self.vwap_distance:.2f}%

**5-Min Candle 1**: ${self.first_5min['open']:.4f} → ${self.first_5min['close']:.4f} (+{((self.first_5min['close']/self.first_5min['open']-1)*100):.2f}%)
**5-Min Candle 2**: ${self.second_5min['open']:.4f} → ${self.second_5min['close']:.4f} (+{((self.second_5min['close']/self.second_5min['open']-1)*100):.2f}%)

Volume: {self.first_5min['volume']:,} + {self.second_5min['volume']:,} = {self.first_5min['volume'] + self.second_5min['volume']:,}

🚀 Potential bounce off VWAP support level detected!"""


class VWAPBounceDetector:
    """Analyzes market data for VWAP bounce patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_data(self, df: pd.DataFrame) -> Optional[VWAPBounceAlert]:
        """
        Analyze the last 10 minutes of data for VWAP bounce pattern.
        
        Args:
            df: DataFrame with 1-minute candlestick data
            
        Returns:
            VWAPBounceAlert if pattern detected, None otherwise
        """
        if len(df) < 10:
            self.logger.debug(f"Insufficient data: {len(df)} rows (need 10)")
            return None
            
        # Get the last 10 rows and sort by timestamp
        df = df.sort_values('timestamp').tail(10).reset_index(drop=True)
        
        # Combine into two 5-minute candlesticks
        first_5min = self._combine_candlesticks(df.iloc[:5])
        second_5min = self._combine_candlesticks(df.iloc[5:])
        
        # Check if both candles are green
        first_green = first_5min['close'] > first_5min['open']
        second_green = second_5min['close'] > second_5min['open']
        
        if not (first_green and second_green):
            self.logger.debug(f"Candles not both green: first={first_green}, second={second_green}")
            return None
            
        # Get the most recent VWAP value
        current_vwap = df.iloc[-1]['vwap']
        
        # Check if either candle is within 7% above VWAP
        first_distance = ((first_5min['close'] / current_vwap) - 1) * 100
        second_distance = ((second_5min['close'] / current_vwap) - 1) * 100
        
        within_7_percent = (0 <= first_distance <= 7) or (0 <= second_distance <= 7)
        
        if not within_7_percent:
            self.logger.debug(f"Not within 7% of VWAP: first={first_distance:.2f}%, second={second_distance:.2f}%")
            return None
            
        # Use the closer distance for reporting
        vwap_distance = min(first_distance, second_distance) if first_distance > 0 and second_distance > 0 else max(first_distance, second_distance)
        
        symbol = df.iloc[-1]['symbol']
        timestamp = pd.to_datetime(df.iloc[-1]['timestamp'])
        
        self.logger.info(f"VWAP bounce pattern detected for {symbol}: {vwap_distance:.2f}% from VWAP")
        
        return VWAPBounceAlert(
            symbol=symbol,
            timestamp=timestamp,
            first_5min=first_5min,
            second_5min=second_5min,
            vwap_distance=vwap_distance,
            current_vwap=current_vwap
        )
        
    def _combine_candlesticks(self, df_slice: pd.DataFrame) -> Dict:
        """Combine multiple 1-minute candles into a single 5-minute candle."""
        return {
            'open': df_slice.iloc[0]['open'],
            'high': df_slice['high'].max(),
            'low': df_slice['low'].min(), 
            'close': df_slice.iloc[-1]['close'],
            'volume': df_slice['volume'].sum(),
            'vwap': df_slice['vwap'].iloc[-1]  # Use the last VWAP value
        }


class MarketDataFileHandler(FileSystemEventHandler):
    """Handles file creation events in the market data directory."""
    
    def __init__(self, vwap_system):
        self.vwap_system = vwap_system
        self.logger = logging.getLogger(__name__)
        
    def on_created(self, event):
        """Handle file creation events."""
        if isinstance(event, FileCreatedEvent) and event.src_path.endswith('.csv'):
            # Extract symbol from filename (format: SYMBOL_YYYYMMDD_HHMMSS.csv)
            filename = Path(event.src_path).name
            if '_' in filename:
                symbol = filename.split('_')[0]
                self.logger.debug(f"New file created for symbol {symbol}: {filename}")
                asyncio.create_task(self.vwap_system.process_symbol_file(event.src_path, symbol))


class VWAPBounceSystem:
    """Main VWAP Bounce Alerts System."""
    
    def __init__(self, target_date: str = None, test_mode: bool = False):
        """
        Initialize VWAP Bounce System.
        
        Args:
            target_date: Date to monitor in YYYY-MM-DD format (default: today)
            test_mode: Run in test mode (no actual alerts sent)
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Store configuration
        self.test_mode = test_mode
        
        # Set target date
        if target_date is None:
            et_tz = pytz.timezone('US/Eastern')
            self.target_date = datetime.now(et_tz).strftime('%Y-%m-%d')
        else:
            self.target_date = target_date
            
        # Initialize components
        self.detector = VWAPBounceDetector()
        self.telegram_poster = TelegramPoster()
        
        # Setup monitoring directory
        self.historical_data_dir = Path("historical_data")
        self.target_data_dir = self.historical_data_dir / self.target_date / "market_data"
        
        # Ensure target directory exists
        if not self.target_data_dir.exists():
            self.target_data_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created monitoring directory: {self.target_data_dir}")
            
        # Initialize file system observer
        self.observer = Observer()
        self.file_handler = MarketDataFileHandler(self)
        
        # Track processed files and alerts sent to avoid duplicates
        self.processed_files = set()
        self.alerted_symbols = set()
        
        self.logger.info(f"VWAP Bounce System initialized for {self.target_date}")
        self.logger.info(f"Monitoring directory: {self.target_data_dir.absolute()}")
        self.logger.info(f"Test mode: {'ENABLED' if test_mode else 'DISABLED'}")
        
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
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler
            try:
                logs_config = get_logs_root_dir()
                log_dir = logs_config.get_component_logs_dir("vwap_bounce_alerts")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                et_tz = pytz.timezone('US/Eastern')
                log_filename = f"vwap_bounce_alerts_{datetime.now(et_tz).strftime('%Y%m%d_%H%M%S')}.log"
                log_file_path = log_dir / log_filename
                
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
                
            logger.setLevel(logging.INFO)
            
        return logger
        
    def _is_within_trading_window(self) -> bool:
        """Check if current time is within the trading alert window."""
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        # Parse market open and close times from config
        market_start = dt_time.fromisoformat(config.alert_window_start)  # Should be close to market open
        market_end = dt_time.fromisoformat(config.alert_window_end)      # 20:00 ET
        
        current_time = now_et.time()
        
        # Check if within trading window
        within_window = market_start <= current_time <= market_end
        
        if not within_window:
            self.logger.debug(f"Outside trading window. Current: {current_time}, Window: {market_start}-{market_end}")
            
        return within_window
        
    async def process_symbol_file(self, file_path: str, symbol: str):
        """
        Process a symbol's market data file for VWAP bounce patterns.
        
        Args:
            file_path: Path to the CSV file
            symbol: Symbol being processed
        """
        try:
            # Avoid processing the same file multiple times
            if file_path in self.processed_files:
                return
                
            self.processed_files.add(file_path)
            
            # Check if we're within trading hours
            if not self._is_within_trading_window():
                self.logger.debug(f"Outside trading window, skipping {symbol}")
                return
                
            # Check if we've already alerted for this symbol today
            if symbol in self.alerted_symbols:
                self.logger.debug(f"Already alerted for {symbol} today, skipping")
                return
                
            # Wait a brief moment to ensure file is fully written
            await asyncio.sleep(0.5)
            
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {e}")
                return
                
            # Analyze for VWAP bounce pattern
            alert = self.detector.analyze_data(df)
            
            if alert:
                await self._send_alert(alert)
                # Mark this symbol as alerted to avoid duplicate alerts
                self.alerted_symbols.add(symbol)
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            
    async def _send_alert(self, alert: VWAPBounceAlert):
        """Send VWAP bounce alert via Telegram and save to historical data."""
        try:
            # Save alert data to historical files
            self._save_alert_data(alert)
            
            message = alert.format_alert_message()
            
            if self.test_mode:
                print(f"[TEST MODE] VWAP Bounce Alert for {alert.symbol}")
                print(message)
                print("-" * 60)
                self.logger.info(f"Test alert generated for {alert.symbol}")
            else:
                # Send to Bruce specifically
                result = self.telegram_poster.send_message_to_user(message, "bruce", urgent=False)
                
                if result['success']:
                    self.logger.info(f"✅ VWAP bounce alert sent to Bruce for {alert.symbol}")
                else:
                    self.logger.error(f"❌ Failed to send alert for {alert.symbol}: {result.get('errors', [])}")
                    
        except Exception as e:
            self.logger.error(f"Error sending alert for {alert.symbol}: {e}")
            
    def _save_alert_data(self, alert: VWAPBounceAlert):
        """Save VWAP bounce alert data to historical files."""
        try:
            # Create vwap_bounce subdirectory in alerts
            alert_dir = self.historical_data_dir / self.target_date / "alerts" / "vwap_bounce"
            alert_dir.mkdir(parents=True, exist_ok=True)
            
            # Create alert filename with timestamp
            alert_filename = f"vwap_bounce_{alert.symbol}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            alert_filepath = alert_dir / alert_filename
            
            # Convert alert to dictionary for JSON serialization
            alert_data = {
                "symbol": alert.symbol,
                "timestamp": alert.timestamp.isoformat(),
                "alert_type": "vwap_bounce",
                "current_vwap": float(alert.current_vwap),
                "vwap_distance_percent": float(alert.vwap_distance),
                "first_5min_candle": {
                    "open": float(alert.first_5min['open']),
                    "high": float(alert.first_5min['high']),
                    "low": float(alert.first_5min['low']),
                    "close": float(alert.first_5min['close']),
                    "volume": int(alert.first_5min['volume']),
                    "vwap": float(alert.first_5min['vwap']),
                    "gain_percent": round(((alert.first_5min['close'] / alert.first_5min['open']) - 1) * 100, 2)
                },
                "second_5min_candle": {
                    "open": float(alert.second_5min['open']),
                    "high": float(alert.second_5min['high']),
                    "low": float(alert.second_5min['low']),
                    "close": float(alert.second_5min['close']),
                    "volume": int(alert.second_5min['volume']),
                    "vwap": float(alert.second_5min['vwap']),
                    "gain_percent": round(((alert.second_5min['close'] / alert.second_5min['open']) - 1) * 100, 2)
                },
                "total_volume": int(alert.first_5min['volume'] + alert.second_5min['volume']),
                "detection_criteria": {
                    "both_candles_green": True,
                    "within_7_percent_of_vwap": True,
                    "max_vwap_distance_threshold": 7.0
                },
                "alert_message": alert.format_alert_message(),
                "generated_at": datetime.now(pytz.timezone('US/Eastern')).isoformat()
            }
            
            # Save to JSON file
            with open(alert_filepath, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
            self.logger.debug(f"Saved VWAP bounce alert data for {alert.symbol} to {alert_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving VWAP bounce alert data for {alert.symbol}: {e}")
            
    async def start_monitoring(self):
        """Start monitoring for new market data files."""
        try:
            self.logger.info("Starting VWAP Bounce monitoring...")
            
            # Setup file system watcher
            self.observer.schedule(
                self.file_handler,
                str(self.target_data_dir),
                recursive=False
            )
            
            # Start the observer
            self.observer.start()
            self.logger.info(f"File system watcher started for: {self.target_data_dir}")
            
            # Process any existing files in the directory
            await self._process_existing_files()
            
            # Keep the system running
            try:
                while True:
                    await asyncio.sleep(10)  # Check every 10 seconds
                    
                    # Stop monitoring if outside trading hours for too long
                    if not self._is_within_trading_window():
                        et_tz = pytz.timezone('US/Eastern')
                        current_time = datetime.now(et_tz).time()
                        end_time = dt_time.fromisoformat(config.alert_window_end)
                        
                        # If it's past 21:00 ET, stop for the day
                        if current_time > dt_time(21, 0):
                            self.logger.info("Past 21:00 ET, stopping monitoring for the day")
                            break
                            
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, shutting down...")
                
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
        finally:
            await self.stop_monitoring()
            
    async def _process_existing_files(self):
        """Process any existing files in the target directory."""
        try:
            existing_files = list(self.target_data_dir.glob("*.csv"))
            if existing_files:
                self.logger.info(f"Processing {len(existing_files)} existing files...")
                
                for file_path in existing_files:
                    # Extract symbol from filename
                    filename = file_path.name
                    if '_' in filename:
                        symbol = filename.split('_')[0]
                        await self.process_symbol_file(str(file_path), symbol)
                        await asyncio.sleep(0.1)  # Brief delay between files
                        
        except Exception as e:
            self.logger.error(f"Error processing existing files: {e}")
            
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        try:
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            self.logger.info("VWAP Bounce monitoring stopped")
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
            
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return {
            'target_date': self.target_date,
            'processed_files_count': len(self.processed_files),
            'alerted_symbols_count': len(self.alerted_symbols),
            'alerted_symbols': list(self.alerted_symbols),
            'monitoring_directory': str(self.target_data_dir),
            'test_mode': self.test_mode
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VWAP Bounce Alerts System")
    
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d'),
        help="Date to monitor in YYYY-MM-DD format (default: today)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (no actual alerts sent)"
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
        
    try:
        # Create and start the system
        system = VWAPBounceSystem(
            target_date=args.date,
            test_mode=args.test
        )
        
        if args.test:
            print("Running in test mode - no alerts will be sent to Telegram")
            
        print(f"Monitoring VWAP bounces for {args.date}")
        print(f"Watching directory: {system.target_data_dir}")
        print("Press Ctrl+C to stop...")
        
        await system.start_monitoring()
        
    except Exception as e:
        logging.error(f"Failed to start VWAP Bounce System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())