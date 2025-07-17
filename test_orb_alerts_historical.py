#!/usr/bin/env python3
"""
Test ORB Alerts Historical Data Processor

This script takes the most current market data from historical_data/2025-07-17/market_data/*.csv
and runs it through the ORB alerts logic to produce alerts in the alerts directory.

Usage:
    python3 test_orb_alerts_historical.py
    python3 test_orb_alerts_historical.py --verbose
    python3 test_orb_alerts_historical.py --dry-run
"""

import os
import sys
import json
import glob
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import logging

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from atoms.config.alert_config import config
from atoms.alerts.alert_formatter import AlertFormatter, ORBAlert
from atoms.alerts.breakout_detector import BreakoutDetector
from atoms.alerts.confidence_scorer import ConfidenceScorer
from atoms.websocket.alpaca_stream import MarketData
from atoms.websocket.data_buffer import DataBuffer
from atoms.indicators.orb_calculator import ORBCalculator


class HistoricalORBProcessor:
    """Process historical market data through ORB alerts logic."""
    
    def __init__(self, target_date: str = "2025-07-17", dry_run: bool = False, verbose: bool = False):
        """
        Initialize the historical ORB processor.
        
        Args:
            target_date: Date to process (YYYY-MM-DD format)
            dry_run: If True, don't save alerts to files
            verbose: Enable verbose logging
        """
        self.target_date = target_date
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_buffer = DataBuffer()
        self.orb_calculator = ORBCalculator()
        self.breakout_detector = BreakoutDetector(self.orb_calculator)  # Pass orb_calculator
        self.confidence_scorer = ConfidenceScorer()
        
        # Setup output directories
        self.alerts_dir = Path("alerts")
        self.alerts_dir.mkdir(exist_ok=True)
        (self.alerts_dir / "bullish").mkdir(exist_ok=True)
        (self.alerts_dir / "bearish").mkdir(exist_ok=True)
        
        self.alert_formatter = AlertFormatter(str(self.alerts_dir))
        
        # Statistics
        self.processed_symbols = 0
        self.generated_alerts = 0
        self.bullish_alerts = 0
        self.bearish_alerts = 0
        
        self.logger.info(f"Historical ORB Processor initialized")
        self.logger.info(f"Target date: {target_date}")
        self.logger.info(f"Dry run: {dry_run}")
        self.logger.info(f"Verbose: {verbose}")
    
    def _get_latest_market_data_files(self) -> Dict[str, str]:
        """
        Get the most recent market data file for each symbol.
        
        Returns:
            Dictionary mapping symbol to latest CSV file path
        """
        market_data_dir = Path(f"historical_data/{self.target_date}/market_data")
        
        if not market_data_dir.exists():
            self.logger.error(f"Market data directory not found: {market_data_dir}")
            return {}
        
        # Find all CSV files
        csv_files = list(market_data_dir.glob("*.csv"))
        
        if not csv_files:
            self.logger.error(f"No CSV files found in {market_data_dir}")
            return {}
        
        # Group by symbol and find latest timestamp
        symbol_files = {}
        for file_path in csv_files:
            try:
                # Extract symbol and timestamp from filename: SYMBOL_YYYYMMDD_HHMMSS.csv
                filename = file_path.name
                parts = filename.replace('.csv', '').split('_')
                
                if len(parts) >= 3:
                    symbol = parts[0]
                    timestamp_str = '_'.join(parts[1:3])  # YYYYMMDD_HHMMSS
                    
                    if symbol not in symbol_files:
                        symbol_files[symbol] = []
                    
                    symbol_files[symbol].append((timestamp_str, str(file_path)))
            
            except Exception as e:
                self.logger.warning(f"Could not parse filename {file_path}: {e}")
                continue
        
        # Get latest file for each symbol
        latest_files = {}
        for symbol, files in symbol_files.items():
            # Sort by timestamp and take the latest
            files.sort(key=lambda x: x[0], reverse=True)
            latest_timestamp, latest_file = files[0]
            latest_files[symbol] = latest_file
            
            self.logger.debug(f"Latest file for {symbol}: {latest_file}")
        
        self.logger.info(f"Found latest market data files for {len(latest_files)} symbols")
        return latest_files
    
    def _load_market_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load market data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with market data or None if error
        """
        try:
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'symbol', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in {file_path}")
                return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add 'open' column if missing (use previous close or current close)
            if 'open' not in df.columns:
                df['open'] = df['close'].shift(1).fillna(df['close'])
                self.logger.debug(f"Added 'open' column to DataFrame")
            
            # Add vwap and trade_count if missing
            if 'vwap' not in df.columns:
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
            
            if 'trade_count' not in df.columns:
                df['trade_count'] = 1
            
            self.logger.debug(f"Loaded {len(df)} rows from {file_path}")
            self.logger.debug(f"DataFrame columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _populate_data_buffer(self, df: pd.DataFrame) -> None:
        """
        Populate the data buffer with market data.
        
        Args:
            df: DataFrame with market data
        """
        for _, row in df.iterrows():
            try:
                market_data = MarketData(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'].to_pydatetime().replace(tzinfo=None),
                    price=float(row['close']),
                    volume=int(row['volume']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    trade_count=int(row.get('trade_count', 1)),
                    vwap=float(row.get('vwap', row['close'])),
                    open=float(row.get('open', row['close']))
                )
                
                self.data_buffer.add_market_data(market_data)
                
            except Exception as e:
                self.logger.error(f"Error adding market data to buffer: {e}")
                continue
    
    def _process_symbol_for_alerts(self, symbol: str) -> List[ORBAlert]:
        """
        Process a symbol's data for ORB alerts.
        
        Args:
            symbol: Symbol to process
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        try:
            # Get symbol data from buffer
            symbol_data = self.data_buffer.get_symbol_data(symbol)
            
            if symbol_data is None or symbol_data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return alerts
            
            if len(symbol_data) < 15:  # Need at least 15 minutes for ORB
                self.logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} rows")
                return alerts
            
            self.logger.debug(f"Processing {symbol} with {len(symbol_data)} data points")
            
            # Calculate ORB levels
            orb_result = self.orb_calculator.calculate_orb_levels(symbol, symbol_data)
            
            if not orb_result or orb_result.orb_high is None or orb_result.orb_low is None:
                self.logger.warning(f"Could not calculate ORB levels for {symbol}")
                return alerts
            
            self.logger.debug(f"ORB levels for {symbol}: High={orb_result.orb_high:.4f}, Low={orb_result.orb_low:.4f}")
            
            # Calculate breakout thresholds
            breakout_threshold = orb_result.orb_high * (1 + config.breakout_threshold)
            breakdown_threshold = orb_result.orb_low * (1 - config.breakout_threshold)
            self.logger.debug(f"Breakout thresholds for {symbol}: Bullish={breakout_threshold:.4f}, Bearish={breakdown_threshold:.4f}")
            
            # Process each data point after ORB period for potential alerts
            orb_end_index = 15  # 15 minutes for ORB period
            
            for i in range(orb_end_index, len(symbol_data)):
                current_row = symbol_data.iloc[i]
                current_time = current_row['timestamp']
                current_price = current_row['close']
                
                # Check if we're in alert window (9:45 AM to 3:45 PM)
                if not self._is_in_alert_window(current_time):
                    continue
                
                # Get historical data up to this point
                historical_data = symbol_data.iloc[:i+1]
                
                # Calculate volume ratio (current vs average)
                current_volume = current_row['volume']
                avg_volume = historical_data['volume'].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # Check for breakout
                breakout_signal = self.breakout_detector.detect_breakout(
                    symbol, current_price, volume_ratio, current_time
                )
                
                # Debug breakout detection (optional)
                # if symbol == "BTOG" and current_time.hour >= 12 and i % 10 == 0:  # Log every 10 minutes for BTOG after 12:00
                #     self.logger.debug(f"BTOG at {current_time}: price={current_price:.4f}, volume_ratio={volume_ratio:.2f}, breakout_type={breakout_signal.breakout_type.value if breakout_signal else 'None'}")
                
                if breakout_signal and breakout_signal.breakout_type.value != "no_breakout":
                    # Calculate technical indicators
                    technical_indicators = self.breakout_detector.calculate_technical_indicators(historical_data)
                    
                    # Calculate confidence score
                    confidence = self.confidence_scorer.calculate_confidence_score(
                        breakout_signal, technical_indicators
                    )
                    
                    self.logger.debug(f"Potential alert for {symbol}: confidence={confidence.total_score:.3f}, min_required={config.min_confidence_score}")
                    
                    # Check if confidence meets minimum threshold
                    if confidence.total_score >= config.min_confidence_score:
                        # Create alert
                        alert = self.alert_formatter.create_alert(
                            breakout_signal, confidence, technical_indicators
                        )
                        
                        alerts.append(alert)
                        self.generated_alerts += 1
                        
                        if alert.breakout_type.value == "bullish_breakout":
                            self.bullish_alerts += 1
                        else:
                            self.bearish_alerts += 1
                        
                        self.logger.info(f"Generated {alert.breakout_type.value} alert for {symbol} at {current_time}: ${current_price:.4f}")
                        
                        # Save alert to file if not dry run
                        if not self.dry_run:
                            self.alert_formatter.save_alert_to_file(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return alerts
    
    def _is_in_alert_window(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is within alert window (9:45 AM to 3:45 PM).
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if in alert window
        """
        try:
            time_only = timestamp.time()
            
            # Parse alert window times
            start_hour, start_minute = map(int, config.alert_window_start.split(':'))
            end_hour, end_minute = map(int, config.alert_window_end.split(':'))
            
            start_time = datetime.strptime(f"{start_hour:02d}:{start_minute:02d}", "%H:%M").time()
            end_time = datetime.strptime(f"{end_hour:02d}:{end_minute:02d}", "%H:%M").time()
            
            return start_time <= time_only <= end_time
            
        except Exception:
            return True  # Default to allowing alerts
    
    def process_all_symbols(self) -> None:
        """Process all symbols and generate alerts."""
        self.logger.info("Starting historical ORB alerts processing...")
        
        # Get latest market data files
        latest_files = self._get_latest_market_data_files()
        
        if not latest_files:
            self.logger.error("No market data files found")
            return
        
        # Process each symbol
        for symbol, file_path in latest_files.items():
                
            self.logger.info(f"Processing {symbol}...")
            
            # Load market data
            df = self._load_market_data(file_path)
            
            if df is None:
                continue
            
            # Clear data buffer for this symbol
            self.data_buffer.clear_symbol(symbol)
            
            # Populate data buffer
            self._populate_data_buffer(df)
            
            # Process for alerts
            alerts = self._process_symbol_for_alerts(symbol)
            
            self.processed_symbols += 1
            self.logger.info(f"Processed {symbol}: {len(alerts)} alerts generated")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print processing summary."""
        print("\n" + "="*60)
        print("HISTORICAL ORB ALERTS PROCESSING SUMMARY")
        print("="*60)
        print(f"Target Date: {self.target_date}")
        print(f"Processed Symbols: {self.processed_symbols}")
        print(f"Total Alerts Generated: {self.generated_alerts}")
        print(f"  • Bullish Alerts: {self.bullish_alerts}")
        print(f"  • Bearish Alerts: {self.bearish_alerts}")
        print(f"Dry Run: {self.dry_run}")
        
        if not self.dry_run:
            print(f"Alerts saved to: {self.alerts_dir}")
        else:
            print("No alerts were saved (dry run mode)")
        
        print("="*60)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process historical market data through ORB alerts logic"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default="2025-07-17",
        help="Target date to process (YYYY-MM-DD format, default: 2025-07-17)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save alerts to files, just process and show summary"
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
    
    try:
        processor = HistoricalORBProcessor(
            target_date=args.date,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        processor.process_all_symbols()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()