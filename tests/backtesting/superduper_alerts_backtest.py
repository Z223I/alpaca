#!/usr/bin/env python3
"""
Superduper Alerts Backtesting Framework

This framework provides comprehensive backtesting capabilities for the superduper alerts system.
It processes historical super alerts chronologically to generate superduper alerts with optional
Telegram notifications, supporting various analysis modes and configurations.

Usage Examples:
    # Basic backtest for specific symbol and date
    python3 tests/backtesting/superduper_alerts_backtest.py --symbol MCVT --date 2025-07-25

    # Backtest with custom settings
    python3 tests/backtesting/superduper_alerts_backtest.py --symbol AAPL --date 2025-07-23 --timeframe 60 --max-alerts 5

    # Dry run (no Telegram notifications)
    python3 tests/backtesting/superduper_alerts_backtest.py --symbol MCVT --date 2025-07-25 --dry-run

    # Batch backtest multiple symbols
    python3 tests/backtesting/superduper_alerts_backtest.py --date 2025-07-25 --batch-mode

    # Analysis only (no alert generation)
    python3 tests/backtesting/superduper_alerts_backtest.py --symbol MCVT --date 2025-07-25 --analysis-only
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import pytz

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from atoms.alerts.superduper_alert_filter import SuperduperAlertFilter
from atoms.alerts.superduper_alert_generator import SuperduperAlertGenerator
from atoms.telegram.orb_alerts import send_orb_alert


class SuperduperAlertsBacktester:
    """Comprehensive backtesting framework for superduper alerts."""
    
    def __init__(self, timeframe_minutes: int = 45, dry_run: bool = False):
        """
        Initialize the backtester.
        
        Args:
            timeframe_minutes: Analysis window in minutes
            dry_run: If True, skip Telegram notifications
        """
        self.timeframe_minutes = timeframe_minutes
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'symbols_processed': 0,
            'super_alerts_processed': 0,
            'superduper_alerts_created': 0,
            'telegram_notifications_sent': 0,
            'alerts_filtered': 0,
            'processing_errors': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging with Eastern Time."""
        class EasternFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                et_tz = pytz.timezone('US/Eastern')
                et_time = datetime.fromtimestamp(record.created, et_tz)
                if datefmt:
                    return et_time.strftime(datefmt)
                return et_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] + ' ET'
        
        logger = logging.getLogger(f"{__name__}_{id(self)}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = EasternFormatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _extract_timestamp_from_path(self, file_path: str) -> datetime:
        """Extract timestamp from super alert file path."""
        try:
            filename = Path(file_path).name
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 4:
                date_part = parts[-2]  # YYYYMMDD
                time_part = parts[-1]  # HHMMSS (possibly with extra digits)
                
                if len(date_part) == 8 and len(time_part) >= 6:
                    time_clean = time_part[:6]  # Take only HHMMSS
                    timestamp_str = f"{date_part}_{time_clean}"
                    return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except Exception as e:
            self.logger.warning(f"Error extracting timestamp from {file_path}: {e}")
        return None
    
    def _find_symbol_files(self, symbol: str, date_str: str) -> List[Tuple[datetime, Path]]:
        """Find and sort super alert files for a symbol on a specific date."""
        super_alerts_dir = Path(f"historical_data/{date_str}/super_alerts/bullish")
        
        if not super_alerts_dir.exists():
            self.logger.error(f"Super alerts directory not found: {super_alerts_dir}")
            return []
        
        # Find all super alert files for this symbol
        super_alert_files = list(super_alerts_dir.glob(f"super_alert_{symbol}_*.json"))
        
        # Sort by timestamp
        timestamped_files = []
        for file_path in super_alert_files:
            timestamp = self._extract_timestamp_from_path(str(file_path))
            if timestamp:
                timestamped_files.append((timestamp, file_path))
        
        timestamped_files.sort(key=lambda x: x[0])
        return timestamped_files
    
    def _find_all_symbols(self, date_str: str) -> List[str]:
        """Find all symbols with super alerts for a specific date."""
        super_alerts_dir = Path(f"historical_data/{date_str}/super_alerts/bullish")
        
        if not super_alerts_dir.exists():
            return []
        
        symbols = set()
        for file_path in super_alerts_dir.glob("super_alert_*.json"):
            parts = file_path.name.split('_')
            if len(parts) >= 3:
                symbol = parts[2]  # super_alert_SYMBOL_timestamp.json
                symbols.add(symbol)
        
        return sorted(list(symbols))
    
    def _determine_urgency(self, trend_type: str, trend_strength: float, analysis_data: Dict) -> bool:
        """Determine if superduper alert should be marked as urgent."""
        if trend_strength > 0.7:
            return True
        
        if trend_type == 'rising':
            momentum = analysis_data.get('price_momentum', 0)
            penetration_change = analysis_data.get('penetration_change', 0)
            
            if momentum > 0.05 and penetration_change > 15:
                return True
        
        elif trend_type == 'consolidating':
            avg_penetration = analysis_data.get('avg_penetration', 0)
            
            if avg_penetration > 40 and trend_strength > 0.5:
                return True
        
        return False
    
    def backtest_symbol(self, symbol: str, date_str: str, max_alerts: int = 10, 
                       analysis_only: bool = False) -> Dict[str, Any]:
        """
        Backtest superduper alerts for a specific symbol and date.
        
        Args:
            symbol: Stock symbol to analyze
            date_str: Date in YYYY-MM-DD format
            max_alerts: Maximum number of Telegram alerts to send
            analysis_only: If True, only perform analysis without creating alerts
            
        Returns:
            Dictionary with backtest results and statistics
        """
        self.logger.info(f"Starting backtest for {symbol} on {date_str}")
        
        # Setup directories
        superduper_alerts_dir = Path(f"historical_data/{date_str}/superduper_alerts/bullish")
        if not analysis_only:
            superduper_alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        filter_obj = SuperduperAlertFilter(self.timeframe_minutes)
        generator = SuperduperAlertGenerator(superduper_alerts_dir, test_mode=self.dry_run) if not analysis_only else None
        
        # Find super alert files
        timestamped_files = self._find_symbol_files(symbol, date_str)
        
        if not timestamped_files:
            self.logger.warning(f"No super alerts found for {symbol} on {date_str}")
            return {'symbol': symbol, 'date': date_str, 'files_found': 0}
        
        self.logger.info(f"Found {len(timestamped_files)} super alerts for {symbol}")
        
        # Process files chronologically
        results = {
            'symbol': symbol,
            'date': date_str,
            'files_found': len(timestamped_files),
            'files_processed': 0,
            'superduper_alerts_created': 0,
            'telegram_notifications_sent': 0,
            'alerts_filtered': 0,
            'processing_errors': 0,
            'trend_analysis': [],
            'price_progression': []
        }
        
        telegram_sent = 0
        
        for i, (timestamp, file_path) in enumerate(timestamped_files):
            if not self.dry_run and telegram_sent >= max_alerts:
                self.logger.info(f"Reached maximum of {max_alerts} Telegram alerts - stopping")
                break
            
            self.logger.info(f"Processing {i+1}/{len(timestamped_files)}: {file_path.name}")
            
            try:
                # Check if should create superduper alert
                should_create, reason, analysis_data = filter_obj.should_create_superduper_alert(str(file_path))
                
                results['files_processed'] += 1
                self.stats['super_alerts_processed'] += 1
                
                if not should_create:
                    self.logger.info(f"  ❌ Filtered: {reason}")
                    results['alerts_filtered'] += 1
                    self.stats['alerts_filtered'] += 1
                    continue
                
                # Load the super alert data
                with open(file_path, 'r') as f:
                    super_alert_data = json.load(f)
                
                # Get trend analysis
                symbol_data = filter_obj.symbol_data.get(symbol)
                if symbol_data:
                    trend_type, trend_strength, detailed_analysis = symbol_data.analyze_trend(self.timeframe_minutes)
                else:
                    self.logger.warning(f"  ⚠️ No symbol data available")
                    continue
                
                if trend_type in ['insufficient_data', 'declining']:
                    self.logger.info(f"  ❌ Invalid trend: {trend_type}")
                    continue
                
                # Store analysis data
                current_price = super_alert_data.get('signal_analysis', {}).get('current_price', 0)
                penetration = super_alert_data.get('signal_analysis', {}).get('penetration_percent', 0)
                
                results['trend_analysis'].append({
                    'timestamp': timestamp.isoformat(),
                    'trend_type': trend_type,
                    'trend_strength': trend_strength,
                    'current_price': current_price,
                    'penetration': penetration,
                    'analysis_data': detailed_analysis
                })
                
                results['price_progression'].append({
                    'timestamp': timestamp.isoformat(),
                    'price': current_price,
                    'penetration': penetration
                })
                
                if analysis_only:
                    self.logger.info(f"  📊 Analysis: {trend_type.upper()} trend, strength {trend_strength:.2f}")
                    continue
                
                # Create superduper alert
                filename = generator.create_and_save_superduper_alert(
                    super_alert_data, analysis_data, trend_type, trend_strength
                )
                
                if filename:
                    results['superduper_alerts_created'] += 1
                    self.stats['superduper_alerts_created'] += 1
                    self.logger.info(f"  ✅ Superduper alert created: {filename}")
                    
                    # Send Telegram notification (if not dry run)
                    if not self.dry_run:
                        try:
                            is_urgent = self._determine_urgency(trend_type, trend_strength, analysis_data)
                            
                            superduper_file_path = superduper_alerts_dir / filename
                            result = send_orb_alert(str(superduper_file_path), urgent=is_urgent, post_only_urgent=False)
                            
                            urgency_type = "URGENT" if is_urgent else "REGULAR"
                            if result['success'] and not result.get('skipped'):
                                telegram_sent += 1
                                results['telegram_notifications_sent'] += 1
                                self.stats['telegram_notifications_sent'] += 1
                                self.logger.info(f"  📤 Telegram sent ({urgency_type}): {result['sent_count']} users | Total sent: {telegram_sent}/{max_alerts}")
                            elif result.get('skipped'):
                                self.logger.info(f"  ⏭️ Telegram skipped: {result.get('reason')}")
                            else:
                                self.logger.warning(f"  ❌ Telegram failed: {result.get('error')}")
                                
                        except Exception as e:
                            self.logger.error(f"  ❌ Telegram error: {e}")
                else:
                    self.logger.warning(f"  ⚠️ Failed to create superduper alert")
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                results['processing_errors'] += 1
                self.stats['processing_errors'] += 1
        
        return results
    
    def batch_backtest(self, date_str: str, max_alerts_per_symbol: int = 5) -> Dict[str, Any]:
        """
        Backtest all symbols for a specific date.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            max_alerts_per_symbol: Maximum alerts per symbol
            
        Returns:
            Dictionary with comprehensive results
        """
        self.logger.info(f"Starting batch backtest for {date_str}")
        
        symbols = self._find_all_symbols(date_str)
        if not symbols:
            self.logger.error(f"No symbols found for {date_str}")
            return {'date': date_str, 'symbols_found': 0}
        
        self.logger.info(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
        
        batch_results = {
            'date': date_str,
            'symbols_found': len(symbols),
            'symbols_processed': 0,
            'symbol_results': {}
        }
        
        for symbol in symbols:
            self.logger.info(f"Processing symbol {symbol}")
            symbol_results = self.backtest_symbol(symbol, date_str, max_alerts_per_symbol)
            batch_results['symbol_results'][symbol] = symbol_results
            batch_results['symbols_processed'] += 1
            self.stats['symbols_processed'] += 1
        
        return batch_results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of backtest results."""
        print("\\n" + "="*80)
        print("📊 BACKTEST SUMMARY")
        print("="*80)
        
        if 'symbol_results' in results:
            # Batch results
            print(f"📅 Date: {results['date']}")
            print(f"🔍 Symbols found: {results['symbols_found']}")
            print(f"⚙️ Symbols processed: {results['symbols_processed']}")
            print("\\n📈 Per-Symbol Results:")
            
            for symbol, symbol_result in results['symbol_results'].items():
                print(f"  {symbol}:")
                print(f"    📁 Super alerts: {symbol_result.get('files_found', 0)}")
                print(f"    🎯 Superduper alerts: {symbol_result.get('superduper_alerts_created', 0)}")
                print(f"    📱 Telegram sent: {symbol_result.get('telegram_notifications_sent', 0)}")
        else:
            # Single symbol results
            print(f"📅 Date: {results['date']}")
            print(f"🔍 Symbol: {results['symbol']}")
            print(f"📁 Super alerts found: {results['files_found']}")
            print(f"⚙️ Files processed: {results['files_processed']}")
            print(f"🎯 Superduper alerts created: {results['superduper_alerts_created']}")
            print(f"📱 Telegram notifications sent: {results['telegram_notifications_sent']}")
            print(f"🚫 Alerts filtered: {results['alerts_filtered']}")
            
            if results.get('price_progression'):
                prices = [p['price'] for p in results['price_progression']]
                penetrations = [p['penetration'] for p in results['price_progression']]
                print(f"💰 Price range: ${min(prices):.4f} → ${max(prices):.4f}")
                print(f"🎯 Penetration range: {min(penetrations):.1f}% → {max(penetrations):.1f}%")
        
        print("\\n🔧 Overall Statistics:")
        print(f"  Symbols processed: {self.stats['symbols_processed']}")
        print(f"  Super alerts processed: {self.stats['super_alerts_processed']}")
        print(f"  Superduper alerts created: {self.stats['superduper_alerts_created']}")
        print(f"  Telegram notifications sent: {self.stats['telegram_notifications_sent']}")
        print(f"  Alerts filtered: {self.stats['alerts_filtered']}")
        print(f"  Processing errors: {self.stats['processing_errors']}")
        print("="*80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Superduper Alerts Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        help="Stock symbol to backtest (required unless --batch-mode)"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to backtest in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--timeframe",
        type=int,
        default=45,
        help="Analysis timeframe in minutes (default: 45)"
    )
    
    parser.add_argument(
        "--max-alerts",
        type=int,
        default=10,
        help="Maximum Telegram alerts to send (default: 10)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Telegram notifications (dry run mode)"
    )
    
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Backtest all symbols for the specified date"
    )
    
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only perform trend analysis without creating alerts"
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
    
    # Validate arguments
    if not args.batch_mode and not args.symbol:
        print("Error: --symbol is required unless using --batch-mode")
        return 1
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize backtester
    backtester = SuperduperAlertsBacktester(
        timeframe_minutes=args.timeframe,
        dry_run=args.dry_run or args.analysis_only
    )
    
    try:
        if args.batch_mode:
            # Batch backtest all symbols
            results = backtester.batch_backtest(args.date, args.max_alerts)
        else:
            # Single symbol backtest
            results = backtester.backtest_symbol(
                args.symbol, args.date, args.max_alerts, args.analysis_only
            )
        
        # Print summary
        backtester.print_summary(results)
        
        return 0
        
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())