#!/usr/bin/env python3
"""
ORB Alerts Backtesting Framework

This comprehensive backtesting system validates the ORB alerts generation logic
using historical market data. It can replay market data chronologically to simulate
real-time alert generation and compare results with actual alerts that were generated.

Usage Examples:
    # Backtest specific symbol and date
    python3 tests/backtesting/alerts_backtest.py --symbol AAPL --date 2025-07-23

    # Backtest with verbose output
    python3 tests/backtesting/alerts_backtest.py --symbol MCVT --date 2025-07-25 --verbose

    # Compare mode - validate against existing alerts
    python3 tests/backtesting/alerts_backtest.py --symbol AAPL --date 2025-07-23 --compare

    # Analysis only mode - no alert generation
    python3 tests/backtesting/alerts_backtest.py --symbol AAPL --date 2025-07-23 --analysis-only

    # Fixed opening range period (15 minutes)
    python3 tests/backtesting/alerts_backtest.py --symbol AAPL --date 2025-07-23
"""

import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pytz

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from atoms.indicators.orb_calculator import ORBCalculator
from atoms.alerts.breakout_detector import BreakoutDetector
from atoms.alerts.confidence_scorer import ConfidenceScorer
from atoms.alerts.alert_formatter import AlertFormatter, ORBAlert
from atoms.websocket.data_buffer import DataBuffer
from atoms.websocket.alpaca_stream import MarketData


class ORBAlertsBacktester:
    """Comprehensive backtesting framework for ORB alerts system."""
    
    def __init__(self, symbol: str, date_str: str, verbose: bool = False):
        """
        Initialize the backtester.
        
        Args:
            symbol: Stock symbol to backtest
            date_str: Date in YYYY-MM-DD format
            verbose: Enable detailed logging
        """
        self.symbol = symbol.upper()
        self.date_str = date_str
        self.verbose = verbose
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components - use default 15 minute ORB period
        self.orb_calculator = ORBCalculator(15)
        self.breakout_detector = BreakoutDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.alert_formatter = AlertFormatter()
        self.data_buffer = DataBuffer()
        
        # Paths
        self.historical_data_dir = Path(f"historical_data/{date_str}")
        self.market_data_dir = self.historical_data_dir / "market_data"
        self.alerts_dir = self.historical_data_dir / "alerts"
        
        # Results tracking
        self.results = {
            'symbol': symbol,
            'date': date_str,
            'market_data_loaded': False,
            'orb_calculated': False,
            'alerts_generated': [],
            'existing_alerts': [],
            'comparison_results': {},
            'statistics': {}
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
            logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        return logger
    
    def load_historical_market_data(self) -> Optional[pd.DataFrame]:
        """Load historical market data for the symbol and date."""
        try:
            if not self.market_data_dir.exists():
                self.logger.error(f"Market data directory not found: {self.market_data_dir}")
                return None
            
            # Find market data files for this symbol
            pattern = f"{self.symbol}_*.csv"
            data_files = list(self.market_data_dir.glob(pattern))
            
            if not data_files:
                self.logger.error(f"No market data files found for {self.symbol} on {self.date_str}")
                return None
            
            # Load and combine all data files (sorted by timestamp)
            all_data = []
            for file_path in sorted(data_files):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        all_data.append(df)
                        self.logger.debug(f"Loaded {len(df)} records from {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {e}")
            
            if not all_data:
                self.logger.error(f"No valid market data loaded for {self.symbol}")
                return None
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Ensure timestamp column exists and is properly formatted
            if 'timestamp' not in combined_data.columns:
                self.logger.error("Market data missing 'timestamp' column")
                return None
            
            # Convert timestamp to datetime
            combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
            
            # Sort by timestamp
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            combined_data = combined_data.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(combined_data)} records for {self.symbol} on {self.date_str}")
            self.results['market_data_loaded'] = True
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error loading historical market data: {e}")
            return None
    
    def simulate_real_time_processing(self, market_data: pd.DataFrame) -> List[ORBAlert]:
        """Simulate real-time market data processing to generate alerts."""
        alerts_generated = []
        
        try:
            et_tz = pytz.timezone('US/Eastern')
            
            # Market open and ORB end times (15 minute ORB period)
            market_open = time(9, 30)  # 9:30 AM ET
            orb_end_time = time(9, 45)  # 9:45 AM ET (15 minutes)
            market_close = time(16, 0)  # 4:00 PM ET
            
            orb_data = None
            orb_calculated = False
            
            self.logger.info(f"Starting simulation with {len(market_data)} data points")
            self.logger.info(f"ORB period: {market_open} to {orb_end_time} ET (15 minutes)")
            
            # Process data chronologically
            for idx, row in market_data.iterrows():
                # Convert pandas Timestamp to datetime if needed
                current_time = row['timestamp']
                if hasattr(current_time, 'to_pydatetime'):
                    current_time = current_time.to_pydatetime()
                
                # Convert to Eastern Time if needed
                if current_time.tzinfo is None:
                    current_time = et_tz.localize(current_time)
                else:
                    current_time = current_time.astimezone(et_tz)
                
                current_time_only = current_time.time()
                
                # Skip pre-market data
                if current_time_only < market_open:
                    continue
                
                # Skip after-hours data
                if current_time_only > market_close:
                    break
                
                # Create MarketData object for this tick
                market_tick = MarketData(
                    symbol=self.symbol,
                    timestamp=current_time,
                    price=float(row.get('close', row.get('price', 0))),
                    volume=int(row.get('volume', 0)),
                    high=float(row.get('high', row.get('price', 0))),
                    low=float(row.get('low', row.get('price', 0))),
                    close=float(row.get('close', row.get('price', 0))),
                    trade_count=int(row.get('trade_count', 0)),
                    vwap=float(row.get('vwap', row.get('price', 0))),
                    open=float(row.get('open', row.get('price', 0)))
                )
                
                # Add to data buffer
                self.data_buffer.add_market_data(market_tick)
                
                # Calculate ORB after ORB period ends
                if not orb_calculated and current_time_only >= orb_end_time:
                    symbol_data = self.data_buffer.get_symbol_data(self.symbol)
                    if symbol_data is not None and len(symbol_data) > 0:
                        try:
                            # The calculate_orb_levels method expects symbol and price_data
                            orb_data = self.orb_calculator.calculate_orb_levels(self.symbol, symbol_data)
                            if orb_data:
                                orb_calculated = True
                                self.results['orb_calculated'] = True
                                self.logger.info(f"ORB calculated: High={orb_data.orb_high:.4f}, Low={orb_data.orb_low:.4f}, Range={orb_data.orb_range:.4f}")
                        except Exception as e:
                            self.logger.error(f"Error calculating ORB: {e}")
                            self.logger.debug(f"Symbol data columns: {list(symbol_data.columns) if hasattr(symbol_data, 'columns') else 'no columns'}")
                            # Only calculate once after ORB period ends
                            orb_calculated = True  # Prevent repeated attempts
                
                # Check for breakouts after ORB is calculated
                if orb_calculated and orb_data:
                    # Calculate volume ratio
                    volume_ratio = self._calculate_volume_ratio(market_tick)
                    
                    # Check for breakout
                    breakout_signal = self.breakout_detector.detect_breakout(
                        symbol=self.symbol,
                        current_price=market_tick.price,
                        volume_ratio=volume_ratio,
                        timestamp=current_time
                    )
                    
                    if breakout_signal and breakout_signal.signal_type != 'none':
                        # Calculate confidence
                        confidence_components = self.confidence_scorer.calculate_confidence(
                            breakout_signal=breakout_signal,
                            current_data=market_tick,
                            orb_data=orb_data,
                            symbol_data=symbol_data
                        )
                        
                        # Format alert
                        orb_alert = self.alert_formatter.format_alert(
                            breakout_signal=breakout_signal,
                            confidence_components=confidence_components,
                            orb_data=orb_data,
                            current_data=market_tick
                        )
                        
                        if orb_alert:
                            alerts_generated.append(orb_alert)
                            self.logger.info(f"Alert generated at {current_time}: {breakout_signal.signal_type} breakout, confidence={confidence_components.total_confidence:.3f}")
            
            self.logger.info(f"Simulation complete. Generated {len(alerts_generated)} alerts")
            return alerts_generated
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {e}")
            return alerts_generated
    
    def _calculate_volume_ratio(self, market_tick) -> float:
        """
        Calculate volume ratio vs average volume.
        
        Args:
            market_tick: Current market data tick
            
        Returns:
            Volume ratio (current / average), defaults to 1.0 if unavailable
        """
        try:
            # Get average volume from recent data
            symbol_data = self.data_buffer.get_symbol_data(self.symbol)
            if symbol_data is None or symbol_data.empty:
                return 1.0
            
            # Calculate average volume from recent data (last 20 records)
            recent_data = symbol_data.tail(20) if len(symbol_data) > 20 else symbol_data
            avg_volume = recent_data['volume'].mean()
            
            if avg_volume == 0:
                return 1.0
            
            return market_tick.volume / avg_volume
            
        except Exception:
            return 1.0
    
    def load_existing_alerts(self) -> List[Dict[str, Any]]:
        """Load existing alerts that were generated for this symbol and date."""
        existing_alerts = []
        
        try:
            if not self.alerts_dir.exists():
                self.logger.warning(f"Alerts directory not found: {self.alerts_dir}")
                return existing_alerts
            
            # Search in both bullish and bearish directories
            for alert_type in ['bullish', 'bearish']:
                alert_type_dir = self.alerts_dir / alert_type
                if alert_type_dir.exists():
                    # Find alert files for this symbol
                    pattern = f"alert_{self.symbol}_*.json"
                    alert_files = list(alert_type_dir.glob(pattern))
                    
                    for alert_file in alert_files:
                        try:
                            with open(alert_file, 'r') as f:
                                alert_data = json.load(f)
                                alert_data['file_path'] = str(alert_file)
                                alert_data['alert_type'] = alert_type
                                existing_alerts.append(alert_data)
                        except Exception as e:
                            self.logger.warning(f"Error loading {alert_file}: {e}")
            
            self.logger.info(f"Loaded {len(existing_alerts)} existing alerts for {self.symbol}")
            return existing_alerts
            
        except Exception as e:
            self.logger.error(f"Error loading existing alerts: {e}")
            return existing_alerts
    
    def compare_results(self, generated_alerts: List[ORBAlert], existing_alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare generated alerts with existing alerts."""
        comparison = {
            'generated_count': len(generated_alerts),
            'existing_count': len(existing_alerts),
            'matches': [],
            'missing_in_generated': [],
            'extra_in_generated': [],
            'accuracy_metrics': {}
        }
        
        try:
            # Convert generated alerts to comparable format
            generated_dict = {}
            for alert in generated_alerts:
                # Create a key based on timestamp and breakout type
                timestamp_key = alert.timestamp.strftime('%H:%M:%S')
                key = f"{timestamp_key}_{alert.breakout_type}"
                generated_dict[key] = alert
            
            # Convert existing alerts to comparable format
            existing_dict = {}
            for alert in existing_alerts:
                timestamp_str = alert.get('timestamp', '')
                breakout_type = alert.get('breakout_type', '')
                if timestamp_str and breakout_type:
                    try:
                        # Extract time from timestamp
                        if 'T' in timestamp_str:
                            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            timestamp_key = dt.strftime('%H:%M:%S')
                        else:
                            timestamp_key = timestamp_str
                        
                        key = f"{timestamp_key}_{breakout_type}"
                        existing_dict[key] = alert
                    except Exception as e:
                        self.logger.warning(f"Error parsing existing alert timestamp: {e}")
            
            # Find matches
            for key in generated_dict:
                if key in existing_dict:
                    comparison['matches'].append({
                        'key': key,
                        'generated': generated_dict[key],
                        'existing': existing_dict[key]
                    })
            
            # Find missing in generated (existing alerts not reproduced)
            for key in existing_dict:
                if key not in generated_dict:
                    comparison['missing_in_generated'].append({
                        'key': key,
                        'existing': existing_dict[key]
                    })
            
            # Find extra in generated (new alerts not in existing)
            for key in generated_dict:
                if key not in existing_dict:
                    comparison['extra_in_generated'].append({
                        'key': key,
                        'generated': generated_dict[key]
                    })
            
            # Calculate accuracy metrics
            if len(existing_alerts) > 0:
                accuracy = len(comparison['matches']) / len(existing_alerts)
                comparison['accuracy_metrics']['recall'] = accuracy
            
            if len(generated_alerts) > 0:
                precision = len(comparison['matches']) / len(generated_alerts)
                comparison['accuracy_metrics']['precision'] = precision
            
            if 'recall' in comparison['accuracy_metrics'] and 'precision' in comparison['accuracy_metrics']:
                recall = comparison['accuracy_metrics']['recall']
                precision = comparison['accuracy_metrics']['precision']
                if recall + precision > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    comparison['accuracy_metrics']['f1_score'] = f1_score
            
            self.logger.info(f"Comparison complete: {len(comparison['matches'])} matches, {len(comparison['missing_in_generated'])} missing, {len(comparison['extra_in_generated'])} extra")
            
        except Exception as e:
            self.logger.error(f"Error in comparison: {e}")
        
        return comparison
    
    def generate_statistics(self, generated_alerts: List[ORBAlert], existing_alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistics from the backtest."""
        stats = {
            'alert_counts': {
                'generated': len(generated_alerts),
                'existing': len(existing_alerts)
            },
            'breakout_types': {
                'generated': {},
                'existing': {}
            },
            'confidence_stats': {
                'generated': {},
                'existing': {}
            },
            'timing_analysis': {},
            'performance_metrics': {}
        }
        
        try:
            # Analyze generated alerts
            if generated_alerts:
                breakout_types = [alert.breakout_type for alert in generated_alerts]
                for bt in set(breakout_types):
                    stats['breakout_types']['generated'][bt] = breakout_types.count(bt)
                
                confidences = [alert.confidence_score for alert in generated_alerts]
                stats['confidence_stats']['generated'] = {
                    'mean': sum(confidences) / len(confidences),
                    'min': min(confidences),
                    'max': max(confidences),
                    'count': len(confidences)
                }
            
            # Analyze existing alerts
            if existing_alerts:
                breakout_types = [alert.get('breakout_type', 'unknown') for alert in existing_alerts]
                for bt in set(breakout_types):
                    stats['breakout_types']['existing'][bt] = breakout_types.count(bt)
                
                confidences = [alert.get('confidence_score', 0) for alert in existing_alerts if alert.get('confidence_score')]
                if confidences:
                    stats['confidence_stats']['existing'] = {
                        'mean': sum(confidences) / len(confidences),
                        'min': min(confidences),
                        'max': max(confidences),
                        'count': len(confidences)
                    }
            
        except Exception as e:
            self.logger.error(f"Error generating statistics: {e}")
        
        return stats
    
    def run_backtest(self, compare_mode: bool = False, analysis_only: bool = False) -> Dict[str, Any]:
        """Run the complete backtest."""
        self.logger.info(f"Starting backtest for {self.symbol} on {self.date_str}")
        
        try:
            # Load historical market data
            market_data = self.load_historical_market_data()
            if market_data is None:
                self.results['error'] = "Failed to load market data"
                return self.results
            
            # Load existing alerts if in compare mode
            existing_alerts = []
            if compare_mode:
                existing_alerts = self.load_existing_alerts()
                self.results['existing_alerts'] = existing_alerts
            
            # Generate alerts from historical data (unless analysis only)
            generated_alerts = []
            if not analysis_only:
                generated_alerts = self.simulate_real_time_processing(market_data)
                self.results['alerts_generated'] = [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'breakout_type': alert.breakout_type,
                        'confidence_score': alert.confidence_score,
                        'current_price': alert.current_price,
                        'orb_high': alert.orb_high,
                        'orb_low': alert.orb_low
                    }
                    for alert in generated_alerts
                ]
            
            # Compare results if in compare mode
            if compare_mode and not analysis_only:
                comparison_results = self.compare_results(generated_alerts, existing_alerts)
                self.results['comparison_results'] = comparison_results
            
            # Generate statistics
            statistics = self.generate_statistics(generated_alerts, existing_alerts)
            self.results['statistics'] = statistics
            
            self.logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            self.results['error'] = str(e)
            return self.results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted backtest results."""
        print("\\n" + "="*80)
        print("📊 ORB ALERTS BACKTEST RESULTS")
        print("="*80)
        
        print(f"🔍 Symbol: {results['symbol']}")
        print(f"📅 Date: {results['date']}")
        print(f"⏱️ ORB Period: 15 minutes")
        print(f"📊 Market Data Loaded: {'✅' if results['market_data_loaded'] else '❌'}")
        print(f"📈 ORB Calculated: {'✅' if results['orb_calculated'] else '❌'}")
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            print("="*80)
            return
        
        # Alert counts
        generated_count = len(results.get('alerts_generated', []))
        existing_count = len(results.get('existing_alerts', []))
        
        print(f"\\n🚨 Alert Generation:")
        print(f"  Generated Alerts: {generated_count}")
        print(f"  Existing Alerts: {existing_count}")
        
        # Statistics
        if 'statistics' in results:
            stats = results['statistics']
            
            print(f"\\n📈 Breakout Types:")
            if 'generated' in stats['breakout_types']:
                print("  Generated:")
                for bt, count in stats['breakout_types']['generated'].items():
                    print(f"    {bt}: {count}")
            
            if 'existing' in stats['breakout_types']:
                print("  Existing:")
                for bt, count in stats['breakout_types']['existing'].items():
                    print(f"    {bt}: {count}")
            
            print(f"\\n🎯 Confidence Statistics:")
            if 'generated' in stats['confidence_stats'] and stats['confidence_stats']['generated']:
                gen_conf = stats['confidence_stats']['generated']
                print(f"  Generated: Mean={gen_conf['mean']:.3f}, Min={gen_conf['min']:.3f}, Max={gen_conf['max']:.3f}")
            
            if 'existing' in stats['confidence_stats'] and stats['confidence_stats']['existing']:
                ex_conf = stats['confidence_stats']['existing']
                print(f"  Existing: Mean={ex_conf['mean']:.3f}, Min={ex_conf['min']:.3f}, Max={ex_conf['max']:.3f}")
        
        # Comparison results
        if 'comparison_results' in results and results['comparison_results']:
            comp = results['comparison_results']
            print(f"\\n🔍 Comparison Results:")
            print(f"  Matches: {len(comp.get('matches', []))}")
            print(f"  Missing in Generated: {len(comp.get('missing_in_generated', []))}")
            print(f"  Extra in Generated: {len(comp.get('extra_in_generated', []))}")
            
            if 'accuracy_metrics' in comp and comp['accuracy_metrics']:
                metrics = comp['accuracy_metrics']
                if 'recall' in metrics:
                    print(f"  Recall: {metrics['recall']:.3f}")
                if 'precision' in metrics:
                    print(f"  Precision: {metrics['precision']:.3f}")
                if 'f1_score' in metrics:
                    print(f"  F1 Score: {metrics['f1_score']:.3f}")
        
        print("="*80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ORB Alerts Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol to backtest"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to backtest in YYYY-MM-DD format"
    )
    
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare generated alerts with existing alerts"
    )
    
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only analyze existing data without generating new alerts"
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
        # Initialize backtester
        backtester = ORBAlertsBacktester(
            symbol=args.symbol,
            date_str=args.date,
            verbose=args.verbose
        )
        
        # Run backtest
        results = backtester.run_backtest(
            compare_mode=args.compare,
            analysis_only=args.analysis_only
        )
        
        # Print results
        backtester.print_results(results)
        
        return 0 if 'error' not in results else 1
        
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())