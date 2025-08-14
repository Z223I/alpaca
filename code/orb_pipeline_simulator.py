#!/usr/bin/env python3
"""
ORB Pipeline Simulator

Simulates real-time ORB alerts processing by feeding historical market data
minute-by-minute through the existing ORB alert engine. This allows for
backtesting and analysis of ORB strategies using historical data.

Usage:
    python3 code/orb_pipeline_simulator.py --symbol AAPL --date 2025-08-13
    python3 code/orb_pipeline_simulator.py --symbol AAPL --date 2025-08-13 --speed 10
    python3 code/orb_pipeline_simulator.py --symbols-file symbols.csv --date 2025-08-13
    python3 code/orb_pipeline_simulator.py --symbol AAPL --date 2025-08-13 --save-alerts
"""

import sys
import asyncio
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
import pytz
import json
import time as time_module

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from atoms.config.alert_config import config
from atoms.alerts.config import get_data_root_dir, get_historical_root_dir, get_logs_root_dir
from molecules.orb_alert_engine import ORBAlertEngine
from atoms.alerts.alert_formatter import ORBAlert
from atoms.websocket.data_buffer import DataBuffer
from atoms.websocket.alpaca_stream import MarketData

# Alpaca API imports for historical data
try:
    from alpaca_trade_api.rest import REST
    ALPACA_AVAILABLE = "legacy"
except ImportError:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame
        ALPACA_AVAILABLE = True
    except ImportError:
        ALPACA_AVAILABLE = False


class ORBPipelineSimulator:
    """Simulates real-time ORB alerts processing using historical data."""

    def __init__(self, 
                 symbol: Optional[str] = None,
                 symbols_file: Optional[str] = None,
                 date: str = None,
                 speed: float = 1.0,
                 save_alerts: bool = False,
                 verbose: bool = False):
        """
        Initialize the ORB pipeline simulator.

        Args:
            symbol: Single symbol to simulate (e.g., 'AAPL')
            symbols_file: Path to CSV file with symbols
            date: Date to simulate in YYYY-MM-DD format
            speed: Playback speed multiplier (1.0 = real-time, 10.0 = 10x faster)
            save_alerts: Save generated alerts to files
            verbose: Enable verbose logging
        """
        self.symbol = symbol.upper() if symbol else None
        self.symbols_file = symbols_file
        self.date = date
        self.speed = speed
        self.save_alerts = save_alerts
        self.verbose = verbose

        # Setup logging
        self.logger = self._setup_logging()

        # Validate inputs
        if not symbol and not symbols_file:
            raise ValueError("Must provide either --symbol or --symbols-file")

        if symbol and symbols_file:
            raise ValueError("Cannot specify both --symbol and --symbols-file")

        # Determine symbols to process
        self.symbols = self._get_symbols()
        self.logger.info(f"Will simulate {len(self.symbols)} symbols: {', '.join(self.symbols[:5])}")
        if len(self.symbols) > 5:
            self.logger.info(f"... and {len(self.symbols) - 5} more")

        # Initialize data buffer
        self.data_buffer = DataBuffer()

        # Initialize directory configurations
        self.historical_config = get_historical_root_dir()
        self.logs_config = get_logs_root_dir()

        # Initialize historical data client
        self.historical_client = self._initialize_historical_client()

        # Results tracking
        self.simulation_results = {
            'date': date,
            'symbols': self.symbols,
            'speed': speed,
            'alerts_generated': [],
            'statistics': {},
            'start_time': None,
            'end_time': None
        }

        # Alert callback for collecting results
        self.generated_alerts = []

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
            formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        return logger

    def _get_symbols(self) -> List[str]:
        """Get list of symbols to simulate."""
        if self.symbol:
            return [self.symbol]
        
        if self.symbols_file:
            symbols_path = Path(self.symbols_file)
            if not symbols_path.exists():
                # Try relative to data directory
                data_config = get_data_root_dir()
                symbols_path = data_config.get_symbols_file_path(self.date)
                
            if not symbols_path.exists():
                raise FileNotFoundError(f"Symbols file not found: {self.symbols_file}")
            
            # Read symbols from CSV
            df = pd.read_csv(symbols_path)
            if 'symbol' in df.columns:
                return df['symbol'].tolist()
            elif 'Symbol' in df.columns:
                return df['Symbol'].tolist()
            else:
                # Assume first column contains symbols
                return df.iloc[:, 0].tolist()

    def _initialize_historical_client(self):
        """Initialize Alpaca historical data client."""
        if not ALPACA_AVAILABLE:
            self.logger.warning("Alpaca API not available - will try to load from CSV files")
            return None

        try:
            if ALPACA_AVAILABLE == "legacy":
                client = REST(
                    key_id=config.api_key,
                    secret_key=config.secret_key,
                    base_url=config.base_url
                )
                self.logger.info("Using legacy alpaca-trade-api for historical data")
                return client
            else:
                client = StockHistoricalDataClient(
                    api_key=config.api_key,
                    secret_key=config.secret_key
                )
                self.logger.info("Using new Alpaca API for historical data")
                return client
        except Exception as e:
            self.logger.warning(f"Could not initialize historical data client: {e}")
            return None

    def _load_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical data for a symbol."""
        # First try to load from existing CSV files
        csv_data = self._load_from_csv(symbol)
        if csv_data is not None and not csv_data.empty:
            self.logger.debug(f"Loaded {len(csv_data)} records from CSV for {symbol}")
            return csv_data

        # If no CSV data, try to fetch from API
        if self.historical_client:
            api_data = self._fetch_from_api(symbol)
            if api_data is not None and not api_data.empty:
                self.logger.debug(f"Fetched {len(api_data)} records from API for {symbol}")
                return api_data

        self.logger.warning(f"No historical data available for {symbol} on {self.date}")
        return None

    def _load_from_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from existing CSV files."""
        try:
            # Use centralized historical data configuration
            historical_date_dir = self.historical_config.get_historical_data_path(self.date)
            data_dir = historical_date_dir / "market_data"
            if not data_dir.exists():
                return None

            # Find CSV files for this symbol
            csv_files = list(data_dir.glob(f"{symbol}_*.csv"))
            if not csv_files:
                return None

            # Use the most recent file
            latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
            
            df = pd.read_csv(latest_file)
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            return df

        except Exception as e:
            self.logger.debug(f"Error loading CSV data for {symbol}: {e}")
            return None

    def _fetch_from_api(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca API."""
        try:
            et_tz = pytz.timezone('US/Eastern')
            
            # Parse target date
            target_date = datetime.strptime(self.date, '%Y-%m-%d')
            target_date = et_tz.localize(target_date)
            
            # Market hours: 9:00 AM to 4:00 PM ET
            start_time = target_date.replace(hour=9, minute=0, second=0, microsecond=0)
            end_time = target_date.replace(hour=16, minute=0, second=0, microsecond=0)

            if ALPACA_AVAILABLE == "legacy":
                # Convert to UTC for API
                start_utc = start_time.astimezone(pytz.UTC)
                end_utc = end_time.astimezone(pytz.UTC)
                start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

                bars = self.historical_client.get_bars(
                    symbol,
                    '1Min',
                    start=start_str,
                    end=end_str,
                    limit=1000,
                    feed='sip'
                )

                if bars:
                    data = []
                    for bar in bars:
                        data.append({
                            'timestamp': bar.t,
                            'symbol': symbol,
                            'open': float(bar.o),
                            'high': float(bar.h),
                            'low': float(bar.l),
                            'close': float(bar.c),
                            'volume': int(bar.v),
                            'trade_count': getattr(bar, 'n', 1),
                            'vwap': getattr(bar, 'vw', float(bar.c))
                        })
                    
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df.sort_values('timestamp')

        except Exception as e:
            self.logger.error(f"Error fetching API data for {symbol}: {e}")

        return None

    def _alert_callback(self, alert: ORBAlert) -> None:
        """Callback to handle generated alerts."""
        self.generated_alerts.append(alert)
        
        # Print alert with simulation timestamp
        print(f"🚨 ALERT: {alert.alert_message}")
        
        if self.save_alerts:
            self._save_alert(alert)

    def _save_alert(self, alert: ORBAlert) -> None:
        """Save alert to file."""
        try:
            # Create alerts directory using logs config pattern
            logs_base = self.logs_config.get_logs_path()
            alerts_dir = logs_base / self.date / "alerts"
            alerts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save alert as JSON
            alert_filename = f"sim_alert_{alert.symbol}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            alert_filepath = alerts_dir / alert_filename
            
            alert_data = {
                'symbol': alert.symbol,
                'timestamp': alert.timestamp.isoformat(),
                'current_price': float(alert.current_price),
                'orb_high': float(alert.orb_high),
                'orb_low': float(alert.orb_low),
                'breakout_type': alert.breakout_type.value,
                'confidence_score': float(alert.confidence_score),
                'alert_message': alert.alert_message,
                'simulation_metadata': {
                    'speed': self.speed,
                    'source': 'pipeline_simulator'
                }
            }
            
            with open(alert_filepath, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
            self.logger.debug(f"Saved alert to {alert_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving alert: {e}")

    async def simulate_symbol(self, symbol: str) -> Dict[str, Any]:
        """Simulate ORB processing for a single symbol."""
        self.logger.info(f"Starting simulation for {symbol}")
        
        # Load historical data
        market_data = self._load_historical_data(symbol)
        if market_data is None or market_data.empty:
            self.logger.warning(f"No data available for {symbol}")
            return {'symbol': symbol, 'status': 'no_data', 'alerts': []}

        # Initialize ORB alert engine for this symbol
        # Create a temporary symbols file for this single symbol
        temp_symbols_file = Path(f"tmp/sim_symbols_{symbol}.csv")
        temp_symbols_file.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame({'symbol': [symbol]}).to_csv(temp_symbols_file, index=False)
        
        alert_engine = ORBAlertEngine(str(temp_symbols_file))
        alert_engine.add_alert_callback(self._alert_callback)

        # Reset alerts for this symbol
        symbol_alerts = []
        
        et_tz = pytz.timezone('US/Eastern')
        total_records = len(market_data)
        processed_records = 0
        
        self.logger.info(f"Processing {total_records} data points for {symbol}")
        print(f"📊 Simulating {symbol} on {self.date} (Speed: {self.speed}x)")
        
        # Process data chronologically
        for idx, row in market_data.iterrows():
            try:
                # Convert to MarketData object
                timestamp = row['timestamp']
                if hasattr(timestamp, 'to_pydatetime'):
                    timestamp = timestamp.to_pydatetime()
                
                # Ensure timezone-naive Eastern Time
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.astimezone(et_tz).replace(tzinfo=None)
                
                market_data_obj = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=row['close'],
                    volume=row['volume'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    trade_count=row.get('trade_count', 1),
                    vwap=row.get('vwap', row['close']),
                    open=row.get('open', row['close'])
                )
                
                # Add to data buffer
                alert_engine.data_buffer.add_market_data(market_data_obj)
                
                # Process through ORB engine (this may generate alerts)
                alert_engine._process_market_data_optimized(market_data_obj)
                
                processed_records += 1
                
                # Apply speed control (simulate real-time delays)
                if self.speed < 100:  # Don't sleep for very fast speeds
                    sleep_time = (60.0 / self.speed) / 1000  # 1 minute = 60 seconds, scaled by speed
                    await asyncio.sleep(sleep_time)
                
                # Progress indicator
                if processed_records % 50 == 0 or processed_records == total_records:
                    progress = (processed_records / total_records) * 100
                    print(f"⏳ Progress: {progress:.1f}% ({processed_records}/{total_records})")
                    
            except Exception as e:
                self.logger.error(f"Error processing record {idx} for {symbol}: {e}")
                continue

        # Clean up temp file
        try:
            temp_symbols_file.unlink()
        except:
            pass

        # Collect alerts generated for this symbol
        symbol_alerts = [alert for alert in self.generated_alerts if alert.symbol == symbol]
        
        result = {
            'symbol': symbol,
            'status': 'completed',
            'records_processed': processed_records,
            'alerts_generated': len(symbol_alerts),
            'alerts': symbol_alerts
        }
        
        self.logger.info(f"Completed {symbol}: {processed_records} records, {len(symbol_alerts)} alerts")
        print(f"✅ {symbol}: {len(symbol_alerts)} alerts generated from {processed_records} data points")
        
        return result

    async def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        print(f"🚀 Starting ORB Pipeline Simulation")
        print(f"📅 Date: {self.date}")
        print(f"🎯 Symbols: {len(self.symbols)}")
        print(f"⚡ Speed: {self.speed}x")
        print(f"💾 Save alerts: {'Yes' if self.save_alerts else 'No'}")
        print("-" * 50)
        
        start_time = datetime.now()
        self.simulation_results['start_time'] = start_time.isoformat()
        
        # Process each symbol
        results = []
        for symbol in self.symbols:
            try:
                result = await self.simulate_symbol(symbol)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to simulate {symbol}: {e}")
                results.append({'symbol': symbol, 'status': 'error', 'error': str(e)})
        
        end_time = datetime.now()
        self.simulation_results['end_time'] = end_time.isoformat()
        
        # Generate summary statistics
        total_alerts = sum(r.get('alerts_generated', 0) for r in results)
        total_records = sum(r.get('records_processed', 0) for r in results)
        completed_symbols = len([r for r in results if r['status'] == 'completed'])
        
        self.simulation_results.update({
            'symbol_results': results,
            'total_alerts': total_alerts,
            'total_records_processed': total_records,
            'completed_symbols': completed_symbols,
            'duration_seconds': (end_time - start_time).total_seconds()
        })
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 SIMULATION SUMMARY")
        print("="*60)
        print(f"📊 Symbols processed: {completed_symbols}/{len(self.symbols)}")
        print(f"📈 Total data points: {total_records:,}")
        print(f"🚨 Total alerts: {total_alerts}")
        print(f"⏱️  Duration: {(end_time - start_time).total_seconds():.1f} seconds")
        print(f"⚡ Effective speed: {self.speed}x")
        
        if total_alerts > 0:
            print(f"\n🔥 Top alert symbols:")
            symbol_alert_counts = {}
            for result in results:
                if result.get('alerts_generated', 0) > 0:
                    symbol_alert_counts[result['symbol']] = result['alerts_generated']
            
            for symbol, count in sorted(symbol_alert_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {symbol}: {count} alerts")
        
        print("="*60)
        
        # Save results if requested
        if self.save_alerts:
            self._save_simulation_results()
        
        return self.simulation_results

    def _save_simulation_results(self) -> None:
        """Save simulation results to file."""
        try:
            # Use centralized logs configuration for results
            logs_base = self.logs_config.get_logs_path()
            results_dir = logs_base / self.date
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"simulation_results_{timestamp}.json"
            
            # Convert ORBAlert objects to serializable format
            serializable_results = self.simulation_results.copy()
            serializable_results['generated_alerts'] = [
                {
                    'symbol': alert.symbol,
                    'timestamp': alert.timestamp.isoformat(),
                    'current_price': float(alert.current_price),
                    'orb_high': float(alert.orb_high),
                    'orb_low': float(alert.orb_low),
                    'breakout_type': alert.breakout_type.value,
                    'confidence_score': float(alert.confidence_score),
                    'alert_message': alert.alert_message
                }
                for alert in self.generated_alerts
            ]
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"💾 Results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Pipeline Simulator")
    
    # Symbol selection (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbol",
        type=str,
        help="Single symbol to simulate (e.g., AAPL)"
    )
    symbol_group.add_argument(
        "--symbols-file",
        type=str,
        help="Path to CSV file containing symbols"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to simulate in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time, 10.0 = 10x faster)"
    )
    
    parser.add_argument(
        "--save-alerts",
        action="store_true",
        help="Save generated alerts to files"
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
    
    try:
        simulator = ORBPipelineSimulator(
            symbol=args.symbol,
            symbols_file=args.symbols_file,
            date=args.date,
            speed=args.speed,
            save_alerts=args.save_alerts,
            verbose=args.verbose
        )
        
        results = await simulator.run_simulation()
        
        # Exit with appropriate code
        if results['completed_symbols'] == 0:
            print("❌ No symbols were successfully processed")
            sys.exit(1)
        elif results['completed_symbols'] < len(results['symbols']):
            print("⚠️  Some symbols failed to process")
            sys.exit(2)
        else:
            print("✅ All symbols processed successfully")
            sys.exit(0)
            
    except Exception as e:
        print(f"💥 Simulation failed: {e}")
        logging.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())