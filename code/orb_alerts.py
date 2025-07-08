"""
ORB Trading Alerts System - Main Entry Point

This is the main entry point for the ORB (Opening Range Breakout) trading alerts system.
Based on PCA analysis showing 82.31% variance explained by ORB patterns.

Usage:
    python3 code/orb_alerts.py                    # Start monitoring all symbols
    python3 code/orb_alerts.py --symbols AAPL,TSLA # Monitor specific symbols
    python3 code/orb_alerts.py --test             # Run in test mode
"""

import asyncio
import argparse
import logging
import sys
import json
from typing import Optional
from datetime import datetime, timedelta, time
from pathlib import Path
import pandas as pd
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config
from molecules.orb_alert_engine import ORBAlertEngine
from atoms.alerts.alert_formatter import ORBAlert
# Temporarily disabled due to package version mismatch
# from atoms.api.get_stock_data import get_stock_data

# Alpaca API imports
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    # Fallback to older alpaca-trade-api if available
    try:
        from alpaca_trade_api.rest import REST
        ALPACA_AVAILABLE = "legacy"
    except ImportError:
        ALPACA_AVAILABLE = False


class ORBAlertSystem:
    """Main ORB Alert System orchestrator."""
    
    def __init__(self, symbols_file: Optional[str] = None, test_mode: bool = False):
        """
        Initialize ORB Alert System.
        
        Args:
            symbols_file: Path to symbols CSV file
            test_mode: Run in test mode (no actual alerts)
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize alert engine
        self.alert_engine = ORBAlertEngine(symbols_file) if symbols_file is not None else ORBAlertEngine()
        self.test_mode = test_mode
        
        # Initialize historical data client
        self.historical_client = None
        if ALPACA_AVAILABLE == True:
            try:
                self.historical_client = StockHistoricalDataClient(
                    api_key=config.api_key,
                    secret_key=config.secret_key
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize historical data client: {e}")
        elif ALPACA_AVAILABLE == "legacy":
            try:
                self.historical_client = REST(
                    key_id=config.api_key,
                    secret_key=config.secret_key,
                    base_url=config.base_url
                )
                self.logger.info("Using legacy alpaca-trade-api for historical data")
            except Exception as e:
                self.logger.warning(f"Could not initialize legacy historical data client: {e}")
        
        # Historical data storage setup
        self.historical_data_dir = Path("historical_data")
        self.historical_data_dir.mkdir(exist_ok=True)
        self._setup_data_storage()
        
        # Add alert callback
        self.alert_engine.add_alert_callback(self._handle_alert)
        
        # Statistics
        self.start_time = None
        self.last_data_save = None
        self.data_save_interval = timedelta(minutes=config.data_save_interval_minutes)
        
        self.logger.info(f"ORB Alert System initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Historical data will be saved to: {self.historical_data_dir.absolute()}")
        self.logger.info(f"Data save interval: {config.data_save_interval_minutes} minutes")
        self.logger.info(f"Start at market open ({config.market_open_time} ET): {config.start_collection_at_open}")
        self.logger.info(f"Fetch opening range data if started late: {config.fetch_opening_range_data}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration with Eastern Time."""
        # Create custom formatter for Eastern Time
        class EasternFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                et_tz = pytz.timezone('US/Eastern')
                et_time = datetime.fromtimestamp(record.created, et_tz)
                if datefmt:
                    return et_time.strftime(datefmt)
                return et_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] + ' ET'
        
        # Configure logging with Eastern Time
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # Only configure if not already configured
            handler = logging.StreamHandler()
            formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _setup_data_storage(self) -> None:
        """Setup historical data storage directories."""
        # Create subdirectories for organized storage
        et_tz = pytz.timezone('US/Eastern')
        today = datetime.now(et_tz).strftime("%Y-%m-%d")
        self.daily_data_dir = self.historical_data_dir / today
        self.daily_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (self.daily_data_dir / "market_data").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts").mkdir(exist_ok=True)
        (self.daily_data_dir / "summary").mkdir(exist_ok=True)
        
        self.logger.info(f"Daily data directory: {self.daily_data_dir}")
    
    def _save_historical_data(self) -> None:
        """Save current market data to historical files."""
        try:
            et_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(et_tz)
            symbols = self.alert_engine.get_monitored_symbols()
            
            for symbol in symbols:
                # Get symbol data from the data buffer
                symbol_data = self.alert_engine.data_buffer.get_symbol_data(symbol)
                
                if symbol_data is not None and not symbol_data.empty:
                    # Save to CSV format for easy analysis
                    filename = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = self.daily_data_dir / "market_data" / filename
                    
                    symbol_data.to_csv(filepath, index=False)
                    self.logger.debug(f"Saved {len(symbol_data)} records for {symbol} to {filename}")
            
            # Save metadata about the data save
            metadata = {
                "timestamp": current_time.isoformat(),
                "symbols_count": len(symbols),
                "symbols": symbols,
                "save_interval_minutes": self.data_save_interval.total_seconds() / 60,
                "format": "CSV",
                "total_records_saved": sum(
                    len(data) if (data := self.alert_engine.data_buffer.get_symbol_data(symbol)) is not None else 0
                    for symbol in symbols
                )
            }
            
            metadata_file = self.daily_data_dir / "summary" / f"save_metadata_{current_time.strftime('%H%M%S')}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.last_data_save = current_time
            self.logger.info(f"Historical data saved for {len(symbols)} symbols at {current_time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error saving historical data: {e}")
    
    def _should_save_data(self) -> bool:
        """Check if it's time to save historical data."""
        if self.last_data_save is None:
            return True
        
        return datetime.now() - self.last_data_save >= self.data_save_interval
    
    def _handle_alert(self, alert: ORBAlert) -> None:
        """
        Handle generated ORB alert.
        
        Args:
            alert: Generated ORB alert
        """
        # Save alert to historical data
        self._save_alert_data(alert)
        
        if self.test_mode:
            print(f"[TEST MODE] {alert.alert_message}")
            self.logger.info(f"Test alert: {alert.symbol} - {alert.priority.value}")
        else:
            # Alert is already printed by the engine
            pass
    
    def _save_alert_data(self, alert: ORBAlert) -> None:
        """Save alert data to historical files."""
        try:
            # Save alert as JSON
            alert_filename = f"alert_{alert.symbol}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            alert_filepath = self.daily_data_dir / "alerts" / alert_filename
            
            # Convert alert to dictionary for JSON serialization
            alert_data = {
                "symbol": alert.symbol,
                "timestamp": alert.timestamp.isoformat(),
                "current_price": alert.current_price,
                "orb_high": alert.orb_high,
                "orb_low": alert.orb_low,
                "orb_range": alert.orb_range,
                "orb_midpoint": alert.orb_midpoint,
                "breakout_type": alert.breakout_type.value,
                "breakout_percentage": alert.breakout_percentage,
                "volume_ratio": alert.volume_ratio,
                "confidence_score": alert.confidence_score,
                "priority": alert.priority.value,
                "confidence_level": alert.confidence_level,
                "recommended_stop_loss": alert.recommended_stop_loss,
                "recommended_take_profit": alert.recommended_take_profit,
                "alert_message": alert.alert_message
            }
            
            with open(alert_filepath, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
            self.logger.debug(f"Saved alert data for {alert.symbol} to {alert_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving alert data: {e}")
    
    async def _periodic_data_save(self) -> None:
        """Background task to periodically save historical data."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self._should_save_data():
                    self._save_historical_data()
                    
            except asyncio.CancelledError:
                self.logger.info("Periodic data save task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in periodic data save: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _wait_for_market_open(self) -> None:
        """Wait until market open time if configured to start at market open."""
        if not config.start_collection_at_open:
            return
        
        # Get current time in Eastern Time
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        # Parse market open time in Eastern Time
        market_open_hour, market_open_minute = map(int, config.market_open_time.split(':'))
        market_open_today_et = now_et.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
        
        # If market open time has already passed today, start immediately
        if now_et >= market_open_today_et:
            self.logger.info(f"Market open time ({config.market_open_time} ET) has passed, starting data collection immediately")
            return
        
        # Calculate wait time until market open
        wait_seconds = (market_open_today_et - now_et).total_seconds()
        wait_minutes = wait_seconds / 60
        
        self.logger.info(f"Current time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self.logger.info(f"Waiting {wait_minutes:.1f} minutes until market open ({config.market_open_time} ET)")
        self.logger.info(f"Data collection will start at: {market_open_today_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Wait until market open
        await asyncio.sleep(wait_seconds)
    
    async def _fetch_opening_range_data(self) -> bool:
        """
        Fetch historical data for the opening range period if we missed it.
        
        Returns:
            True if data was successfully fetched, False otherwise
        """
        if not config.fetch_opening_range_data:
            self.logger.info("Opening range data fetching disabled in configuration")
            print("=" * 80)
            print("⚠️  OPENING RANGE DATA FETCH DISABLED")
            print("Historical data fetching is disabled in configuration")
            print("ORB alerts may not function properly if started after market open")
            print("=" * 80)
            return True
            
        if not self.historical_client:
            self.logger.warning("Historical data client not available - cannot fetch opening range data")
            return False
        
        try:
            # Get current time in Eastern Time
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            
            # Parse market open time for today in Eastern Time
            market_open_hour, market_open_minute = map(int, config.market_open_time.split(':'))
            market_open_today_et = now_et.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
            
            # Calculate opening range end time (market open + 15 minutes)
            orb_end_time_et = market_open_today_et + timedelta(minutes=config.orb_period_minutes)
            
            # Check if we're past the opening range period
            if now_et <= orb_end_time_et:
                self.logger.info("Still within opening range period - no historical data fetch needed")
                
                # Print message to screen that we have opening data
                if now_et >= market_open_today_et:
                    print("=" * 80)
                    print("✅ OPENING RANGE DATA AVAILABLE")
                    print("Started during opening range period - collecting data in real-time")
                    print(f"ORB period: {market_open_today_et.strftime('%H:%M')} - {orb_end_time_et.strftime('%H:%M')} ET")
                    print("ORB alerts will be operational after opening range completes!")
                    print("=" * 80)
                
                return True
            
            self.logger.info(f"Fetching opening range data from {market_open_today_et.strftime('%H:%M')} to {orb_end_time_et.strftime('%H:%M')}")
            
            # Get symbols to fetch data for
            symbols = self.alert_engine.get_monitored_symbols()
            
            # Fetch 1-minute bars for the opening range period
            self.logger.info("Fetching historical opening range data...")
            
            try:
                if ALPACA_AVAILABLE == "legacy":
                    # Use legacy alpaca-trade-api
                    bars_data = self._fetch_with_legacy_api(symbols, market_open_today_et, orb_end_time_et)
                else:
                    # Use new alpaca API (currently disabled)
                    bars_data = None
                
                if bars_data and hasattr(bars_data, 'df') and not bars_data.df.empty:
                    # Process and inject the historical data into the data buffer
                    data_count = 0
                    for symbol in symbols:
                        symbol_bars = bars_data.df[bars_data.df['symbol'] == symbol]
                        
                        if not symbol_bars.empty:
                            # Convert to MarketData format and add to buffer
                            for _, bar in symbol_bars.iterrows():
                                from atoms.websocket.alpaca_stream import MarketData
                                
                                # Normalize timestamp to timezone-naive Eastern Time to match websocket data
                                timestamp = bar['timestamp']
                                et_tz = pytz.timezone('US/Eastern')
                                if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                                    # Convert timezone-aware to timezone-naive Eastern Time
                                    timestamp = timestamp.astimezone(et_tz).replace(tzinfo=None)
                                elif hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                                    # Convert timezone-aware to timezone-naive Eastern Time
                                    timestamp = timestamp.astimezone(et_tz).replace(tzinfo=None)
                                
                                market_data = MarketData(
                                    symbol=symbol,
                                    timestamp=timestamp,
                                    price=bar['close'],
                                    volume=bar['volume'],
                                    high=bar['high'],
                                    low=bar['low'],
                                    close=bar['close'],
                                    trade_count=bar.get('trade_count', 1),
                                    vwap=bar.get('vwap', bar['close'])
                                )
                                
                                # Add to data buffer
                                self.alert_engine.data_buffer.add_market_data(market_data)
                                data_count += 1
                    
                    self.logger.info(f"Successfully fetched and loaded {data_count} opening range data points")
                    
                    # Save opening range data to tmp directory for each symbol
                    self._save_opening_range_data_to_tmp(bars_data, market_open_today_et, orb_end_time_et)
                    
                    # Print success message to screen
                    print("=" * 80)
                    print("✅ OPENING RANGE DATA COMPLETE")
                    print(f"Successfully fetched opening range data for {len(symbols)} symbols")
                    print(f"Time period: {market_open_today_et.strftime('%H:%M')} - {orb_end_time_et.strftime('%H:%M')} ET")
                    print(f"Total data points loaded: {data_count}")
                    print("ORB alerts are now fully operational!")
                    print("=" * 80)
                    
                    return True
                else:
                    self.logger.warning("No historical data received for opening range period")
                    return False
                    
            except Exception as fetch_error:
                self.logger.error(f"Error during historical data fetch: {fetch_error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error fetching opening range data: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            
            # Print error message to screen
            print("=" * 80)
            print("❌ OPENING RANGE DATA FETCH FAILED")
            print(f"Error: {e}")
            print("ORB alerts may not function properly without opening range data")
            print("Check API credentials and network connection")
            print("=" * 80)
            
            return False
    
    def _save_opening_range_data_to_tmp(self, bars_data, market_open_time: datetime, orb_end_time: datetime) -> None:
        """
        Save opening range data for each symbol to tmp directory.
        
        Args:
            bars_data: The fetched historical data
            market_open_time: Market open datetime
            orb_end_time: ORB period end datetime
        """
        try:
            # Ensure tmp directory exists
            from pathlib import Path
            tmp_dir = Path("tmp")
            tmp_dir.mkdir(exist_ok=True)
            
            if hasattr(bars_data, 'df') and not bars_data.df.empty:
                # Get unique symbols
                symbols = bars_data.df['symbol'].unique()
                
                for symbol in symbols:
                    symbol_data = bars_data.df[bars_data.df['symbol'] == symbol].copy()
                    
                    if not symbol_data.empty:
                        # Sort by timestamp
                        symbol_data = symbol_data.sort_values('timestamp')
                        
                        # Create filename
                        date_str = market_open_time.strftime('%Y%m%d')
                        filename = f"{symbol}_opening_range_{date_str}.csv"
                        filepath = tmp_dir / filename
                        
                        # Save to CSV
                        symbol_data.to_csv(filepath, index=False)
                        
                        self.logger.info(f"Saved opening range data for {symbol} to {filepath}")
                        
                        # Also print summary for each symbol
                        orb_high = symbol_data['high'].max()
                        orb_low = symbol_data['low'].min()
                        orb_range = orb_high - orb_low
                        data_points = len(symbol_data)
                        
                        print(f"📊 {symbol}: {data_points} bars | ORB High: ${orb_high:.3f} | ORB Low: ${orb_low:.3f} | Range: ${orb_range:.3f}")
                
                print(f"💾 Opening range data saved to tmp/ directory")
            
        except Exception as e:
            self.logger.error(f"Error saving opening range data to tmp: {e}")
    
    def _fetch_with_legacy_api(self, symbols, start_time, end_time):
        """Fetch historical data using legacy alpaca-trade-api."""
        try:
            # Determine which feed to use based on configuration
            feed = 'iex' if "paper" in config.base_url else 'sip'
            self.logger.info(f"Using legacy API to fetch data for {len(symbols)} symbols with {feed.upper()} feed")
            
            # Convert symbols to list if needed
            symbol_list = symbols if isinstance(symbols, list) else list(symbols)
            
            # Fetch data for each symbol (legacy API doesn't support multi-symbol requests well)
            all_bars = []
            
            for symbol in symbol_list:
                try:
                    # Fetch 1-minute bars (legacy API needs specific format)
                    # Convert to UTC and format as RFC3339
                    start_utc = start_time.astimezone(pytz.UTC)
                    end_utc = end_time.astimezone(pytz.UTC)
                    start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
                    end_str = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
                    
                    # Use the feed determined at the beginning of the function
                    
                    bars = self.historical_client.get_bars(
                        symbol,
                        '1Min',
                        start=start_str,
                        end=end_str,
                        limit=20,  # Extra buffer for 15 minutes
                        feed=feed  # Use IEX for paper trading, SIP for live trading
                    )
                    
                    if bars:
                        for bar in bars:
                            bar_data = {
                                'timestamp': bar.t,
                                'symbol': symbol,
                                'open': float(bar.o),
                                'high': float(bar.h),
                                'low': float(bar.l),
                                'close': float(bar.c),
                                'volume': int(bar.v),
                                'trade_count': getattr(bar, 'n', 1),
                                'vwap': getattr(bar, 'vw', float(bar.c))
                            }
                            all_bars.append(bar_data)
                        
                        self.logger.info(f"Fetched {len(bars)} bars for {symbol}")
                    else:
                        self.logger.warning(f"No data returned for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {e}")
            
            if all_bars:
                # Convert to DataFrame-like structure
                import pandas as pd
                df = pd.DataFrame(all_bars)
                
                # Convert timestamps to timezone-naive Eastern Time for consistency
                if not df.empty and 'timestamp' in df.columns:
                    et_tz = pytz.timezone('US/Eastern')
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    if df['timestamp'].dt.tz is not None:
                        # Convert timezone-aware to timezone-naive Eastern Time
                        df['timestamp'] = df['timestamp'].dt.tz_convert(et_tz).dt.tz_localize(None)
                    else:
                        # If somehow timezone-naive, assume UTC and convert
                        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(et_tz).dt.tz_localize(None)
                
                # Create mock bars_data object with df attribute
                class MockBarsData:
                    def __init__(self, dataframe):
                        self.df = dataframe
                
                return MockBarsData(df)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error in legacy API fetch: {e}")
            return None
    
    async def start(self) -> None:
        """Start the ORB Alert System."""
        self.logger.info("Starting ORB Alert System...")
        
        # Start periodic data saving task
        data_save_task = None
        
        try:
            # Validate configuration
            config_errors = config.validate()
            if config_errors:
                self.logger.error(f"Configuration errors: {config_errors}")
                return
            
            # Wait for market open if configured
            await self._wait_for_market_open()
            
            # Set start time after market open wait
            et_tz = pytz.timezone('US/Eastern')
            self.start_time = datetime.now(et_tz)
            self.logger.info(f"Starting data collection at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Fetch opening range data if we started after market open
            await self._fetch_opening_range_data()
            
            # Start alert engine
            symbols = self.alert_engine.get_monitored_symbols()
            self.logger.info(f"Monitoring {len(symbols)} symbols: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                self.logger.info(f"... and {len(symbols) - 10} more symbols")
            
            # Start periodic data saving in background
            data_save_task = asyncio.create_task(self._periodic_data_save())
            self.logger.info(f"Started periodic data saving task (every {config.data_save_interval_minutes} minutes)")
            
            # Start the alert engine
            await self.alert_engine.start()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in alert engine: {e}")
        finally:
            # Cancel data saving task
            if data_save_task:
                data_save_task.cancel()
                try:
                    await data_save_task
                except asyncio.CancelledError:
                    pass
                    
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the ORB Alert System."""
        self.logger.info("Stopping ORB Alert System...")
        
        try:
            await self.alert_engine.stop()
            self.logger.info("ORB Alert System stopped")
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def get_statistics(self) -> dict:
        """Get system statistics."""
        engine_stats = self.alert_engine.get_stats()
        return {
            'alerts_generated': engine_stats.total_alerts_generated,
            'symbols_count': engine_stats.symbols_monitored,
            'start_time': self.start_time,
            'engine_stats': engine_stats
        }
    
    def print_daily_summary(self) -> None:
        """Print daily summary statistics."""
        summary = self.alert_engine.get_daily_summary()
        
        print("\n" + "="*60)
        print(f"ORB Alert System - Daily Summary")
        print(f"Date: {summary.get('date', 'N/A')}")
        print(f"Total Alerts: {summary.get('total_alerts', 0)}")
        print(f"Average Confidence: {summary.get('avg_confidence', 0):.3f}")
        print(f"Max Confidence: {summary.get('max_confidence', 0):.3f}")
        
        priority_breakdown = summary.get('priority_breakdown', {})
        print(f"Priority Breakdown:")
        for priority, count in priority_breakdown.items():
            print(f"  {priority}: {count}")
        
        print("="*60 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Trading Alerts System")
    
    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Path to symbols CSV file (default: data/symbols.csv)"
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
        "--summary",
        action="store_true",
        help="Show daily summary and exit"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start system
    try:
        system = ORBAlertSystem(
            symbols_file=args.symbols_file,
            test_mode=args.test
        )
        
        if args.summary:
            # Show summary and exit
            system.print_daily_summary()
            return
        
        if args.test:
            print("Running in test mode - alerts will be marked as [TEST MODE]")
        
        await system.start()
        
    except Exception as e:
        logging.error(f"Failed to start ORB Alert System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())