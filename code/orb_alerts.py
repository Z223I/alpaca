"""
ORB Trading Alerts System - Main Entry Point

This is the main entry point for the ORB (Opening Range Breakout) trading alerts system.
Based on PCA analysis showing 82.31% variance explained by ORB patterns.

Usage:
    python3 code/orb_alerts.py                      # Start monitoring all symbols (SIP feed)
    python3 code/orb_alerts.py --symbols AAPL,TSLA  # Monitor specific symbols
    python3 code/orb_alerts.py --test               # Run in test mode
    python3 code/orb_alerts.py --use-iex            # Use IEX data feed instead of SIP
    python3 code/orb_alerts.py --date 2025-08-13    # Process specific date (default: today)
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
sys.path.append(str(Path(__file__).parent.parent))

from atoms.config.alert_config import config
from atoms.alerts.config import get_logs_root_dir, get_data_root_dir
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

    def __init__(self, symbols_file: Optional[str] = None, test_mode: bool = False, use_iex: bool = False, date: Optional[str] = None):
        """
        Initialize ORB Alert System.

        Args:
            symbols_file: Path to symbols CSV file
            test_mode: Run in test mode (no actual alerts)
            use_iex: Use IEX data feed instead of SIP (default: False, uses SIP)
            date: Date to process in YYYY-MM-DD format (default: today)
        """
        # Setup logging
        self.logger = self._setup_logging()

        # Store date parameter or default to today
        if date is None:
            et_tz = pytz.timezone('US/Eastern')
            self.target_date = datetime.now(et_tz).strftime('%Y-%m-%d')
        else:
            self.target_date = date

        # Initialize alert engine with centralized data configuration
        if symbols_file is not None:
            # Use provided symbols file path
            self.alert_engine = ORBAlertEngine(symbols_file)
        else:
            # Use centralized data configuration to determine symbols file path
            data_config = get_data_root_dir()
            default_symbols_file = data_config.get_symbols_file_path(self.target_date)
            
            # Hard failure if expected symbols file doesn't exist
            if not default_symbols_file.exists():
                raise FileNotFoundError(f"Required symbols file not found: {default_symbols_file}")
            
            self.alert_engine = ORBAlertEngine(str(default_symbols_file))
            self.logger.info(f"Using default symbols file: {default_symbols_file}")
        self.test_mode = test_mode
        self.use_iex = use_iex

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

        # Log data feed selection
        feed_type = "IEX" if self.use_iex else "SIP"
        self.logger.info(f"ORB Alert System initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Using {feed_type} data feed")
        self.logger.info(f"Historical data will be saved to: {self.historical_data_dir.absolute()}")
        self.logger.info(f"Data save interval: {config.data_save_interval_minutes} minutes")
        self.logger.info(f"Data collection starts at {config.market_open_time} ET, ORB period: {config.orb_start_time}-{config.alert_window_start} ET")

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
            # Setup console handler
            console_handler = logging.StreamHandler()
            formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Setup file handler using centralized logs config
            try:
                logs_config = get_logs_root_dir()
                log_dir = logs_config.get_component_logs_dir("orb_alerts")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                et_tz = pytz.timezone('US/Eastern')
                log_filename = f"orb_alerts_{datetime.now(et_tz).strftime('%Y%m%d_%H%M%S')}.log"
                log_file_path = log_dir / log_filename
                
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            except Exception as e:
                # If file logging fails, continue with console logging only
                logger.warning(f"Could not setup file logging: {e}")
            
            logger.setLevel(logging.INFO)

        return logger

    def _setup_data_storage(self) -> None:
        """Setup historical data storage directories."""
        # Create subdirectories for organized storage using target date
        self.daily_data_dir = self.historical_data_dir / self.target_date
        self.daily_data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different data types
        (self.daily_data_dir / "market_data").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts" / "bullish").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts" / "bearish").mkdir(exist_ok=True)
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
                    # Data buffer contains historical data from 9:00 AM plus real-time websocket data
                    combined_data = symbol_data

                    # Clean up old files for this symbol before saving new one
                    self._cleanup_old_symbol_files(symbol)

                    # Save to CSV format for easy analysis
                    filename = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = self.daily_data_dir / "market_data" / filename

                    combined_data.to_csv(filepath, index=False)
                    self.logger.debug(f"Saved {len(combined_data)} records (historical + real-time data from 9:00 AM) for {symbol} to {filename}")

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

    def _cleanup_old_symbol_files(self, symbol: str) -> None:
        """
        Remove old historical data files for a symbol, keeping only the latest.

        Args:
            symbol: The symbol to clean up files for
        """
        try:
            market_data_dir = self.daily_data_dir / "market_data"

            # Find all files for this symbol
            symbol_files = list(market_data_dir.glob(f"{symbol}_*.csv"))

            if len(symbol_files) <= 1:
                # No cleanup needed if 1 or fewer files exist
                return

            # Sort files by modification time (newest first)
            symbol_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            # Keep only the latest file, remove the rest
            files_to_remove = symbol_files[1:]  # All except the first (newest)

            for old_file in files_to_remove:
                try:
                    old_file.unlink()
                    self.logger.debug(f"Removed old data file: {old_file.name}")
                except OSError as e:
                    self.logger.warning(f"Could not remove old file {old_file.name}: {e}")

            if files_to_remove:
                self.logger.debug(f"Cleaned up {len(files_to_remove)} old data files for {symbol}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old files for {symbol}: {e}")


    def _should_save_data(self) -> bool:
        """Check if it's time to save historical data."""
        if self.last_data_save is None:
            return True

        # Use Eastern Time for consistency with last_data_save
        et_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(et_tz)

        # Ensure both timestamps are timezone-aware for comparison
        if self.last_data_save.tzinfo is None:
            # If last_data_save is timezone-naive, assume it's in Eastern Time
            last_save_aware = et_tz.localize(self.last_data_save)
        else:
            # Convert to Eastern Time if it's in a different timezone
            last_save_aware = self.last_data_save.astimezone(et_tz)

        return current_time - last_save_aware >= self.data_save_interval

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
            # Determine subdirectory based on breakout type
            if alert.breakout_type.value == "bullish_breakout":
                subdir = "bullish"
            elif alert.breakout_type.value == "bearish_breakdown":
                subdir = "bearish"
            else:
                subdir = ""  # Save in root alerts directory for other types

            # Save alert as JSON
            alert_filename = f"alert_{alert.symbol}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

            if subdir:
                alert_dir = self.daily_data_dir / "alerts" / subdir
                alert_filepath = alert_dir / alert_filename
                # Ensure subdirectory exists
                alert_dir.mkdir(parents=True, exist_ok=True)
            else:
                alert_dir = self.daily_data_dir / "alerts"
                alert_filepath = alert_dir / alert_filename
                # Ensure alerts directory exists
                alert_dir.mkdir(parents=True, exist_ok=True)

            # Convert alert to dictionary for JSON serialization
            alert_data = {
                "symbol": alert.symbol,
                "timestamp": alert.timestamp.isoformat(),
                "current_price": float(alert.current_price),
                "orb_high": float(alert.orb_high),
                "orb_low": float(alert.orb_low),
                "orb_range": float(alert.orb_range),
                "orb_midpoint": float(alert.orb_midpoint),
                "breakout_type": alert.breakout_type.value,
                "breakout_percentage": float(alert.breakout_percentage),
                "volume_ratio": float(alert.volume_ratio),
                "confidence_score": float(alert.confidence_score),
                "priority": alert.priority.value,
                "confidence_level": alert.confidence_level,
                "recommended_stop_loss": float(alert.recommended_stop_loss),
                "recommended_take_profit": float(alert.recommended_take_profit),
                "alert_message": alert.alert_message,
                # Candlestick data for super alert filtering
                "low_price": float(alert.low_price) if alert.low_price is not None else None,
                "high_price": float(alert.high_price) if alert.high_price is not None else None,
                "open_price": float(alert.open_price) if alert.open_price is not None else None,
                "close_price": float(alert.close_price) if alert.close_price is not None else None,
                "volume": int(alert.volume) if alert.volume is not None else None,
                # EMA Technical Indicators - ensure proper JSON serialization
                "ema_9": float(alert.ema_9) if alert.ema_9 is not None else None,
                "ema_20": float(alert.ema_20) if alert.ema_20 is not None else None,
                "ema_9_above_20": bool(alert.ema_9_above_20) if alert.ema_9_above_20 is not None else None,
                "ema_9_below_20": bool(alert.ema_9_below_20) if alert.ema_9_below_20 is not None else None,
                "ema_divergence": float(alert.ema_divergence) if alert.ema_divergence is not None else None
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

        # If data collection start time has already passed today, start immediately
        if now_et >= market_open_today_et:
            self.logger.info(f"Data collection start time ({config.market_open_time} ET) has passed, starting data collection immediately")
            return

        # Calculate wait time until data collection start
        wait_seconds = (market_open_today_et - now_et).total_seconds()
        wait_minutes = wait_seconds / 60

        self.logger.info(f"Current time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self.logger.info(f"Waiting {wait_minutes:.1f} minutes until data collection starts ({config.market_open_time} ET)")
        self.logger.info(f"Data collection will start at: {market_open_today_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self.logger.info(f"ORB period will be: {config.orb_start_time}-{config.alert_window_start} ET")

        # Wait until data collection start time
        await asyncio.sleep(wait_seconds)

    async def _fetch_historical_data_from_data_start(self) -> bool:
        """
        Fetch historical data from data collection start time (9:00 AM) to current time.
        This ensures EMA20 can be calculated by market open (9:30 AM) and complete data coverage 
        regardless of when the system starts.

        Returns:
            True if data was successfully fetched, False otherwise
        """
        if not self.historical_client:
            self.logger.warning("Historical data client not available - cannot fetch historical data")
            return False

        try:
            # Get current time in Eastern Time
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)

            # Parse data collection start time for today in Eastern Time
            data_start_hour, data_start_minute = map(int, config.market_open_time.split(':'))
            data_start_today_et = now_et.replace(hour=data_start_hour, minute=data_start_minute, second=0, microsecond=0)

            # Check if we're before data collection start time
            if now_et < data_start_today_et:
                self.logger.info("Before data collection start time - no historical data fetch needed")
                return True

            # Calculate how much data we need to fetch
            minutes_since_data_start = (now_et - data_start_today_et).total_seconds() / 60

            if minutes_since_data_start < 1:
                self.logger.info("Started within 1 minute of data collection start - minimal historical data needed")
                return True

            # Get symbols to fetch data for
            symbols = self.alert_engine.get_monitored_symbols()

            # Fetch 1-minute bars from data collection start to current time
            self.logger.info(f"Fetching historical data from {data_start_today_et.strftime('%H:%M')} to {now_et.strftime('%H:%M')} ET ({minutes_since_data_start:.1f} minutes)...")
            print(f"📊 Fetching historical data from 9:00 AM to {now_et.strftime('%H:%M')} ET for {len(symbols)} symbols...")

            try:
                if ALPACA_AVAILABLE == "legacy":
                    # Use legacy alpaca-trade-api
                    bars_data = self._fetch_with_legacy_api(symbols, data_start_today_et, now_et)
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
                                    vwap=bar.get('vwap', bar['close']),
                                    open=bar.get('open', bar['close'])
                                )

                                # Add to data buffer
                                self.alert_engine.data_buffer.add_market_data(market_data)
                                data_count += 1

                    self.logger.info(f"Successfully fetched and loaded {data_count} historical data points")
                    print(f"✅ Successfully loaded {data_count} historical data points from 9:00 AM")

                    return True
                else:
                    self.logger.warning("No historical data received from data collection start")
                    return False

            except Exception as fetch_error:
                self.logger.error(f"Error during historical data fetch: {fetch_error}")
                return False

        except Exception as e:
            self.logger.error(f"Error fetching historical data from data collection start: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _fetch_with_legacy_api(self, symbols, start_time, end_time):
        """Fetch historical data using legacy alpaca-trade-api."""
        try:
            # Determine which feed to use based on configuration and CLI argument
            if self.use_iex:
                feed = 'iex'
            else:
                feed = 'sip'  # Default to SIP

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
                        limit=1000,  # Increased limit for full day coverage
                        feed=feed  # Use SIP by default, IEX if --use-iex flag is specified
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

                        self.logger.debug(f"Fetched {len(bars)} bars for {symbol}")
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

            # Fetch historical data from 9:00 AM to current time (ALWAYS)
            await self._fetch_historical_data_from_data_start()

            # Start alert engine
            symbols = self.alert_engine.get_monitored_symbols()

            # Display up to 25 symbols, five per line
            if len(symbols) <= 25:
                # Display all symbols, five per line
                for i in range(0, len(symbols), 5):
                    line_symbols = symbols[i:i+5]
                    if i == 0:
                        self.logger.info(f"Monitoring {len(symbols)} symbols: {', '.join(line_symbols)}")
                    else:
                        self.logger.info(f"                           {', '.join(line_symbols)}")
            else:
                # Display first 25 symbols, five per line, then show count of remaining
                for i in range(0, 25, 5):
                    line_symbols = symbols[i:i+5]
                    if i == 0:
                        self.logger.info(f"Monitoring {len(symbols)} symbols: {', '.join(line_symbols)}")
                    else:
                        self.logger.info(f"                           {', '.join(line_symbols)}")
                self.logger.info(f"... and {len(symbols) - 25} more symbols")

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
        help="Path to symbols CSV file (default: auto-detected based on date from centralized data config)"
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

    parser.add_argument(
        "--use-iex",
        action="store_true",
        help="Use IEX data feed instead of SIP (default: SIP)"
    )

    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d'),
        help="Date to process in YYYY-MM-DD format (default: today)"
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
            test_mode=args.test,
            use_iex=args.use_iex,
            date=args.date
        )

        if args.summary:
            # Show summary and exit
            system.print_daily_summary()
            return

        if args.test:
            print("Running in test mode - alerts will be marked as [TEST MODE]")

        # Show data feed selection
        feed_type = "IEX" if args.use_iex else "SIP"
        print(f"Using {feed_type} data feed for market data")

        await system.start()

    except Exception as e:
        logging.error(f"Failed to start ORB Alert System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())