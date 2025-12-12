#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
"""
Squeeze Alerts System - Real-time WebSocket Monitoring

Phase 1: Monitor WebSocket and listen to subscribed stocks (eavesdropping)
Phase 2: Add squeeze detection logic (future)

This system connects to the trade stream WebSocket server and monitors real-time
trades for a list of symbols. It subscribes to symbols from a CSV file and
logs all incoming trade data.

Architecture:
    Alpaca WebSocket → Backend Server (8765) → Proxy (8766) → This Script

Usage:
    python3 cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
    python3 cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py --symbols-file data_master/master.csv
    python3 cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py --symbols AAPL,TSLA,MSFT
    python3 cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py --max-symbols 50 --verbose
"""

import asyncio
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import List, Set, Dict, Deque, Any, Optional
from collections import deque
import websockets
import pytz

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))  # cgi-bin/molecules/alpaca_molecules/
molecules_dir = os.path.dirname(script_dir)  # cgi-bin/molecules/
cgi_bin_dir = os.path.dirname(molecules_dir)  # cgi-bin/
project_root = os.path.dirname(cgi_bin_dir)  # project root
sys.path.insert(0, project_root)

from atoms.telegram.telegram_post import TelegramPoster
from atoms.telegram.user_manager import UserManager
from atoms.api.init_alpaca_client import init_alpaca_client
from atoms.api.collect_premarket_data import collect_premarket_data
from atoms.api.fundamental_data import FundamentalDataFetcher

# Import market data - use relative import since we're in the same directory
import importlib.util
market_data_path = os.path.join(script_dir, 'market_data.py')
spec = importlib.util.spec_from_file_location("market_data", market_data_path)
market_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(market_data_module)
AlpacaMarketData = market_data_module.AlpacaMarketData

# Extended trading hours constants (ET)
EXTENDED_HOURS_START = dt_time(4, 0)   # 4:00 AM ET - Premarket start
EXTENDED_HOURS_END = dt_time(20, 0)    # 8:00 PM ET - After-hours end
MARKET_OPEN = dt_time(9, 30)           # 9:30 AM ET - Regular market open
MARKET_CLOSE = dt_time(16, 0)          # 4:00 PM ET - Regular market close


class SqueezeAlertsMonitor:
    """Real-time trade monitoring system for squeeze detection."""

    # ===== OUTCOME TRACKING CONFIGURATION =====
    # Enable/disable outcome tracking globally
    OUTCOME_TRACKING_ENABLED = True

    # Duration to track outcomes after squeeze detection (minutes)
    OUTCOME_TRACKING_DURATION_MINUTES = 10

    # Specific intervals for recording price snapshots (in seconds)
    # Combines sub-minute intervals (10s, 20s, 30s, 40s, 50s, 90s, 150s)
    # with minute intervals (60s=1min through 600s=10min)
    OUTCOME_TRACKING_INTERVALS_SECONDS = [
        10, 20, 30, 40, 50, 90, 150,           # Sub-minute intervals
        60, 120, 180, 240, 300, 360, 420, 480, 540, 600  # 1-10 minute intervals
    ]

    # Time tolerance for interval recording (seconds)
    OUTCOME_INTERVAL_TOLERANCE_SECONDS = 5  # Reduced from 30s for tighter sub-minute tracking

    # Maximum concurrent followups to track (memory/performance limit)
    OUTCOME_MAX_CONCURRENT_FOLLOWUPS = 100

    # Stop loss threshold (percentage below entry price)
    OUTCOME_STOP_LOSS_PERCENT = 7.5

    # Target gain thresholds to track achievement (percentages above entry)
    OUTCOME_TARGET_THRESHOLDS = [5.0, 10.0, 15.0]

    def __init__(
        self,
        symbols: List[str] = None,
        symbols_file: str = None,
        max_symbols: int = None,
        verbose: bool = False,
        websocket_url: str = "ws://localhost:8766",
        test_duration: int = None,
        use_existing: bool = False,
        squeeze_percent: float = 2.0
    ):
        """
        Initialize Squeeze Alerts Monitor.

        Args:
            symbols: List of symbols to monitor (e.g., ["AAPL", "TSLA"])
            symbols_file: Path to CSV file containing symbols
            max_symbols: Maximum number of symbols to monitor (None = unlimited)
            verbose: Enable verbose logging
            websocket_url: WebSocket server URL
            test_duration: Run for N seconds then stop and print summary (test mode)
            use_existing: Auto-subscribe to existing symbols from other clients
            squeeze_percent: Percent price increase in 10 seconds to trigger squeeze alert (default: 2.0%)
        """
        self.symbols_to_monitor: List[str] = []
        self.max_symbols = max_symbols
        self.websocket_url = websocket_url
        self.verbose = verbose
        self.test_duration = test_duration
        self.use_existing = use_existing
        self.squeeze_percent = squeeze_percent

        # Setup logging
        self.logger = self._setup_logging(verbose)

        # Statistics tracking
        self.trade_count = 0
        self.trades_by_symbol: Dict[str, int] = {}
        self.active_subscriptions: Set[str] = set()
        self.start_time = None
        self.test_mode = test_duration is not None

        # Squeeze detection: rolling 10-second window of trades per symbol
        # Each symbol stores: deque of (timestamp, price, size) tuples
        self.price_history: Dict[str, Deque[tuple]] = {}
        self.squeeze_count = 0
        self.squeezes_by_symbol: Dict[str, int] = {}

        # Volume history tracking for Phase 1 metrics
        # Each symbol stores: deque of (timestamp, volume) tuples for 5-minute rolling window
        self.volume_history: Dict[str, Deque[tuple]] = {}

        # Day open and low prices per symbol (for distance calculations)
        self.day_open_prices: Dict[str, float] = {}
        self.day_low_prices: Dict[str, float] = {}

        # SPY price tracking for concurrent change calculation
        self.spy_squeeze_start_price: Dict[str, float] = {}  # symbol -> SPY price at squeeze start
        self.latest_spy_price: float = None  # Latest SPY price seen
        self.latest_spy_timestamp: datetime = None  # When SPY price was last updated

        # ===== OUTCOME TRACKING DATA STRUCTURES =====
        # Active followups: tracks outcomes for squeezes in progress
        # Key format: "AAPL_2025-12-12_152045" (symbol_date_time)
        self.active_followups: Dict[str, Dict[str, Any]] = {}

        # Cumulative volume since squeeze start (for each followup)
        self.followup_volume_tracking: Dict[str, int] = {}

        # Cumulative trades since squeeze start (for each followup)
        self.followup_trades_tracking: Dict[str, int] = {}

        # Telegram integration
        self.telegram_poster = TelegramPoster()
        self.user_manager = UserManager()

        # Market data client for HOD/premarket high
        self.market_data = AlpacaMarketData()

        # Setup directories for saving squeeze alerts
        self.et_tz = pytz.timezone('US/Eastern')
        self.today = datetime.now(self.et_tz).strftime('%Y-%m-%d')
        self.squeeze_alerts_sent_dir = Path(project_root) / "historical_data" / self.today / "squeeze_alerts_sent"
        self.squeeze_alerts_sent_dir.mkdir(parents=True, exist_ok=True)

        # Load squeeze counts from existing alert files (survives restarts, resets daily)
        self._load_squeeze_counts()

        # Manual symbols file monitoring
        self.manual_symbols_file = Path(project_root) / "data" / "manual_symbols.json"
        self.last_subscription_time = None

        # Gains tracking (stores premarket gains per symbol)
        self.gains_per_symbol: Dict[str, Dict] = {}
        self.last_gains_update = None
        self.alpaca_client = None  # Will be initialized when first needed

        # Volume surge and fundamental data tracking
        self.volume_surge_data: Dict[str, Dict] = {}  # Stores volume surge ratio per symbol
        self.fundamental_data: Dict[str, Dict] = {}  # Stores float shares, market cap per symbol
        self.fundamental_fetcher = FundamentalDataFetcher(verbose=verbose)
        self.scanner_dir = Path(project_root) / "historical_data" / self.today / "scanner"
        self.symbol_list_csv_path = self.scanner_dir / "symbol_list.csv"
        self.hourly_volume_data: Dict[str, int] = {}  # Stores total volume since 04:00 ET per symbol

        # Load symbols
        if use_existing:
            # Will query existing symbols from backend after connection
            self.logger.info("Will auto-subscribe to existing symbols from backend")
        elif symbols:
            self.symbols_to_monitor = [s.upper() for s in symbols]
        elif symbols_file:
            self.symbols_to_monitor = self._load_symbols_from_file(symbols_file)

        # Apply max_symbols limit
        if self.max_symbols and len(self.symbols_to_monitor) > self.max_symbols:
            self.logger.info(f"Limiting symbols from {len(self.symbols_to_monitor)} to {self.max_symbols}")
            self.symbols_to_monitor = self.symbols_to_monitor[:self.max_symbols]

        self.logger.info(f"Initialized with {len(self.symbols_to_monitor)} symbols to monitor")

    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

    def _load_symbols_from_file(self, filepath: str) -> List[str]:
        """Load symbols from CSV file."""
        symbols = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common column names for symbols
                    symbol = row.get('symbol') or row.get('Symbol') or row.get('ticker') or row.get('Ticker')
                    if symbol:
                        symbols.append(symbol.upper())
            self.logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
        except FileNotFoundError:
            self.logger.error(f"Symbols file not found: {filepath}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error loading symbols from {filepath}: {e}")
            sys.exit(1)
        return symbols

    def _load_manual_symbols(self) -> List[str]:
        """Load manually added symbols from JSON file."""
        manual_symbols = []

        if not self.manual_symbols_file.exists():
            self.logger.debug(f"Manual symbols file not found: {self.manual_symbols_file}")
            return manual_symbols

        try:
            with open(self.manual_symbols_file, 'r') as f:
                data = json.load(f)

            symbols = data.get('symbols', [])
            manual_symbols = [s.strip().upper() for s in symbols if s.strip()]

            if manual_symbols:
                self.logger.info(f"Loaded {len(manual_symbols)} manual symbols from {self.manual_symbols_file}")

        except Exception as e:
            self.logger.error(f"Error loading manual symbols file {self.manual_symbols_file}: {e}")

        return manual_symbols

    def _load_symbol_volume_data(self) -> Dict[str, Dict]:
        """
        Load volume surge data from symbol_list.csv.

        Returns:
            Dictionary mapping symbols to volume surge data
        """
        volume_data = {}

        if not self.symbol_list_csv_path.exists():
            self.logger.debug(f"⚠️ Symbol list CSV not found: {self.symbol_list_csv_path}")
            return volume_data

        try:
            with open(self.symbol_list_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('symbol', '').strip().upper()
                    if symbol:
                        volume_surge_detected = row.get('volume_surge_detected', 'False').strip()
                        volume_surge_ratio = row.get('volume_surge_ratio', '')

                        # Convert volume_surge_detected to boolean
                        detected = volume_surge_detected.lower() in ('true', '1', 'yes')

                        # Convert volume_surge_ratio to float if available
                        ratio = None
                        if volume_surge_ratio and volume_surge_ratio.strip():
                            try:
                                ratio = float(volume_surge_ratio)
                            except ValueError:
                                pass

                        volume_data[symbol] = {
                            'volume_surge_detected': detected,
                            'volume_surge_ratio': ratio
                        }

            self.logger.info(f"📊 Loaded volume surge data for {len(volume_data)} symbols from {self.symbol_list_csv_path}")

        except Exception as e:
            self.logger.error(f"❌ Error loading symbol volume data: {e}")

        return volume_data

    def _fetch_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch fundamental data for symbols.

        Returns:
            Dictionary mapping symbols to fundamental data
        """
        fundamental_data = {}

        if not symbols:
            return fundamental_data

        self.logger.info(f"📊 Fetching fundamental data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                data = self.fundamental_fetcher.get_fundamental_data(symbol)

                # Only store if we got valid data
                if data and data.get('source') != 'none':
                    fundamental_data[symbol] = data

                    if self.verbose:
                        shares_str = f"{data['shares_outstanding']:,}" if data['shares_outstanding'] else "N/A"
                        float_str = f"{data['float_shares']:,}" if data['float_shares'] else "N/A"
                        cap_str = f"${data['market_cap']:,}" if data['market_cap'] else "N/A"
                        self.logger.debug(
                            f"  {symbol}: Shares: {shares_str} | Float: {float_str} | "
                            f"Cap: {cap_str} | Source: {data['source']}")

            except Exception as e:
                self.logger.debug(f"  {symbol}: Could not fetch fundamental data: {e}")

        self.logger.info(f"✅ Fetched fundamental data for {len(fundamental_data)}/{len(symbols)} symbols")

        return fundamental_data

    async def _collect_hourly_volume_data(self, symbols: List[str]) -> Dict[str, int]:
        """
        Collect 1-hour candlesticks from 04:00 ET to now and sum volume for float rotation.

        Args:
            symbols: List of symbols to collect hourly volume for

        Returns:
            Dictionary mapping symbols to their total volume since 04:00 ET
        """
        if not symbols or not self.alpaca_client:
            return {}

        try:
            import alpaca_trade_api as tradeapi
            from datetime import timedelta

            volume_dict = {}

            # Calculate time range (04:00 ET today to now)
            current_et = datetime.now(self.et_tz)

            # Start at 04:00 ET today
            start_time = current_et.replace(hour=4, minute=0, second=0, microsecond=0)

            # If current time is before 04:00 ET, use yesterday's 04:00 ET
            if current_et.hour < 4:
                start_time = start_time - timedelta(days=1)

            end_time = current_et

            self.logger.debug(f"📊 Fetching hourly volume from {start_time.strftime('%H:%M ET')} to {end_time.strftime('%H:%M ET')}")

            # Collect hourly data for each symbol
            for symbol in symbols:
                try:
                    # Fetch 1-hour bars
                    bars = self.alpaca_client.get_bars(
                        symbol,
                        tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Hour),  # 1-hour bars
                        start=start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        end=end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        limit=100,  # Up to 100 hours
                        feed='sip'
                    )

                    if bars and len(bars) > 0:
                        # Sum volume from all hourly bars
                        total_volume = 0
                        bar_count = 0

                        for bar in bars:
                            total_volume += int(bar.v)
                            bar_count += 1

                        volume_dict[symbol] = total_volume

                        self.logger.debug(
                            f"📊 {symbol}: Summed {bar_count} hourly bars "
                            f"(04:00 ET to now) = {total_volume:,} volume")

                except Exception as symbol_error:
                    self.logger.debug(f"⚠️ Error collecting hourly volume for {symbol}: {symbol_error}")
                    continue

            if volume_dict:
                self.logger.info(f"📊 Collected hourly volume for {len(volume_dict)} symbols")

            return volume_dict

        except Exception as e:
            self.logger.error(f"❌ Error collecting hourly volume data: {e}")
            return {}

    async def _collect_volume_and_fundamental_data(self):
        """
        Collect volume surge and fundamental data for all subscribed symbols.

        This method runs periodically to update surge ratios and float shares.
        """
        try:
            # Load volume surge data from CSV
            volume_data = self._load_symbol_volume_data()
            if volume_data:
                self.volume_surge_data = volume_data

            # Get current subscribed symbols
            if not self.active_subscriptions:
                self.logger.debug("No active subscriptions, skipping fundamental data collection")
                return

            symbols_list = list(self.active_subscriptions)

            if not symbols_list:
                return

            # Initialize Alpaca client if needed
            if self.alpaca_client is None:
                self._initialize_alpaca_client()

            # Fetch fundamental data for subscribed symbols
            fundamental_data = self._fetch_fundamental_data(symbols_list)
            if fundamental_data:
                # Merge with existing data
                self.fundamental_data.update(fundamental_data)

            # Collect hourly volume data for float rotation calculation
            hourly_volume = await self._collect_hourly_volume_data(symbols_list)
            if hourly_volume:
                self.hourly_volume_data.update(hourly_volume)

            self.logger.info(
                f"✅ Updated volume surge data ({len(self.volume_surge_data)} symbols), "
                f"fundamental data ({len(self.fundamental_data)} symbols), and "
                f"hourly volume data ({len(self.hourly_volume_data)} symbols)"
            )

        except Exception as e:
            self.logger.error(f"❌ Error collecting volume and fundamental data: {e}")
            import traceback
            traceback.print_exc()

    async def _volume_fundamental_collection_task(self):
        """
        Background task that collects volume surge and fundamental data every 5 minutes.

        Runs continuously while the monitor is active.
        """
        self.logger.info("🔄 Starting volume/fundamental data collection background task")

        # Initial collection
        await self._collect_volume_and_fundamental_data()

        while True:
            try:
                # Wait 5 minutes (300 seconds)
                await asyncio.sleep(300)

                # Collect data
                await self._collect_volume_and_fundamental_data()

            except asyncio.CancelledError:
                self.logger.info("⚠️  Volume/fundamental data collection task cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in volume/fundamental data collection task: {e}")
                import traceback
                traceback.print_exc()
                # Continue running even if there's an error
                await asyncio.sleep(300)

    def _get_all_symbols_to_monitor(self) -> List[str]:
        """
        Get all symbols to monitor from all sources (manual symbols + existing list).
        Always includes SPY for market context analysis.

        Returns:
            Combined list of unique symbols
        """
        all_symbols = set()

        # Add existing symbols (from initial load or use_existing)
        all_symbols.update(self.symbols_to_monitor)

        # Add manual symbols
        manual_symbols = self._load_manual_symbols()
        all_symbols.update(manual_symbols)

        # Always include SPY for market context (Phase 1 enhancement)
        all_symbols.add('SPY')

        # Convert to sorted list
        return sorted(list(all_symbols))

    async def connect_and_monitor(self):
        """Connect to WebSocket and start monitoring trades."""
        self.start_time = datetime.now()
        self.logger.info("="*70)
        self.logger.info("Squeeze Alerts Monitor - Starting")
        self.logger.info("="*70)
        self.logger.info(f"WebSocket URL: {self.websocket_url}")
        self.logger.info(f"Symbols to monitor: {len(self.symbols_to_monitor)}")
        if self.verbose:
            self.logger.debug(f"Symbol list: {', '.join(self.symbols_to_monitor[:10])}{'...' if len(self.symbols_to_monitor) > 10 else ''}")
        self.logger.info("="*70)

        try:
            async with websockets.connect(self.websocket_url) as ws:
                self.logger.info("✅ Connected to WebSocket server")

                # Query existing symbols if use_existing mode
                if self.use_existing:
                    await self._query_and_use_existing_symbols(ws)

                # Subscribe to all symbols
                await self._subscribe_all_symbols(ws)

                # Start gains collection background task
                gains_task = asyncio.create_task(self._gains_collection_task())
                self.logger.info("✅ Started gains collection background task")

                # Start volume/fundamental data collection background task
                volume_fund_task = asyncio.create_task(self._volume_fundamental_collection_task())
                self.logger.info("✅ Started volume/fundamental data collection background task")

                try:
                    # Start monitoring loop
                    await self._monitor_trades(ws)
                finally:
                    # Cancel background tasks when monitoring stops
                    gains_task.cancel()
                    volume_fund_task.cancel()
                    try:
                        await gains_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await volume_fund_task
                    except asyncio.CancelledError:
                        pass

        except ConnectionRefusedError:
            self.logger.error(f"❌ Connection refused to {self.websocket_url}")
            self.logger.error("   Make sure the WebSocket server is running:")
            self.logger.error("   ./services/restart_websocket_services.sh")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"❌ WebSocket error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    async def _query_and_use_existing_symbols(self, ws):
        """Query backend for existing symbol subscriptions and use them."""
        self.logger.info("🔍 Querying backend for existing symbol subscriptions...")

        # Send health check to get active symbols
        health_msg = json.dumps({"action": "health"})
        await ws.send(health_msg)

        try:
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)

            if data['type'] == 'health':
                active_symbols = data.get('active_symbols', [])
                if active_symbols:
                    self.symbols_to_monitor = active_symbols
                    self.logger.info(f"✅ Found {len(active_symbols)} existing subscriptions: {', '.join(active_symbols)}")
                else:
                    self.logger.warning("⚠️  No existing subscriptions found, will not subscribe to any defaults")
                    self.symbols_to_monitor = []
            else:
                self.logger.warning(f"⚠️  Unexpected response type: {data['type']}")

        except asyncio.TimeoutError:
            self.logger.error("❌ Timeout waiting for health response")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"❌ Error querying existing symbols: {e}")
            sys.exit(1)

    async def _refresh_existing_symbols(self, ws):
        """
        Re-query backend for existing symbol subscriptions during refresh cycle.

        This is called periodically (every 60 seconds) to pick up new symbols
        that other clients (like momentum_alerts) may have subscribed to.
        """
        self.logger.debug("🔄 Refreshing existing symbol subscriptions from backend...")

        # Send health check to get active symbols
        health_msg = json.dumps({"action": "health"})
        await ws.send(health_msg)

        try:
            # Wait for health response - may need to consume trade messages first
            max_attempts = 50  # Limit attempts to avoid infinite loop
            for attempt in range(max_attempts):
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(response)

                if data['type'] == 'health':
                    active_symbols = data.get('active_symbols', [])

                    # Get previously monitored symbols
                    old_symbols = set(self.symbols_to_monitor)
                    new_symbol_set = set(active_symbols)

                    # Find newly added symbols
                    added_symbols = new_symbol_set - old_symbols
                    removed_symbols = old_symbols - new_symbol_set

                    if added_symbols:
                        self.logger.info(f"🆕 Found {len(added_symbols)} new symbols from backend: {', '.join(sorted(added_symbols))}")

                    if removed_symbols:
                        self.logger.info(f"📉 {len(removed_symbols)} symbols no longer active: {', '.join(sorted(removed_symbols))}")

                    # Update monitored symbols
                    self.symbols_to_monitor = active_symbols

                    if not active_symbols:
                        self.logger.debug("🔄 No active subscriptions found during refresh")

                    # Successfully received health response
                    break
                elif data['type'] == 'trade':
                    # Skip trade messages and continue waiting for health response
                    continue
                else:
                    self.logger.warning(f"⚠️  Unexpected response type during refresh: {data['type']}")
                    continue

        except asyncio.TimeoutError:
            self.logger.warning("⚠️  Timeout waiting for health response during refresh (will retry next cycle)")
        except Exception as e:
            self.logger.warning(f"⚠️  Error refreshing existing symbols: {e} (will retry next cycle)")

    async def _subscribe_all_symbols(self, ws, refresh: bool = False):
        """
        Subscribe to all symbols in the monitor list.

        Args:
            ws: WebSocket connection
            refresh: If True, this is a periodic refresh and only subscribes to new symbols
        """
        # Get all symbols including manual ones
        all_symbols = self._get_all_symbols_to_monitor()

        if not all_symbols:
            self.logger.warning("⚠️  No symbols to subscribe to")
            # Still update last subscription time so refresh cycle can trigger
            self.last_subscription_time = datetime.now()
            return

        # Determine which symbols need subscription
        if refresh:
            # Only subscribe to symbols not already subscribed
            symbols_to_subscribe = [s for s in all_symbols if s not in self.active_subscriptions]
            if not symbols_to_subscribe:
                self.logger.debug("🔄 All symbols already subscribed")
                # Still update last subscription time to maintain refresh cycle
                self.last_subscription_time = datetime.now()
                return
            self.logger.info(f"🔄 Subscribing to {len(symbols_to_subscribe)} new symbols (refresh)...")
        else:
            # Initial subscription - subscribe to all
            symbols_to_subscribe = all_symbols
            self.logger.info(f"📤 Subscribing to {len(symbols_to_subscribe)} symbols...")

        # Send subscription requests
        for symbol in symbols_to_subscribe:
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "symbol": symbol
            })
            await ws.send(subscribe_msg)
            if self.verbose:
                self.logger.debug(f"   → Sent subscribe request for {symbol}")

        # Wait for subscription confirmations
        confirmed = 0
        timeout_count = 0
        max_messages = len(symbols_to_subscribe) * 3  # Allow for 'connecting' + 'subscribed' + buffer

        if not refresh:
            self.logger.info("⏳ Waiting for subscription confirmations...")

        for _ in range(max_messages):
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(response)

                if data['type'] == 'connecting':
                    if self.verbose:
                        self.logger.debug(f"   🔄 {data.get('symbol', 'unknown')}: Connecting...")
                elif data['type'] == 'subscribed':
                    symbol = data.get('symbol')
                    self.active_subscriptions.add(symbol)
                    confirmed += 1
                    if refresh:
                        self.logger.info(f"   ✅ Subscribed to {symbol}")
                    else:
                        self.logger.info(f"   ✅ Subscribed to {symbol} ({confirmed}/{len(symbols_to_subscribe)})")
                elif data['type'] == 'error':
                    self.logger.error(f"   ❌ Subscription error: {data.get('message')}")
                elif data['type'] == 'trade':
                    # During refresh, we might receive trades - just continue waiting
                    continue

                # Exit early if all subscriptions confirmed
                if confirmed == len(symbols_to_subscribe):
                    break

            except asyncio.TimeoutError:
                timeout_count += 1
                if timeout_count > 5:  # Stop waiting after 5 consecutive timeouts
                    break

        if refresh and confirmed > 0:
            self.logger.info(f"✅ Subscribed to {confirmed} new symbols")
        elif not refresh:
            self.logger.info(f"✅ Successfully subscribed to {confirmed}/{len(symbols_to_subscribe)} symbols")
            if confirmed < len(symbols_to_subscribe):
                self.logger.warning(f"⚠️  {len(symbols_to_subscribe) - confirmed} symbols failed to subscribe")

        # Update last subscription time
        self.last_subscription_time = datetime.now()

    def _load_squeeze_counts(self):
        """
        Load squeeze counts by counting existing alert files for today.

        This is the source of truth - counts actual alert files created today.
        Cannot become corrupted and self-heals on restart.
        """
        try:
            # Count alert files for each symbol: alert_{SYMBOL}_{DATE}_*.json
            counts = {}

            if self.squeeze_alerts_sent_dir.exists():
                # Get all alert files for today
                alert_files = list(self.squeeze_alerts_sent_dir.glob('alert_*_*.json'))

                # Count files per symbol
                for alert_file in alert_files:
                    # Filename format: alert_SYMBOL_YYYY-MM-DD_HHMMSS.json
                    parts = alert_file.stem.split('_')
                    if len(parts) >= 2:
                        symbol = parts[1]  # Second part is symbol
                        counts[symbol] = counts.get(symbol, 0) + 1

                self.squeezes_by_symbol = counts
                total = sum(counts.values())
                self.logger.info(f"✅ Counted {total} squeezes across {len(counts)} symbols from alert files for {self.today}")
            else:
                self.squeezes_by_symbol = {}
                self.logger.info(f"📝 No alert directory for {self.today}, starting fresh")

        except Exception as e:
            self.logger.error(f"❌ Error counting squeeze alerts: {e}")
            import traceback
            traceback.print_exc()
            self.squeezes_by_symbol = {}

    def _check_date_change(self):
        """
        Check if the date has changed and update directories if needed.

        This handles the case where the system runs past midnight and needs
        to update from one day's directories to the next day's directories.
        """
        current_date = datetime.now(self.et_tz).strftime('%Y-%m-%d')

        if current_date != self.today:
            old_date = self.today
            self.logger.info(f"📅 Date changed from {old_date} to {current_date}")
            self._update_date_directories(current_date)

    def _update_date_directories(self, new_date: str):
        """
        Update all date-dependent directory paths to use the new date.
        Also resets squeeze counts for the new day.

        Args:
            new_date: New date string in YYYY-MM-DD format
        """
        try:
            self.logger.info(f"🔄 Updating directory paths for new date: {new_date}")

            # Archive old squeeze counts before resetting
            old_count = len(self.squeezes_by_symbol)
            old_total = sum(self.squeezes_by_symbol.values())

            # Update the date
            self.today = new_date

            # Update directory path
            self.squeeze_alerts_sent_dir = Path(project_root) / "historical_data" / self.today / "squeeze_alerts_sent"

            # Create new directory
            self.squeeze_alerts_sent_dir.mkdir(parents=True, exist_ok=True)

            # Reset squeeze counts for new day (will be recounted from alert files)
            self.logger.info(f"📊 Archived {old_total} squeezes across {old_count} symbols from previous day")
            self.squeezes_by_symbol = {}

            self.logger.info(f"✅ Directory paths updated successfully for {new_date}")
            self.logger.info(f"📁 Squeeze alerts dir: {self.squeeze_alerts_sent_dir}")

        except Exception as e:
            self.logger.error(f"❌ Error updating directory paths: {e}")

    def _is_gains_collection_window(self) -> bool:
        """
        Check if current time is within the extended trading hours (0400-2000 ET).

        Returns:
            True if within the collection window, False otherwise
        """
        current_et = datetime.now(self.et_tz)
        current_time = current_et.time()
        current_weekday = current_et.weekday()  # Monday = 0, Sunday = 6

        # Only Monday-Friday
        if current_weekday >= 5:
            return False

        # Check if between extended hours (04:00 AM - 08:00 PM ET)
        return EXTENDED_HOURS_START <= current_time <= EXTENDED_HOURS_END

    def _initialize_alpaca_client(self):
        """Initialize Alpaca client if not already initialized."""
        if self.alpaca_client is None:
            try:
                self.alpaca_client = init_alpaca_client("alpaca", "Bruce", "paper")
                self.logger.debug("✅ Initialized Alpaca client for gains collection")
            except Exception as e:
                self.logger.error(f"❌ Error initializing Alpaca client: {e}")
                raise

    async def _collect_gains_data(self):
        """
        Collect premarket gains data for all subscribed symbols.

        This method runs every 30 seconds and calls collect_premarket_data
        to get gains for each subscribed symbol.
        """
        try:
            # Get current subscribed symbols
            if not self.active_subscriptions:
                self.logger.debug("No active subscriptions, skipping gains collection")
                return

            symbols_list = list(self.active_subscriptions)

            if not symbols_list:
                return

            self.logger.debug(f"🔄 Collecting gains data for {len(symbols_list)} symbols")

            # Initialize client if needed
            if self.alpaca_client is None:
                self._initialize_alpaca_client()

            # Create a simple criteria object
            class PremarketCriteria:
                def __init__(self):
                    self.feed = "sip"
                    self.lookback_days = 7
                    self.min_price = None
                    self.max_price = None
                    self.min_volume = None
                    self.min_gain_percent = None

            criteria = PremarketCriteria()

            # Call collect_premarket_data
            premarket_data = collect_premarket_data(
                client=self.alpaca_client,
                symbols=symbols_list,
                criteria=criteria,
                et_tz=self.et_tz,
                market_close=MARKET_CLOSE,
                verbose=self.verbose,
                tracked_symbols=[]
            )

            # Process and store gains
            for symbol, data in premarket_data.items():
                if 'premarket_bars' in data and not data['premarket_bars'].empty:
                    # Calculate gain percentage
                    previous_close = data['previous_close']
                    current_bar = data['premarket_bars'].iloc[-1]
                    current_price = float(current_bar['close'])
                    gain_percent = ((current_price - previous_close) / previous_close) * 100

                    # Store in class variable
                    self.gains_per_symbol[symbol] = {
                        'current_price': current_price,
                        'previous_close': previous_close,
                        'gain_percent': gain_percent,
                        'previous_close_time': data['previous_close_time'],
                        'last_updated': datetime.now(self.et_tz)
                    }

            self.last_gains_update = datetime.now(self.et_tz)
            self.logger.info(f"✅ Updated gains for {len(self.gains_per_symbol)} symbols")

        except Exception as e:
            self.logger.error(f"❌ Error collecting gains data: {e}")
            import traceback
            traceback.print_exc()

    async def _gains_collection_task(self):
        """
        Background task that collects gains data every 30 seconds.

        Runs from 0400 to 2000 ET.
        """
        self.logger.info("🔄 Starting gains collection background task")

        while True:
            try:
                # Wait 30 seconds
                await asyncio.sleep(30)

                # Check if we're in the collection window
                if not self._is_gains_collection_window():
                    if self.verbose:
                        self.logger.debug("Outside gains collection window (0400-2000 ET)")
                    continue

                # Collect gains data
                await self._collect_gains_data()

            except asyncio.CancelledError:
                self.logger.info("⚠️  Gains collection task cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in gains collection task: {e}")
                import traceback
                traceback.print_exc()
                # Continue running even if there's an error
                await asyncio.sleep(30)

    async def _monitor_trades(self, ws):
        """Monitor and log incoming trades (Phase 1: Eavesdropping)."""
        self.logger.info("="*70)
        if self.test_mode:
            self.logger.info(f"🧪 TEST MODE - Running for {self.test_duration} seconds")
        else:
            self.logger.info("🎧 EAVESDROPPING MODE - Monitoring live trades")
            self.logger.info("   Refreshing subscriptions every 60 seconds")
            self.logger.info("   Press Ctrl+C to stop")
        self.logger.info("="*70)

        # Track end time for test mode
        end_time = None
        if self.test_mode:
            end_time = asyncio.get_event_loop().time() + self.test_duration

        try:
            while True:
                # Check if date has changed (e.g., past midnight)
                self._check_date_change()

                # Check if test duration has elapsed
                if self.test_mode and asyncio.get_event_loop().time() >= end_time:
                    raise KeyboardInterrupt("Test duration completed")

                # Check if we should refresh subscriptions (every 60 seconds)
                if self.last_subscription_time:
                    time_since_last_subscription = (datetime.now() - self.last_subscription_time).total_seconds()
                    if time_since_last_subscription >= 60:
                        # If in use_existing mode, re-query backend for new symbols
                        if self.use_existing:
                            await self._refresh_existing_symbols(ws)
                        await self._subscribe_all_symbols(ws, refresh=True)

                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(message)

                    if data['type'] == 'trade':
                        await self._handle_trade(data)
                    elif data['type'] == 'error':
                        self.logger.error(f"❌ Error: {data.get('message')}")
                    elif self.verbose:
                        self.logger.debug(f"📥 Received: {data['type']}")

                except asyncio.TimeoutError:
                    # Just continue to check time - no need to log
                    continue

        except KeyboardInterrupt:
            if self.test_mode:
                self.logger.info(f"\n⏱️  Test completed after {self.test_duration} seconds")
            else:
                self.logger.info("\n⚠️  Shutting down...")
            self._print_statistics()
        except Exception as e:
            self.logger.error(f"❌ Error in monitor loop: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_trade(self, trade_data: dict):
        """
        Handle incoming trade data and detect squeezes.

        Args:
            trade_data: Trade data from WebSocket
                {
                    'type': 'trade',
                    'symbol': 'AAPL',
                    'data': {
                        'timestamp': '2025-11-21T14:50:39.123456',
                        'price': 272.00,
                        'size': 100,
                        'exchange': 'Q',
                        'conditions': ['@']
                    }
                }
        """
        symbol = trade_data['symbol']
        data = trade_data['data']

        # Update statistics
        self.trade_count += 1
        self.trades_by_symbol[symbol] = self.trades_by_symbol.get(symbol, 0) + 1

        # Extract trade info
        price = data['price']
        size = data['size']
        exchange = data.get('exchange', 'N/A')
        timestamp_str = data['timestamp']

        # Parse timestamp and convert to ET
        from dateutil import parser as date_parser
        timestamp = date_parser.parse(timestamp_str)

        # Convert to ET timezone if not already
        if timestamp.tzinfo is None:
            # If naive, assume UTC
            timestamp = pytz.UTC.localize(timestamp).astimezone(self.et_tz)
        else:
            # If timezone-aware, convert to ET
            timestamp = timestamp.astimezone(self.et_tz)

        # Track volume history for Phase 1 metrics (5-minute rolling window)
        self._track_volume_history(symbol, timestamp, size)

        # Track day open and low prices for Phase 1 metrics
        self._track_day_prices(symbol, price, timestamp)

        # Track SPY price for concurrent change calculation (store latest SPY price)
        if symbol == 'SPY':
            self.latest_spy_price = price
            self.latest_spy_timestamp = timestamp

        # Check outcome tracking intervals for this symbol
        self._check_outcome_intervals(symbol, timestamp, price, size)

        # Log trade (suppress in test mode unless verbose)
        if not self.test_mode or self.verbose:
            self.logger.info(
                f"📊 {symbol:6s} @ ${price:8.2f} x {size:6,d} shares  "
                f"[{exchange}]  {timestamp_str}"
            )

        # Squeeze detection: Track prices in 10-second rolling window
        await self._detect_squeeze(symbol, timestamp, price, size)

    def _track_volume_history(self, symbol: str, timestamp: datetime, volume: int):
        """
        Track volume history for 1min and 5min rolling averages.

        Maintains a 5-minute rolling window of (timestamp, volume) tuples per symbol.
        """
        # Initialize volume history for this symbol if needed
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque()

        # Add current trade volume to history
        self.volume_history[symbol].append((timestamp, volume))

        # Remove volumes older than 5 minutes
        from datetime import timedelta
        cutoff_time = timestamp - timedelta(minutes=5)

        while (self.volume_history[symbol] and
               self.volume_history[symbol][0][0] < cutoff_time):
            self.volume_history[symbol].popleft()

    def _track_day_prices(self, symbol: str, price: float, timestamp: datetime):
        """
        Track day's open and low prices for distance calculations.

        Args:
            symbol: Stock symbol
            price: Current price
            timestamp: Trade timestamp
        """
        # Track day's open price (first trade at or after 9:30 AM ET)
        if symbol not in self.day_open_prices:
            # Check if this is during regular hours (9:30 AM - 4:00 PM ET)
            hour = timestamp.hour
            minute = timestamp.minute

            if (hour == 9 and minute >= 30) or (10 <= hour < 16):
                # This is the first trade we're seeing during regular hours
                self.day_open_prices[symbol] = price

        # Track day's low price (minimum price seen today)
        if symbol not in self.day_low_prices:
            self.day_low_prices[symbol] = price
        else:
            self.day_low_prices[symbol] = min(self.day_low_prices[symbol], price)

    async def _detect_squeeze(self, symbol: str, timestamp: datetime, price: float, size: int):
        """
        Detect if a squeeze is happening based on 10-second rolling window.

        A squeeze is detected when price increases by >= squeeze_percent in 10 seconds.

        Args:
            symbol: Stock symbol
            timestamp: Trade timestamp
            price: Trade price
            size: Trade size
        """
        # Initialize price history for this symbol if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque()

        # Store SPY price at window start (for concurrent change calculation)
        was_empty = len(self.price_history[symbol]) == 0

        # Add current trade to history (timestamp, price, size)
        self.price_history[symbol].append((timestamp, price, size))

        # If this is the first trade in a new window, store SPY price
        if was_empty and symbol != 'SPY' and self.latest_spy_price is not None:
            self.spy_squeeze_start_price[symbol] = self.latest_spy_price

        # Remove trades older than 10 seconds
        from datetime import timedelta
        cutoff_time = timestamp - timedelta(seconds=10)

        while self.price_history[symbol] and self.price_history[symbol][0][0] < cutoff_time:
            self.price_history[symbol].popleft()

        # Need at least 2 prices to detect a squeeze
        if len(self.price_history[symbol]) < 2:
            return

        # Get first and last prices in the window (chronologically)
        first_price = self.price_history[symbol][0][1]  # Oldest trade in window - (timestamp, price, size)
        last_price = self.price_history[symbol][-1][1]  # Newest trade in window - (timestamp, price, size)

        # Calculate percent change from first to last
        if first_price > 0:
            percent_change = ((last_price - first_price) / first_price) * 100

            # Check if we have a squeeze (only upward movement)
            if percent_change >= self.squeeze_percent:
                # Check if we already reported this squeeze recently
                # Only report once per symbol per 30-second window to avoid spam
                last_squeeze_time = getattr(self, '_last_squeeze_time', {})
                if symbol not in last_squeeze_time or (timestamp - last_squeeze_time[symbol]).total_seconds() > 30:
                    self._report_squeeze(symbol, first_price, last_price, percent_change, timestamp, size)

                    # Track last squeeze time
                    if not hasattr(self, '_last_squeeze_time'):
                        self._last_squeeze_time = {}
                    self._last_squeeze_time[symbol] = timestamp

    def _calculate_ema_values(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate EMA 9 and EMA 21 from 1-minute candlesticks.

        Args:
            symbol: Stock symbol
            timestamp: Current timestamp

        Returns:
            Dictionary containing EMA 9 and EMA 21 values, or None if calculation fails
        """
        try:
            import alpaca_trade_api as tradeapi
            import pandas as pd
            from atoms.utils.calculate_ema import calculate_ema

            if not self.alpaca_client:
                return {'ema_9': None, 'ema_21': None, 'error': 'Alpaca client not available'}

            # Calculate time range - fetch enough bars for EMA 21 calculation
            # Need at least 21 bars, but fetch 30 to have sufficient data
            end_time = timestamp
            start_time = timestamp - timedelta(minutes=35)

            # Fetch 1-minute bars
            bars = self.alpaca_client.get_bars(
                symbol,
                tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Minute),
                start=start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                end=end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                limit=35,
                feed='sip'
            )

            if not bars or len(bars) < 21:
                return {'ema_9': None, 'ema_21': None, 'error': f'Insufficient bars: {len(bars) if bars else 0}'}

            # Convert bars to DataFrame
            bar_data = []
            for bar in bars:
                bar_dict = {
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'timestamp': bar.t
                }
                bar_data.append(bar_dict)

            df = pd.DataFrame(bar_data)
            df.set_index('timestamp', inplace=True)

            # Calculate EMA 9
            success_9, ema_9_series = calculate_ema(df, period=9)

            # Calculate EMA 21
            success_21, ema_21_series = calculate_ema(df, period=21)

            if success_9 and success_21:
                # Get the latest (most recent) EMA values
                ema_9_value = ema_9_series.iloc[-1] if len(ema_9_series) > 0 else None
                ema_21_value = ema_21_series.iloc[-1] if len(ema_21_series) > 0 else None

                # Only return valid values (not zero)
                if ema_9_value and ema_9_value > 0 and ema_21_value and ema_21_value > 0:
                    return {
                        'ema_9': round(ema_9_value, 2),
                        'ema_21': round(ema_21_value, 2),
                        'error': None
                    }
                else:
                    return {'ema_9': None, 'ema_21': None, 'error': 'EMA values are zero or invalid'}
            else:
                return {'ema_9': None, 'ema_21': None, 'error': 'EMA calculation failed'}

        except Exception as e:
            self.logger.error(f"Error calculating EMAs for {symbol}: {e}")
            return {'ema_9': None, 'ema_21': None, 'error': str(e)}

    def _calculate_macd_values(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate MACD (Moving Average Convergence Divergence) from 1-minute candlesticks.

        Args:
            symbol: Stock symbol
            timestamp: Current timestamp

        Returns:
            Dictionary containing MACD, Signal, and Histogram values, or None if calculation fails
        """
        try:
            import alpaca_trade_api as tradeapi
            import pandas as pd
            from atoms.utils.calculate_macd import calculate_macd

            if not self.alpaca_client:
                return {'macd': None, 'signal': None, 'histogram': None, 'error': 'Alpaca client not available'}

            # Calculate time range - fetch enough bars for MACD calculation
            # Need at least 26 bars (slow EMA length), but fetch 35 to have sufficient data
            end_time = timestamp
            start_time = timestamp - timedelta(minutes=35)

            # Fetch 1-minute bars
            bars = self.alpaca_client.get_bars(
                symbol,
                tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Minute),
                start=start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                end=end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                limit=35,
                feed='sip'
            )

            if not bars or len(bars) < 26:
                return {'macd': None, 'signal': None, 'histogram': None, 'error': f'Insufficient bars: {len(bars) if bars else 0}'}

            # Convert bars to DataFrame
            bar_data = []
            for bar in bars:
                bar_dict = {
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'timestamp': bar.t
                }
                bar_data.append(bar_dict)

            df = pd.DataFrame(bar_data)
            df.set_index('timestamp', inplace=True)

            # Calculate MACD with default parameters (12, 26, 9)
            success, macd_result = calculate_macd(df, fast_length=12, slow_length=26,
                                                 signal_length=9, source='close')

            if success:
                # Get the latest (most recent) MACD values
                macd_value = macd_result['macd'].iloc[-1] if len(macd_result['macd']) > 0 else None
                signal_value = macd_result['signal'].iloc[-1] if len(macd_result['signal']) > 0 else None
                histogram_value = macd_result['histogram'].iloc[-1] if len(macd_result['histogram']) > 0 else None

                # Return values (MACD can be negative, so don't check > 0)
                if macd_value is not None and signal_value is not None and histogram_value is not None:
                    return {
                        'macd': round(macd_value, 4),
                        'signal': round(signal_value, 4),
                        'histogram': round(histogram_value, 4),
                        'error': None
                    }
                else:
                    return {'macd': None, 'signal': None, 'histogram': None, 'error': 'MACD values are invalid'}
            else:
                return {'macd': None, 'signal': None, 'histogram': None, 'error': 'MACD calculation failed'}

        except Exception as e:
            self.logger.error(f"Error calculating MACD for {symbol}: {e}")
            return {'macd': None, 'signal': None, 'histogram': None, 'error': str(e)}

    def _calculate_phase1_metrics(self, symbol: str, timestamp: datetime, last_price: float) -> Dict[str, Any]:
        """
        Calculate Phase 1 enhancement metrics for squeeze alerts.

        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            last_price: Current price

        Returns:
            Dictionary containing Phase 1 metrics
        """
        metrics = {}

        # ===== 1. TIMING METRICS =====
        # Calculate time since market open (9:30 AM ET)
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        time_since_open = (timestamp - market_open).total_seconds() / 60  # in minutes
        metrics['time_since_market_open_minutes'] = int(time_since_open)

        # Hour of day
        metrics['hour_of_day'] = timestamp.hour

        # Market session
        hour = timestamp.hour
        minute = timestamp.minute
        time_value = hour + minute / 60.0

        if 9.5 <= time_value < 11:
            metrics['market_session'] = 'early'
        elif 11 <= time_value < 14:
            metrics['market_session'] = 'mid_day'
        elif 14 <= time_value < 15:
            metrics['market_session'] = 'power_hour'
        elif 15 <= time_value <= 16:
            metrics['market_session'] = 'close'
        else:
            metrics['market_session'] = 'extended'

        # Squeeze number today for this symbol
        # Note: squeezes_by_symbol is already incremented in _report_squeeze before this is called
        metrics['squeeze_number_today'] = self.squeezes_by_symbol.get(symbol, 0)

        # Minutes since last squeeze
        if hasattr(self, '_last_squeeze_time') and symbol in self._last_squeeze_time:
            last_squeeze = self._last_squeeze_time[symbol]
            minutes_since = (timestamp - last_squeeze).total_seconds() / 60
            metrics['minutes_since_last_squeeze'] = round(minutes_since, 2)
        else:
            metrics['minutes_since_last_squeeze'] = None

        # ===== 2. VOLUME INTELLIGENCE =====
        # Calculate window volume from price_history
        if symbol in self.price_history:
            window_volume = sum(trade[2] for trade in self.price_history[symbol])  # trade[2] is size
            metrics['window_volume'] = int(window_volume)

            # Calculate volume vs recent averages from volume_history
            if symbol in self.volume_history and len(self.volume_history[symbol]) > 0:
                from datetime import timedelta

                # Calculate 1-minute average volume
                one_min_ago = timestamp - timedelta(minutes=1)
                one_min_volumes = [v for t, v in self.volume_history[symbol] if t >= one_min_ago]

                if len(one_min_volumes) > 0:
                    one_min_avg = sum(one_min_volumes) / len(one_min_volumes)
                    metrics['window_volume_vs_1min_avg'] = round(window_volume / one_min_avg, 2) if one_min_avg > 0 else None
                else:
                    metrics['window_volume_vs_1min_avg'] = None

                # Calculate 5-minute average volume
                five_min_volumes = [v for t, v in self.volume_history[symbol]]

                if len(five_min_volumes) > 0:
                    five_min_avg = sum(five_min_volumes) / len(five_min_volumes)
                    metrics['window_volume_vs_5min_avg'] = round(window_volume / five_min_avg, 2) if five_min_avg > 0 else None
                else:
                    metrics['window_volume_vs_5min_avg'] = None

                # Determine volume trend (compare recent 2min vs previous 3min)
                two_min_ago = timestamp - timedelta(minutes=2)
                recent_volumes = [v for t, v in self.volume_history[symbol] if t >= two_min_ago]
                older_volumes = [v for t, v in self.volume_history[symbol] if t < two_min_ago]

                if len(recent_volumes) > 0 and len(older_volumes) > 0:
                    recent_avg = sum(recent_volumes) / len(recent_volumes)
                    older_avg = sum(older_volumes) / len(older_volumes)

                    if recent_avg > older_avg * 1.2:  # 20% increase
                        metrics['volume_trend'] = 'increasing'
                    elif recent_avg < older_avg * 0.8:  # 20% decrease
                        metrics['volume_trend'] = 'decreasing'
                    else:
                        metrics['volume_trend'] = 'stable'
                else:
                    metrics['volume_trend'] = None
            else:
                metrics['window_volume_vs_1min_avg'] = None
                metrics['window_volume_vs_5min_avg'] = None
                metrics['volume_trend'] = None
        else:
            metrics['window_volume'] = None
            metrics['window_volume_vs_1min_avg'] = None
            metrics['window_volume_vs_5min_avg'] = None
            metrics['volume_trend'] = None

        # ===== 3. PRICE LEVEL CONTEXT =====
        # Get gain data for previous close
        gain_data = self.gains_per_symbol.get(symbol)

        if gain_data:
            prev_close = gain_data.get('previous_close')
            if prev_close and prev_close > 0:
                metrics['distance_from_prev_close_percent'] = round(
                    ((last_price - prev_close) / prev_close) * 100, 2)
            else:
                metrics['distance_from_prev_close_percent'] = None
        else:
            metrics['distance_from_prev_close_percent'] = None

        # Distance from VWAP
        candlestick = self.market_data.get_latest_candlestick(symbol)
        vwap = candlestick.get('vwap')

        if vwap and vwap > 0:
            metrics['distance_from_vwap_percent'] = round(
                ((last_price - vwap) / vwap) * 100, 2)
        else:
            metrics['distance_from_vwap_percent'] = None

        # Distance from day low (using tracked day_low_prices)
        if symbol in self.day_low_prices and self.day_low_prices[symbol] > 0:
            metrics['distance_from_day_low_percent'] = round(
                ((last_price - self.day_low_prices[symbol]) / self.day_low_prices[symbol]) * 100, 2)
        else:
            metrics['distance_from_day_low_percent'] = None

        # Distance from day open (using tracked day_open_prices)
        if symbol in self.day_open_prices and self.day_open_prices[symbol] > 0:
            metrics['distance_from_open_percent'] = round(
                ((last_price - self.day_open_prices[symbol]) / self.day_open_prices[symbol]) * 100, 2)
        else:
            metrics['distance_from_open_percent'] = None

        # ===== 4. RISK/REWARD METRICS =====
        # Get day highs for target calculation
        day_highs = self.market_data.get_day_highs(symbol, timestamp)
        regular_hod = day_highs.get('regular_hours_hod')

        # Estimated stop loss (7.5% below current price as default)
        stop_loss_percent = 7.5
        estimated_stop = last_price * (1 - stop_loss_percent / 100)
        metrics['estimated_stop_loss_price'] = round(estimated_stop, 4)
        metrics['stop_loss_distance_percent'] = stop_loss_percent

        # Potential target (assume HOD if available, otherwise 10% above current)
        if regular_hod and regular_hod > last_price:
            potential_target = regular_hod
        else:
            potential_target = last_price * 1.10  # 10% above current

        metrics['potential_target_price'] = round(potential_target, 4)

        # Risk/reward ratio
        potential_gain = potential_target - last_price
        potential_loss = last_price - estimated_stop

        if potential_loss > 0:
            metrics['risk_reward_ratio'] = round(potential_gain / potential_loss, 2)
        else:
            metrics['risk_reward_ratio'] = None

        # ===== 5. MARKET CONTEXT (SPY) =====
        # Get SPY gain data
        spy_data = self.gains_per_symbol.get('SPY')

        if spy_data:
            spy_gain = spy_data.get('gain_percent')
            metrics['spy_percent_change_day'] = round(spy_gain, 2) if spy_gain is not None else None
        else:
            metrics['spy_percent_change_day'] = None

        # SPY concurrent change (SPY movement during the same squeeze window)
        if (symbol in self.spy_squeeze_start_price and
            self.latest_spy_price is not None and
            self.spy_squeeze_start_price[symbol] > 0):

            spy_start = self.spy_squeeze_start_price[symbol]
            spy_end = self.latest_spy_price
            spy_change = ((spy_end - spy_start) / spy_start) * 100
            metrics['spy_percent_change_concurrent'] = round(spy_change, 2)
        else:
            metrics['spy_percent_change_concurrent'] = None

        return metrics

    # ===== OUTCOME TRACKING METHODS =====

    def _start_outcome_tracking(self, symbol: str, squeeze_timestamp: datetime,
                                squeeze_price: float, alert_filename: str) -> None:
        """
        Initialize outcome tracking for a squeeze alert.

        Called from _report_squeeze() after saving the alert JSON file.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            squeeze_timestamp: Datetime when squeeze was detected
            squeeze_price: Price at squeeze detection (entry price for tracking)
            alert_filename: Name of the alert JSON file (to update later with outcomes)
        """
        if not self.OUTCOME_TRACKING_ENABLED:
            return

        # Check concurrent tracking limit
        if len(self.active_followups) >= self.OUTCOME_MAX_CONCURRENT_FOLLOWUPS:
            self.logger.warning(
                f"⚠️  Outcome tracking limit reached ({self.OUTCOME_MAX_CONCURRENT_FOLLOWUPS}). "
                f"Skipping tracking for {symbol}"
            )
            return

        # Create unique key for this followup: symbol_timestamp
        # Format: "AAPL_2025-12-12_152045"
        key = f"{symbol}_{squeeze_timestamp.strftime('%Y-%m-%d_%H%M%S')}"

        # Check if already tracking (shouldn't happen, but defensive)
        if key in self.active_followups:
            self.logger.warning(
                f"⚠️  Already tracking outcomes for {key}, skipping duplicate"
            )
            return

        # Calculate tracking window end time
        end_time = squeeze_timestamp + timedelta(
            minutes=self.OUTCOME_TRACKING_DURATION_MINUTES
        )

        # Calculate first interval time (first interval in seconds list)
        first_interval_seconds = self.OUTCOME_TRACKING_INTERVALS_SECONDS[0]
        first_interval_time = squeeze_timestamp + timedelta(seconds=first_interval_seconds)

        # Initialize target threshold tracking
        reached_targets = {}
        for threshold in self.OUTCOME_TARGET_THRESHOLDS:
            reached_targets[threshold] = {
                'reached': False,
                'minute': None,
                'price': None,
                'timestamp': None
            }

        # Initialize profitable snapshots for each interval (keyed by seconds)
        profitable_snapshots = {
            interval_sec: None for interval_sec in self.OUTCOME_TRACKING_INTERVALS_SECONDS
        }

        # Create followup data structure
        self.active_followups[key] = {
            # Identification
            'symbol': symbol,
            'squeeze_timestamp': squeeze_timestamp,
            'squeeze_price': squeeze_price,
            'alert_filename': alert_filename,

            # Tracking window
            'start_time': squeeze_timestamp,
            'end_time': end_time,

            # Interval tracking state (using index into OUTCOME_TRACKING_INTERVALS_SECONDS)
            'next_interval_index': 0,
            'next_interval_time': first_interval_time,
            'intervals_recorded': [],  # Will store interval times in seconds
            'interval_data': {},  # Keyed by seconds (e.g., 10, 20, 30, 60, etc.)

            # Running statistics (updated on every trade)
            'max_price_seen': squeeze_price,
            'min_price_seen': squeeze_price,
            'max_gain_percent': 0.0,
            'max_gain_minute': 0,
            'max_gain_timestamp': squeeze_timestamp,
            'max_drawdown_percent': 0.0,
            'max_drawdown_minute': 0,
            'max_drawdown_timestamp': squeeze_timestamp,

            # Stop loss tracking
            'reached_stop_loss': False,
            'stop_loss_minute': None,
            'stop_loss_price': None,
            'stop_loss_timestamp': None,

            # Target threshold tracking
            'reached_targets': reached_targets,

            # Profitability snapshots
            'profitable_snapshots': profitable_snapshots,

            # Last seen price (for handling gaps)
            'last_seen_price': squeeze_price,
            'last_seen_timestamp': squeeze_timestamp
        }

        # Initialize cumulative counters
        self.followup_volume_tracking[key] = 0
        self.followup_trades_tracking[key] = 0

        self.logger.info(
            f"📊 Started outcome tracking for {symbol} "
            f"(entry: ${squeeze_price:.4f}, duration: {self.OUTCOME_TRACKING_DURATION_MINUTES}min, "
            f"key: {key})"
        )

    def _check_outcome_intervals(self, symbol: str, timestamp: datetime,
                                  price: float, size: int) -> None:
        """
        Check if any outcome intervals are due for recording.

        Called from _handle_trade() on EVERY trade for symbols with active followups.
        Updates running statistics and records interval data when interval times are reached.

        Args:
            symbol: Stock symbol
            timestamp: Current trade timestamp
            price: Current trade price
            size: Current trade size
        """
        if not self.OUTCOME_TRACKING_ENABLED:
            return

        # Find all active followups for this symbol
        # Multiple squeezes for same symbol can be tracked concurrently
        keys_to_check = [k for k in self.active_followups.keys()
                        if k.startswith(f"{symbol}_")]

        if not keys_to_check:
            return  # No active tracking for this symbol

        keys_to_finalize = []

        for key in keys_to_check:
            followup = self.active_followups[key]

            # Update cumulative volume and trades
            self.followup_volume_tracking[key] += size
            self.followup_trades_tracking[key] += 1

            # Update last seen price (for gap handling)
            followup['last_seen_price'] = price
            followup['last_seen_timestamp'] = timestamp

            # Update running statistics (max/min, stop loss, targets)
            self._update_followup_statistics(key, price, timestamp)

            # Check if tracking period has ended
            if timestamp >= followup['end_time']:
                keys_to_finalize.append(key)
                continue

            # Check if market has closed (don't track into extended hours)
            if timestamp.time() >= datetime.strptime("16:00:00", "%H:%M:%S").time():
                self.logger.info(
                    f"🔔 Market closed, finalizing outcome tracking for {symbol} "
                    f"at {timestamp.strftime('%H:%M:%S')}"
                )
                keys_to_finalize.append(key)
                continue

            # Check if next interval is due
            next_interval_index = followup['next_interval_index']

            # Check if we have more intervals to track
            if next_interval_index >= len(self.OUTCOME_TRACKING_INTERVALS_SECONDS):
                # All intervals recorded, finalize
                keys_to_finalize.append(key)
                continue

            next_interval_time = followup['next_interval_time']
            interval_seconds = self.OUTCOME_TRACKING_INTERVALS_SECONDS[next_interval_index]

            # Define tolerance window
            tolerance = timedelta(seconds=self.OUTCOME_INTERVAL_TOLERANCE_SECONDS)

            # Check if current trade is within tolerance of next interval time
            if timestamp >= (next_interval_time - tolerance):
                # Record this interval
                self._record_outcome_interval(
                    key=key,
                    interval_seconds=interval_seconds,
                    timestamp=timestamp,
                    price=price,
                    volume=self.followup_volume_tracking[key],
                    trades=self.followup_trades_tracking[key]
                )

                # Advance to next interval
                next_interval_index += 1

                if next_interval_index < len(self.OUTCOME_TRACKING_INTERVALS_SECONDS):
                    # More intervals to track
                    followup['next_interval_index'] = next_interval_index
                    next_interval_seconds = self.OUTCOME_TRACKING_INTERVALS_SECONDS[next_interval_index]
                    followup['next_interval_time'] = followup['start_time'] + timedelta(
                        seconds=next_interval_seconds
                    )
                else:
                    # All intervals recorded, finalize
                    keys_to_finalize.append(key)

        # Finalize completed followups
        for key in keys_to_finalize:
            self._finalize_outcome_tracking(key)

    def _update_followup_statistics(self, key: str, price: float,
                                    timestamp: datetime) -> None:
        """
        Update running statistics for an active followup.

        Called on EVERY trade during the tracking period to capture:
        - Maximum price and gain (and when they occurred)
        - Minimum price and drawdown (and when they occurred)
        - Stop loss hits
        - Target threshold achievements

        This ensures we capture rapid moves that happen between interval snapshots.

        Args:
            key: Followup key (symbol_timestamp format)
            price: Current trade price
            timestamp: Current trade timestamp
        """
        followup = self.active_followups[key]
        squeeze_price = followup['squeeze_price']

        # Calculate gain/loss from squeeze entry price
        gain_percent = ((price - squeeze_price) / squeeze_price) * 100

        # Calculate elapsed time in minutes
        elapsed = (timestamp - followup['start_time']).total_seconds() / 60
        elapsed_minute = int(elapsed) + 1  # Convert to 1-indexed minute

        # Update maximum price and gain
        if price > followup['max_price_seen']:
            followup['max_price_seen'] = price
            followup['max_gain_percent'] = gain_percent
            followup['max_gain_minute'] = elapsed_minute
            followup['max_gain_timestamp'] = timestamp

            if self.verbose:
                self.logger.debug(
                    f"📈 {followup['symbol']} new high: ${price:.4f} "
                    f"({gain_percent:+.2f}%) at T+{elapsed_minute}min"
                )

        # Update minimum price and drawdown
        if price < followup['min_price_seen']:
            followup['min_price_seen'] = price
            followup['max_drawdown_percent'] = gain_percent
            followup['max_drawdown_minute'] = elapsed_minute
            followup['max_drawdown_timestamp'] = timestamp

            if self.verbose:
                self.logger.debug(
                    f"📉 {followup['symbol']} new low: ${price:.4f} "
                    f"({gain_percent:+.2f}%) at T+{elapsed_minute}min"
                )

        # Check stop loss threshold
        stop_loss_threshold = -self.OUTCOME_STOP_LOSS_PERCENT
        if not followup['reached_stop_loss'] and gain_percent <= stop_loss_threshold:
            followup['reached_stop_loss'] = True
            followup['stop_loss_price'] = price
            followup['stop_loss_minute'] = elapsed_minute
            followup['stop_loss_timestamp'] = timestamp

            self.logger.warning(
                f"🛑 {followup['symbol']} hit stop loss: ${price:.4f} "
                f"({gain_percent:.2f}%) at T+{elapsed_minute}min"
            )

        # Check target thresholds
        for threshold in self.OUTCOME_TARGET_THRESHOLDS:
            target_info = followup['reached_targets'][threshold]

            if not target_info['reached'] and gain_percent >= threshold:
                target_info['reached'] = True
                target_info['price'] = price
                target_info['minute'] = elapsed_minute
                target_info['timestamp'] = timestamp

                self.logger.info(
                    f"🎯 {followup['symbol']} hit +{threshold}% target: ${price:.4f} "
                    f"at T+{elapsed_minute}min"
                )

    def _record_outcome_interval(self, key: str, interval_seconds: int,
                                  timestamp: datetime, price: float,
                                  volume: int, trades: int) -> None:
        """
        Record data snapshot for a specific outcome interval.

        Called when a trade occurs at (or near) an interval time.
        Stores the price, volume, and other metrics at this point in time.

        Args:
            key: Followup key
            interval_seconds: Interval time in seconds (e.g., 10, 20, 30, 60, 120, etc.)
            timestamp: Trade timestamp (actual time, may differ slightly from target)
            price: Trade price at this interval
            volume: Cumulative volume since squeeze start
            trades: Cumulative number of trades since squeeze start
        """
        followup = self.active_followups[key]
        squeeze_price = followup['squeeze_price']

        # Calculate gain from entry
        gain_percent = ((price - squeeze_price) / squeeze_price) * 100

        # Record interval data (keyed by seconds)
        followup['interval_data'][interval_seconds] = {
            'timestamp': timestamp.isoformat(),
            'price': float(price),
            'volume_since_squeeze': int(volume),
            'trades_since_squeeze': int(trades),
            'gain_percent': round(gain_percent, 2)
        }

        # Mark this interval as recorded
        followup['intervals_recorded'].append(interval_seconds)

        # Record profitability snapshot
        followup['profitable_snapshots'][interval_seconds] = (price > squeeze_price)

        # Format interval display (show seconds for <60s, otherwise minutes)
        if interval_seconds < 60:
            interval_display = f"{interval_seconds}s"
        else:
            interval_display = f"{interval_seconds // 60}min"

        if self.verbose:
            self.logger.debug(
                f"📊 {followup['symbol']} T+{interval_display}: "
                f"${price:.4f} ({gain_percent:+.2f}%) "
                f"[{len(followup['intervals_recorded'])}/{len(self.OUTCOME_TRACKING_INTERVALS_SECONDS)} intervals]"
            )

    def _build_outcome_summary(self, followup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build summary statistics from followup data.

        Called when outcome tracking is complete to calculate final metrics
        and aggregate statistics for analysis.

        Args:
            followup: Followup data dictionary

        Returns:
            Dictionary containing summary statistics
        """
        # Get final interval data (T+600s=10min or last recorded interval)
        final_interval_seconds = 600  # 10 minutes in seconds

        if final_interval_seconds in followup['interval_data']:
            final_data = followup['interval_data'][final_interval_seconds]
            final_price = final_data['price']
            final_gain = final_data['gain_percent']
        elif followup['intervals_recorded']:
            # Use last recorded interval if T+10min not reached
            last_interval = max(followup['intervals_recorded'])
            final_data = followup['interval_data'][last_interval]
            final_price = final_data['price']
            final_gain = final_data['gain_percent']
        else:
            # No intervals recorded (shouldn't happen, but defensive)
            final_price = followup['squeeze_price']
            final_gain = 0.0

        # Build target achievement summary
        targets_achieved = {}
        for threshold in self.OUTCOME_TARGET_THRESHOLDS:
            target_info = followup['reached_targets'][threshold]
            threshold_int = int(threshold)

            targets_achieved[f'achieved_{threshold_int}pct'] = target_info['reached']
            targets_achieved[f'time_to_{threshold_int}pct_minutes'] = target_info['minute']
            targets_achieved[f'price_at_{threshold_int}pct'] = (
                float(target_info['price']) if target_info['price'] is not None else None
            )

        # Build profitability summary for key intervals (in seconds)
        profitable_at = {}
        key_intervals = [
            (10, '10s'), (20, '20s'), (30, '30s'),  # Sub-minute snapshots
            (60, '1min'), (120, '2min'), (300, '5min'), (600, '10min')  # Minute snapshots
        ]
        for interval_sec, label in key_intervals:
            if interval_sec in followup['profitable_snapshots']:
                profitable_at[f'profitable_at_{label}'] = (
                    followup['profitable_snapshots'][interval_sec]
                )

        # Build complete summary
        summary = {
            # Max/min statistics
            'max_price': float(followup['max_price_seen']),
            'max_gain_percent': round(followup['max_gain_percent'], 2),
            'max_gain_reached_at_minute': followup['max_gain_minute'],

            'min_price': float(followup['min_price_seen']),
            'max_drawdown_percent': round(followup['max_drawdown_percent'], 2),
            'max_drawdown_reached_at_minute': followup['max_drawdown_minute'],

            # Final statistics
            'price_at_10min': float(final_price),
            'final_gain_percent': round(final_gain, 2),

            # Stop loss
            'reached_stop_loss': followup['reached_stop_loss'],
            'time_to_stop_loss_minutes': followup['stop_loss_minute'],
            'price_at_stop_loss': (
                float(followup['stop_loss_price'])
                if followup['stop_loss_price'] is not None else None
            ),

            # Profitability snapshots
            **profitable_at,

            # Target achievements
            **targets_achieved,

            # Tracking metadata
            'intervals_recorded': followup['intervals_recorded'],
            'intervals_recorded_count': len(followup['intervals_recorded']),
            'tracking_completed': (
                len(followup['intervals_recorded']) == len(self.OUTCOME_TRACKING_INTERVALS_SECONDS)
            )
        }

        return summary

    def _finalize_outcome_tracking(self, key: str) -> None:
        """
        Finalize outcome tracking and save results.

        Called when:
        - All intervals have been recorded (10 minutes elapsed)
        - Market closes during tracking period
        - Tracking period end time is reached

        Calculates final summary statistics and updates the alert JSON file.

        Args:
            key: Followup key to finalize
        """
        if key not in self.active_followups:
            self.logger.warning(f"⚠️  Cannot finalize {key}: not found in active followups")
            return

        followup = self.active_followups[key]

        # Build outcome summary statistics
        summary = self._build_outcome_summary(followup)

        # Update alert JSON file with outcome data
        self._update_alert_with_outcomes(
            alert_filename=followup['alert_filename'],
            followup=followup,
            summary=summary
        )

        # Clean up tracking data
        del self.active_followups[key]

        if key in self.followup_volume_tracking:
            del self.followup_volume_tracking[key]

        if key in self.followup_trades_tracking:
            del self.followup_trades_tracking[key]

        # Log completion
        completion_status = "✅ COMPLETE" if summary['tracking_completed'] else "⏸️  PARTIAL"
        self.logger.info(
            f"{completion_status} Outcome tracking for {followup['symbol']}: "
            f"max gain {summary['max_gain_percent']:+.2f}% @ T+{summary['max_gain_reached_at_minute']}min, "
            f"final {summary['final_gain_percent']:+.2f}% @ T+10min "
            f"({summary['intervals_recorded_count']}/{len(self.OUTCOME_TRACKING_INTERVALS_SECONDS)} intervals)"
        )

    def _update_alert_with_outcomes(self, alert_filename: str, followup: Dict[str, Any],
                                     summary: Dict[str, Any]) -> None:
        """
        Update the original alert JSON file with outcome tracking data.

        Reads the existing alert JSON, adds the outcome_tracking section,
        and writes it back to disk.

        Args:
            alert_filename: Name of alert JSON file (e.g., "alert_AAPL_2025-12-12_152045.json")
            followup: Complete followup data dictionary
            summary: Summary statistics dictionary
        """
        try:
            filepath = self.squeeze_alerts_sent_dir / alert_filename

            # Check if file exists
            if not filepath.exists():
                self.logger.error(
                    f"❌ Cannot update outcomes: alert file not found: {alert_filename}"
                )
                return

            # Read existing alert data
            with open(filepath, 'r') as f:
                alert_data = json.load(f)

            # Add outcome tracking section
            alert_data['outcome_tracking'] = {
                # Configuration
                'enabled': True,
                'tracking_start': followup['start_time'].isoformat(),
                'tracking_end': followup['end_time'].isoformat(),
                'squeeze_entry_price': float(followup['squeeze_price']),
                'duration_minutes': self.OUTCOME_TRACKING_DURATION_MINUTES,
                'interval_seconds': self.OUTCOME_TRACKING_INTERVALS_SECONDS,

                # Interval snapshots (keyed by seconds: 10, 20, 30, 60, 120, etc.)
                'intervals': followup['interval_data'],

                # Summary statistics
                'summary': summary
            }

            # Write updated data back to file
            with open(filepath, 'w') as f:
                json.dump(alert_data, f, indent=2)

            self.logger.debug(f"📝 Updated {alert_filename} with outcome data")

        except Exception as e:
            self.logger.error(f"❌ Error updating alert with outcomes: {e}")
            import traceback
            traceback.print_exc()

    def _get_price_status(self, current_price: float, reference_high: float) -> tuple:
        """
        Determine the status color and icon based on how close current price is to a reference high.

        Args:
            current_price: Current trade price
            reference_high: Reference high price (HOD or premarket high)

        Returns:
            Tuple of (icon, color_name, percent_off)
            - Green if within 1.5% of high
            - Red if 5% or more below high
            - Yellow otherwise
        """
        if reference_high is None or reference_high == 0:
            return ("⚪", "white", None)

        percent_off = ((reference_high - current_price) / reference_high) * 100 * -1

        if percent_off >= -1.5:
            return ("🟢", "green", percent_off)
        elif percent_off <= -5.0:
            return ("🔴", "red", percent_off)
        else:
            return ("🟡", "yellow", percent_off)

    def _report_squeeze(self, symbol: str, first_price: float, last_price: float,
                       percent_change: float, timestamp: datetime, size: int):
        """
        Report a detected squeeze to stdout and Telegram.

        Only reports squeezes for symbols with ≥10% day gain (yellow or green icons).

        Args:
            symbol: Stock symbol
            first_price: First price in 10-second window (oldest trade)
            last_price: Last price in 10-second window (newest trade)
            percent_change: Percent increase from first to last
            timestamp: Current timestamp
            size: Current trade size
        """
        # Check gain data first - only report squeezes for yellow/green gains (≥10%)
        gain_data = self.gains_per_symbol.get(symbol)
        gain_data_error = None

        if gain_data:
            gain_percent = gain_data.get('gain_percent')
            if gain_percent is not None and gain_percent < 10.0:
                # Skip reporting this squeeze - gain is below 10% (red icon)
                if self.verbose:
                    self.logger.debug(f"⏭️  Skipping squeeze for {symbol}: gain {gain_percent:.2f}% < 10%")
                return
        else:
            # No gain data available - log error but still send alert
            gain_data_error = f"No gain data available for {symbol}"
            self.logger.error(f"⚠️  {gain_data_error} during squeeze detection")

        self.squeeze_count += 1
        self.squeezes_by_symbol[symbol] = self.squeezes_by_symbol.get(symbol, 0) + 1

        # Note: Counts are persisted via alert files themselves, loaded on restart
        # No need for separate file writes - alert files are the source of truth

        # Get HOD and premarket high data
        day_highs = self.market_data.get_day_highs(symbol, timestamp)
        premarket_high = day_highs.get('premarket_high')
        regular_hod = day_highs.get('regular_hours_hod')

        # Get latest candlestick with VWAP data
        candlestick = self.market_data.get_latest_candlestick(symbol)
        vwap = candlestick.get('vwap')

        # Verify VWAP is in the data and log if missing
        if vwap is None:
            if 'error' in candlestick:
                self.logger.warning(f"⚠️  Could not get VWAP for {symbol}: {candlestick['error']}")
            else:
                self.logger.warning(f"⚠️  VWAP not available in candlestick data for {symbol}")

        # Get latest quote data for bid/ask spread
        quote_data = self.market_data.get_latest_quote_data(symbol)
        bid_price = quote_data.get('bid_price')
        ask_price = quote_data.get('ask_price')

        # Calculate spread and spread percentage
        spread = None
        spread_percent = None
        if bid_price is not None and ask_price is not None and bid_price > 0 and ask_price > 0:
            spread = ask_price - bid_price
            if last_price > 0:
                spread_percent = (spread / last_price) * 100

        # Determine status for premarket high
        pm_icon, pm_color, pm_percent_off = self._get_price_status(last_price, premarket_high)

        # Determine status for regular hours HOD
        hod_icon, hod_color, hod_percent_off = self._get_price_status(last_price, regular_hod)

        # Determine VWAP status: green if price > VWAP, red otherwise
        if vwap is not None:
            if last_price > vwap:
                vwap_icon = "🟢"
                vwap_color = "green"
            else:
                vwap_icon = "🔴"
                vwap_color = "red"
        else:
            vwap_icon = "⚪"
            vwap_color = "white"

        # Get gain data for this symbol
        gain_data = self.gains_per_symbol.get(symbol)
        gain_percent = None
        gain_icon = "⚪"
        gain_color = "white"

        if gain_data:
            gain_percent = gain_data.get('gain_percent')
            if gain_percent is not None:
                if gain_percent >= 30.0:
                    gain_icon = "🟢"
                    gain_color = "green"
                elif gain_percent >= 10.0:
                    gain_icon = "🟡"
                    gain_color = "yellow"
                else:
                    # Anything below 10% is red
                    gain_icon = "🔴"
                    gain_color = "red"

        # Get volume surge data for this symbol
        volume_surge_data = self.volume_surge_data.get(symbol, {})
        volume_surge_ratio = volume_surge_data.get('volume_surge_ratio')

        # Get fundamental data for this symbol
        fundamental_data = self.fundamental_data.get(symbol, {})
        float_shares = fundamental_data.get('float_shares')

        # Calculate float rotation using hourly volume data (04:00 ET to now)
        # Float Rotation = Total Volume (hourly bars from 04:00 ET) / Float Shares
        total_volume_since_0400 = self.hourly_volume_data.get(symbol)
        float_rotation = None
        float_rotation_percent = None

        if float_shares and float_shares > 0 and total_volume_since_0400 is not None:
            # Calculate float rotation as a ratio
            float_rotation = total_volume_since_0400 / float_shares
            float_rotation_percent = float_rotation * 100

            if self.verbose:
                self.logger.debug(
                    f"📊 {symbol}: Volume since 04:00 ET: {total_volume_since_0400:,} | "
                    f"Float: {float_shares:,} | "
                    f"Float Rotation: {float_rotation:.4f}x ({float_rotation_percent:.2f}%)")

        # Calculate EMA 9 and EMA 21 from 1-minute candlesticks
        ema_data = self._calculate_ema_values(symbol, timestamp)
        ema_9 = ema_data.get('ema_9')
        ema_21 = ema_data.get('ema_21')
        ema_error = ema_data.get('error')

        if ema_error and self.verbose:
            self.logger.debug(f"⚠️  EMA calculation for {symbol}: {ema_error}")

        # Calculate MACD from 1-minute candlesticks
        macd_data = self._calculate_macd_values(symbol, timestamp)
        macd = macd_data.get('macd')
        macd_signal = macd_data.get('signal')
        macd_histogram = macd_data.get('histogram')
        macd_error = macd_data.get('error')

        if macd_error and self.verbose:
            self.logger.debug(f"⚠️  MACD calculation for {symbol}: {macd_error}")

        # Determine price icon based on last_price
        if 2 <= last_price <= 20:
            price_icon = "🟢"  # Green: $2-$20 (sweet spot)
        elif (20 < last_price < 30) or (1 <= last_price < 2):
            price_icon = "🟡"  # Yellow: $20-$30 or $1-$2
        else:
            price_icon = "🔴"  # Red: <$1 or >=$30

        # Print to stdout (not logger, so it's always visible)
        print(f"\n{'='*70}")
        print(f"🚀 SQUEEZE DETECTED - {symbol}")
        print(f"{'='*70}")
        print(f"Time:           {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Price Range:    ${first_price:.2f} → {price_icon} ${last_price:.2f}")
        print(f"Change:         +{percent_change:.2f}% in 10 seconds")
        print(f"Window Trades:  {len(self.price_history[symbol])} trades")

        # Display gain from previous close
        if gain_percent is not None:
            print(f"Day Gain:       {gain_icon} {gain_percent:+.2f}%")
        else:
            print(f"Day Gain:       {gain_icon} N/A")

        # Display VWAP status
        if vwap:
            print(f"VWAP:           {vwap_icon} ${vwap:.2f}")
        else:
            print(f"VWAP:           {vwap_icon} N/A")

        # Display premarket high status
        if premarket_high:
            print(f"Premarket High: {pm_icon} ${premarket_high:.2f} ({pm_percent_off:.1f}% off)")
        else:
            print(f"Premarket High: {pm_icon} N/A")

        # Display regular hours HOD status
        if regular_hod:
            print(f"Regular HOD:    {hod_icon} ${regular_hod:.2f} ({hod_percent_off:.1f}% off)")
        else:
            print(f"Regular HOD:    {hod_icon} N/A")

        # Display volume surge ratio
        if volume_surge_ratio is not None:
            # Green if > 5x, red otherwise
            surge_icon = "🟢" if volume_surge_ratio > 5 else "🔴"
            print(f"Surge Ratio:    {surge_icon} {volume_surge_ratio:.2f}x")
        else:
            print(f"Surge Ratio:    ⚪ N/A")

        # Display float shares
        if float_shares is not None:
            if float_shares >= 1_000_000_000:
                float_str = f"{float_shares / 1_000_000_000:.2f}B"
            elif float_shares >= 1_000_000:
                float_str = f"{float_shares / 1_000_000:.2f}M"
            else:
                float_str = f"{float_shares:,.0f}"
            # Green if < 20M, yellow if < 50M, red otherwise
            if float_shares < 20_000_000:
                float_icon = "🟢"
            elif float_shares < 50_000_000:
                float_icon = "🟡"
            else:
                float_icon = "🔴"
            print(f"Float Shares:   {float_icon} {float_str}")
        else:
            print(f"Float Shares:   ⚪ N/A")

        # Display float rotation
        if float_rotation is not None:
            # Determine emoji based on float rotation value
            if float_rotation > 1:
                rotation_emoji = "🟢"  # Green if > 1
            elif float_rotation < 0.8:
                rotation_emoji = "🔴"  # Red if < 0.8
            else:
                rotation_emoji = "🟡"  # Yellow otherwise
            print(f"Float Rotation: {rotation_emoji} {float_rotation:.2f}x")
        else:
            print(f"Float Rotation: ⚪ N/A")

        # Display spread and spread percentage
        if spread is not None and spread_percent is not None:
            # Red if spread < 0.5%, yellow if < 1.5%, green otherwise
            if spread_percent < 0.5:
                spread_icon = "🔴"
            elif spread_percent < 1.5:
                spread_icon = "🟡"
            else:
                spread_icon = "🟢"
            print(f"Spread:         {spread_icon} ${spread:.4f} ({spread_percent:.2f}%)")
        else:
            print(f"Spread:         ⚪ N/A")

        # Display EMA 9 and EMA 21
        if ema_9 is not None and ema_21 is not None:
            # Determine EMA icon based on EMA 9 vs EMA 21
            if ema_9 > ema_21:
                ema_icon = "🟢"  # Green if EMA 9 > EMA 21 (bullish)
            else:
                ema_icon = "🔴"  # Red if EMA 9 < EMA 21 (bearish)
            print(f"EMA 9/21:       {ema_icon} ${ema_9:.2f} / ${ema_21:.2f}")
        else:
            print(f"EMA 9/21:       ⚪ N/A")

        # Display MACD
        if macd is not None and macd_signal is not None and macd_histogram is not None:
            # Determine MACD icon based on histogram (MACD - Signal)
            if macd_histogram > 0:
                macd_icon = "🟢"  # Green if histogram > 0 (bullish)
            else:
                macd_icon = "🔴"  # Red if histogram <= 0 (bearish)
            print(f"MACD:           {macd_icon} {macd:.4f} / {macd_signal:.4f} / {macd_histogram:.4f}")
        else:
            print(f"MACD:           ⚪ N/A")

        print(f"Total Squeezes: {self.squeeze_count} ({self.squeezes_by_symbol[symbol]} for {symbol})")
        print(f"{'='*70}\n")

        # Also log it
        self.logger.info(f"🚀 SQUEEZE: {symbol} +{percent_change:.2f}% (${first_price:.2f} → ${last_price:.2f})")

        # Calculate Phase 1 enhancement metrics for data analysis
        phase1_metrics = self._calculate_phase1_metrics(symbol, timestamp, last_price)

        # Send Telegram alert to users with squeeze_alerts=true
        sent_users = self._send_telegram_alert(
            symbol, first_price, last_price, percent_change, timestamp, size,
            premarket_high, pm_icon, pm_percent_off,
            regular_hod, hod_icon, hod_percent_off,
            vwap, vwap_icon,
            gain_percent, gain_icon, gain_data_error,
            volume_surge_ratio, float_shares, float_rotation,
            spread, spread_percent,
            ema_9, ema_21,
            macd, macd_signal, macd_histogram
        )

        # Save squeeze alert to JSON file for scanner display (includes Phase 1 metrics)
        filename = self._save_squeeze_alert_sent(
            symbol, first_price, last_price, percent_change, timestamp, size, sent_users,
            premarket_high, pm_icon, pm_color, pm_percent_off,
            regular_hod, hod_icon, hod_color, hod_percent_off,
            vwap, vwap_icon, vwap_color,
            gain_percent, gain_icon, gain_color, gain_data_error,
            volume_surge_ratio, float_shares, float_rotation, float_rotation_percent,
            spread, spread_percent,
            ema_9, ema_21,
            macd, macd_signal, macd_histogram,
            phase1_metrics
        )

        # Start outcome tracking for this squeeze (if enabled and alert was saved successfully)
        if filename:
            self._start_outcome_tracking(
                symbol=symbol,
                squeeze_timestamp=timestamp,
                squeeze_price=last_price,
                alert_filename=filename
            )

    def _send_telegram_alert(self, symbol: str, first_price: float, last_price: float,
                            percent_change: float, timestamp: datetime, size: int,
                            premarket_high: float, pm_icon: str, pm_percent_off: float,
                            regular_hod: float, hod_icon: str, hod_percent_off: float,
                            vwap: float, vwap_icon: str,
                            gain_percent: float, gain_icon: str, gain_data_error: str,
                            volume_surge_ratio: float, float_shares: float, float_rotation: float,
                            spread: float, spread_percent: float,
                            ema_9: float, ema_21: float,
                            macd: float, macd_signal: float, macd_histogram: float):
        """
        Send Telegram alert to users with squeeze_alerts=true.

        Args:
            symbol: Stock symbol
            first_price: First price in 10-second window (oldest trade)
            last_price: Last price in 10-second window (newest trade)
            percent_change: Percent increase from first to last
            timestamp: Current timestamp
            size: Current trade size
            premarket_high: Premarket high price
            pm_icon: Premarket high status icon
            pm_percent_off: Percent off premarket high
            regular_hod: Regular hours HOD
            hod_icon: Regular hours HOD status icon
            hod_percent_off: Percent off regular hours HOD
            vwap: VWAP price
            vwap_icon: VWAP status icon
            gain_percent: Day gain percentage from previous close
            gain_icon: Day gain status icon
            gain_data_error: Error message if gain data is unavailable
            volume_surge_ratio: Volume surge ratio (multiplier)
            float_shares: Float shares outstanding
            float_rotation: Float rotation (total volume / float shares)
            spread: Bid-ask spread (ask_price - bid_price)
            spread_percent: Spread as percentage of latest price
            ema_9: EMA 9 value from 1-minute candlesticks
            ema_21: EMA 21 value from 1-minute candlesticks
            macd: MACD line value from 1-minute candlesticks
            macd_signal: MACD signal line value from 1-minute candlesticks
            macd_histogram: MACD histogram value from 1-minute candlesticks
        """
        try:
            # Get users with squeeze_alerts=true
            squeeze_users = self.user_manager.get_squeeze_alert_users()

            if not squeeze_users:
                self.logger.debug("No users with squeeze_alerts=true found")
                return

            # Determine price icon based on last_price
            if 2 <= last_price <= 20:
                price_icon = "🟢"  # Green: $2-$20 (sweet spot)
            elif (20 < last_price < 30) or (1 <= last_price < 2):
                price_icon = "🟡"  # Yellow: $20-$30 or $1-$2
            else:
                price_icon = "🔴"  # Red: <$1 or >=$30

            # Build VWAP line
            if vwap:
                vwap_line = f"📊 VWAP: {vwap_icon} ${vwap:.2f}\n"
            else:
                vwap_line = f"📊 VWAP: {vwap_icon} N/A\n"

            # Build premarket high line
            if premarket_high:
                pm_line = f"📊 PM High: {pm_icon} ${premarket_high:.2f} ({pm_percent_off:.1f}% off)\n"
            else:
                pm_line = f"📊 PM High: {pm_icon} N/A\n"

            # Build regular hours HOD line
            if regular_hod:
                hod_line = f"📊 HOD: {hod_icon} ${regular_hod:.2f} ({hod_percent_off:.1f}% off)\n"
            else:
                hod_line = f"📊 HOD: {hod_icon} N/A\n"

            # Build day gain line
            if gain_percent is not None:
                gain_line = f"📊 Day Gain: {gain_icon} <b>{gain_percent:+.2f}%</b>\n"
            else:
                gain_line = f"📊 Day Gain: {gain_icon} N/A\n"

            # Add error line if gain data is missing
            error_line = ""
            if gain_data_error:
                error_line = f"⚠️ <i>{gain_data_error}</i>\n"

            # Build volume surge ratio line
            if volume_surge_ratio is not None:
                # Green if > 5x, red otherwise
                surge_icon = "🟢" if volume_surge_ratio > 5 else "🔴"
                surge_line = f"📊 Surge Ratio: {surge_icon} {volume_surge_ratio:.2f}x\n"
            else:
                surge_line = f"📊 Surge Ratio: ⚪ N/A\n"

            # Build float shares line
            if float_shares is not None:
                if float_shares >= 1_000_000_000:
                    float_str = f"{float_shares / 1_000_000_000:.2f}B"
                elif float_shares >= 1_000_000:
                    float_str = f"{float_shares / 1_000_000:.2f}M"
                else:
                    float_str = f"{float_shares:,.0f}"
                # Green if < 20M, yellow if < 50M, red otherwise
                if float_shares < 20_000_000:
                    float_icon = "🟢"
                elif float_shares < 50_000_000:
                    float_icon = "🟡"
                else:
                    float_icon = "🔴"
                float_shares_line = f"📊 Float Shares: {float_icon} {float_str}\n"
            else:
                float_shares_line = f"📊 Float Shares: ⚪ N/A\n"

            # Build float rotation line
            if float_rotation is not None:
                # Determine emoji based on float rotation value
                if float_rotation > 1:
                    rotation_emoji = "🟢"  # Green if > 1
                elif float_rotation < 0.8:
                    rotation_emoji = "🔴"  # Red if < 0.8
                else:
                    rotation_emoji = "🟡"  # Yellow otherwise
                float_rotation_line = f"📊 Float Rotation: {rotation_emoji} {float_rotation:.2f}x\n"
            else:
                float_rotation_line = f"📊 Float Rotation: ⚪ N/A\n"

            # Build spread line
            if spread is not None and spread_percent is not None:
                # Red if spread < 0.5%, yellow if < 1.5%, green otherwise
                if spread_percent < 0.5:
                    spread_icon = "🔴"
                elif spread_percent < 1.5:
                    spread_icon = "🟡"
                else:
                    spread_icon = "🟢"
                spread_line = f"📊 Spread: {spread_icon} ${spread:.4f} ({spread_percent:.2f}%)\n"
            else:
                spread_line = f"📊 Spread: ⚪ N/A\n"

            # Build EMA line
            if ema_9 is not None and ema_21 is not None:
                # Determine EMA icon based on EMA 9 vs EMA 21
                if ema_9 > ema_21:
                    ema_icon = "🟢"  # Green if EMA 9 > EMA 21 (bullish)
                else:
                    ema_icon = "🔴"  # Red if EMA 9 < EMA 21 (bearish)
                ema_line = f"📊 EMA 9/21: {ema_icon} ${ema_9:.2f} / ${ema_21:.2f}\n"
            else:
                ema_line = f"📊 EMA 9/21: ⚪ N/A\n"

            # Build MACD line
            if macd is not None and macd_signal is not None and macd_histogram is not None:
                # Determine MACD icon based on histogram
                if macd_histogram > 0:
                    macd_icon = "🟢"  # Green if histogram > 0 (bullish)
                else:
                    macd_icon = "🔴"  # Red if histogram <= 0 (bearish)
                macd_line = f"📊 MACD: {macd_icon} {macd:.4f} / {macd_signal:.4f} / {macd_histogram:.4f}\n"
            else:
                macd_line = f"📊 MACD: ⚪ N/A\n"

            # Format Telegram message
            message = (
                f"🚀 <b>SQUEEZE ALERT - {symbol}</b>\n\n"
                f"⏰ Time: {timestamp.strftime('%H:%M:%S ET')}\n"
                f"📈 Price: ${first_price:.2f} → {price_icon} ${last_price:.2f}\n"
                f"📊 Change: <b>+{percent_change:.2f}%</b> in 10 seconds\n"
                f"{gain_line}"
                f"{error_line}"
                f"{vwap_line}"
                f"{pm_line}"
                f"{hod_line}"
                f"{ema_line}"
                f"{macd_line}"
                f"{surge_line}"
                f"{float_shares_line}"
                f"{float_rotation_line}"
                f"{spread_line}"
                f"📉 Trades: {len(self.price_history[symbol])} in window\n\n"
                f"#Squeeze #{symbol}"
            )

            # Send to all squeeze alert users
            sent_count = 0
            for user in squeeze_users:
                username = user.get('username', 'Unknown')
                result = self.telegram_poster.send_message_to_user(
                    message, username, urgent=True)  # Use urgent=True for squeeze alerts

                if result['success']:
                    sent_count += 1
                    self.logger.debug(f"Sent squeeze alert to {username}")
                else:
                    self.logger.warning(f"Failed to send squeeze alert to {username}: {result.get('error')}")

            if sent_count > 0:
                self.logger.info(f"✅ Sent squeeze alert to {sent_count} user(s)")

            # Return list of usernames who received the alert
            return [user.get('username', 'Unknown') for user in squeeze_users]

        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _save_squeeze_alert_sent(self, symbol: str, first_price: float, last_price: float,
                                  percent_change: float, timestamp: datetime, size: int,
                                  sent_to_users: List[str],
                                  premarket_high: float, pm_icon: str, pm_color: str, pm_percent_off: float,
                                  regular_hod: float, hod_icon: str, hod_color: str, hod_percent_off: float,
                                  vwap: float, vwap_icon: str, vwap_color: str,
                                  gain_percent: float, gain_icon: str, gain_color: str, gain_data_error: str,
                                  volume_surge_ratio: float, float_shares: float, float_rotation: float,
                                  float_rotation_percent: float,
                                  spread: float, spread_percent: float,
                                  ema_9: float, ema_21: float,
                                  macd: float, macd_signal: float, macd_histogram: float,
                                  phase1_metrics: Dict[str, Any]) -> str:
        """
        Save sent squeeze alert to JSON file for scanner display.

        Args:
            symbol: Stock symbol
            first_price: First price in 10-second window (oldest trade)
            last_price: Last price in 10-second window (newest trade)
            percent_change: Percent increase from first to last
            timestamp: Current timestamp
            size: Current trade size
            sent_to_users: List of usernames who received the alert
            premarket_high: Premarket high price
            pm_icon: Premarket high status icon
            pm_color: Premarket high status color
            pm_percent_off: Percent off premarket high
            regular_hod: Regular hours HOD
            hod_icon: Regular hours HOD status icon
            hod_color: Regular hours HOD status color
            hod_percent_off: Percent off regular hours HOD
            vwap: VWAP price
            vwap_icon: VWAP status icon
            vwap_color: VWAP status color
            gain_percent: Day gain percentage from previous close
            gain_icon: Day gain status icon
            gain_color: Day gain status color
            gain_data_error: Error message if gain data is unavailable
            volume_surge_ratio: Volume surge ratio (multiplier)
            float_shares: Float shares outstanding
            float_rotation: Float rotation (total volume / float shares)
            float_rotation_percent: Float rotation as percentage
            spread: Bid-ask spread (ask_price - bid_price)
            spread_percent: Spread as percentage of latest price
            ema_9: EMA 9 value from 1-minute candlesticks
            ema_21: EMA 21 value from 1-minute candlesticks
            macd: MACD line value from 1-minute candlesticks
            macd_signal: MACD signal line value from 1-minute candlesticks
            macd_histogram: MACD histogram value from 1-minute candlesticks
            phase1_metrics: Dictionary containing Phase 1 enhancement metrics
        """
        try:
            # Create filename with timestamp
            filename = f"alert_{symbol}_{timestamp.strftime('%Y-%m-%d_%H%M%S')}.json"
            filepath = self.squeeze_alerts_sent_dir / filename

            # Ensure timestamp is timezone-aware in ET
            if timestamp.tzinfo is None:
                timestamp = self.et_tz.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(self.et_tz)

            # Build alert data
            alert_json = {
                'symbol': symbol,
                'timestamp': timestamp.isoformat(),
                'first_price': float(first_price),
                'last_price': float(last_price),
                'percent_change': float(percent_change),
                'size': int(size),
                'window_trades': len(self.price_history.get(symbol, [])),
                'squeeze_threshold': float(self.squeeze_percent),
                'sent_to_users': sent_to_users,
                'sent_count': len(sent_to_users),
                'day_gain': float(gain_percent) if gain_percent is not None else None,
                'day_gain_status': {
                    'icon': gain_icon,
                    'color': gain_color
                },
                'vwap': float(vwap) if vwap else None,
                'vwap_status': {
                    'icon': vwap_icon,
                    'color': vwap_color
                },
                'premarket_high': float(premarket_high) if premarket_high else None,
                'premarket_high_status': {
                    'icon': pm_icon,
                    'color': pm_color,
                    'percent_off': float(pm_percent_off) if pm_percent_off is not None else None
                },
                'regular_hours_hod': float(regular_hod) if regular_hod else None,
                'regular_hours_hod_status': {
                    'icon': hod_icon,
                    'color': hod_color,
                    'percent_off': float(hod_percent_off) if hod_percent_off is not None else None
                },
                'volume_surge_ratio': float(volume_surge_ratio) if volume_surge_ratio is not None else None,
                'float_shares': float(float_shares) if float_shares is not None else None,
                'float_rotation': float(float_rotation) if float_rotation is not None else None,
                'float_rotation_percent': float(float_rotation_percent) if float_rotation_percent is not None else None,
                'spread': float(spread) if spread is not None else None,
                'spread_percent': float(spread_percent) if spread_percent is not None else None,
                'ema_9': float(ema_9) if ema_9 is not None else None,
                'ema_21': float(ema_21) if ema_21 is not None else None,
                'macd': float(macd) if macd is not None else None,
                'macd_signal': float(macd_signal) if macd_signal is not None else None,
                'macd_histogram': float(macd_histogram) if macd_histogram is not None else None
            }

            # Add error field if present
            if gain_data_error:
                alert_json['error'] = gain_data_error

            # Add Phase 1 enhancement metrics for data analysis
            if phase1_metrics:
                alert_json['phase1_analysis'] = phase1_metrics

            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(alert_json, f, indent=2)

            self.logger.debug(f"📝 Saved squeeze alert for {symbol} to {filename}")

            return filename

        except Exception as e:
            self.logger.error(f"❌ Error saving squeeze alert: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _print_statistics(self):
        """Print monitoring statistics."""
        if not self.start_time:
            return

        runtime = (datetime.now() - self.start_time).total_seconds()

        self.logger.info("\n" + "="*70)
        self.logger.info("📈 MONITORING STATISTICS")
        self.logger.info("="*70)
        self.logger.info(f"Runtime: {runtime:.1f} seconds")
        self.logger.info(f"Total trades received: {self.trade_count:,}")
        self.logger.info(f"Active subscriptions: {len(self.active_subscriptions)}")
        self.logger.info(f"Squeeze threshold: {self.squeeze_percent}%")
        self.logger.info(f"Total squeezes detected: {self.squeeze_count}")

        if self.trade_count > 0:
            self.logger.info(f"Trades per second: {self.trade_count/runtime:.2f}")

            # Top 10 most active symbols
            if self.trades_by_symbol:
                sorted_symbols = sorted(
                    self.trades_by_symbol.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.logger.info("\nTop 10 Most Active Symbols:")
                for i, (symbol, count) in enumerate(sorted_symbols[:10], 1):
                    self.logger.info(f"   {i:2d}. {symbol:6s} - {count:5,d} trades")

            # Symbols with squeezes
            if self.squeezes_by_symbol:
                self.logger.info("\nSymbols with Squeezes:")
                sorted_squeezes = sorted(
                    self.squeezes_by_symbol.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for i, (symbol, count) in enumerate(sorted_squeezes, 1):
                    self.logger.info(f"   {i:2d}. {symbol:6s} - {count} squeeze(s)")

        self.logger.info("="*70)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Squeeze Alerts System - Real-time WebSocket Monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor existing symbols from other clients (default behavior)
  python3 squeeze_alerts.py

  # Run 10-second test on existing symbols
  python3 squeeze_alerts.py --test 10

  # Monitor specific symbols (explicit)
  python3 squeeze_alerts.py --symbols AAPL,TSLA,MSFT,NVDA

  # Monitor symbols from CSV file
  python3 squeeze_alerts.py --symbols-file data_master/master.csv

  # Limit to first 50 symbols with verbose output
  python3 squeeze_alerts.py --symbols-file data_master/master.csv --max-symbols 50 --verbose

  # Test mode with specific symbols
  python3 squeeze_alerts.py --symbols AAPL,TSLA,NVDA --test 10 --verbose

  # Explicit use-existing flag (same as default if no symbols specified)
  python3 squeeze_alerts.py --use-existing
        """
    )

    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to monitor (e.g., AAPL,TSLA,MSFT)'
    )

    parser.add_argument(
        '--symbols-file',
        type=str,
        help='Path to CSV file containing symbols (default: data_master/master.csv)'
    )

    parser.add_argument(
        '--max-symbols',
        type=int,
        help='Maximum number of symbols to monitor (useful for testing)'
    )

    parser.add_argument(
        '--websocket-url',
        type=str,
        default='ws://localhost:8766',
        help='WebSocket server URL (default: ws://localhost:8766)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--test',
        type=int,
        metavar='SECONDS',
        help='Test mode: run for N seconds, then stop and print summary report'
    )

    parser.add_argument(
        '--use-existing',
        action='store_true',
        help='Auto-subscribe to existing symbols from other clients (ignores --symbols and --symbols-file)'
    )


    args = parser.parse_args()

    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Auto-enable use_existing if neither --symbols nor --symbols-file specified
    use_existing = args.use_existing
    if not use_existing and not args.symbols and not args.symbols_file:
        use_existing = True

    # Create and run monitor
    monitor = SqueezeAlertsMonitor(
        symbols=symbols,
        symbols_file=args.symbols_file,
        max_symbols=args.max_symbols,
        verbose=args.verbose,
        websocket_url=args.websocket_url,
        test_duration=args.test,
        use_existing=use_existing
    )

    await monitor.connect_and_monitor()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
