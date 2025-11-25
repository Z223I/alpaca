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
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Deque
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


class SqueezeAlertsMonitor:
    """Real-time trade monitoring system for squeeze detection."""

    def __init__(
        self,
        symbols: List[str] = None,
        symbols_file: str = None,
        max_symbols: int = None,
        verbose: bool = False,
        websocket_url: str = "ws://localhost:8766",
        test_duration: int = None,
        use_existing: bool = False,
        squeeze_percent: float = 1.5
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
            squeeze_percent: Percent price increase in 10 seconds to trigger squeeze alert (default: 1.5%)
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
        # Each symbol stores: deque of (timestamp, price) tuples
        self.price_history: Dict[str, Deque[tuple]] = {}
        self.squeeze_count = 0
        self.squeezes_by_symbol: Dict[str, int] = {}

        # Telegram integration
        self.telegram_poster = TelegramPoster()
        self.user_manager = UserManager()

        # Setup directories for saving squeeze alerts
        self.et_tz = pytz.timezone('US/Eastern')
        self.today = datetime.now(self.et_tz).strftime('%Y-%m-%d')
        self.squeeze_alerts_sent_dir = Path(project_root) / "historical_data" / self.today / "squeeze_alerts_sent"
        self.squeeze_alerts_sent_dir.mkdir(parents=True, exist_ok=True)

        # Manual symbols file monitoring
        self.manual_symbols_file = Path(project_root) / "data" / "manual_symbols.json"
        self.last_subscription_time = None

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

    def _get_all_symbols_to_monitor(self) -> List[str]:
        """
        Get all symbols to monitor from all sources (manual symbols + existing list).

        Returns:
            Combined list of unique symbols
        """
        all_symbols = set()

        # Add existing symbols (from initial load or use_existing)
        all_symbols.update(self.symbols_to_monitor)

        # Add manual symbols
        manual_symbols = self._load_manual_symbols()
        all_symbols.update(manual_symbols)

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

                # Start monitoring loop
                await self._monitor_trades(ws)

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
            return

        # Determine which symbols need subscription
        if refresh:
            # Only subscribe to symbols not already subscribed
            symbols_to_subscribe = [s for s in all_symbols if s not in self.active_subscriptions]
            if not symbols_to_subscribe:
                self.logger.debug("🔄 All symbols already subscribed")
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

        Args:
            new_date: New date string in YYYY-MM-DD format
        """
        try:
            self.logger.info(f"🔄 Updating directory paths for new date: {new_date}")

            # Update the date
            self.today = new_date

            # Update directory path
            self.squeeze_alerts_sent_dir = Path(project_root) / "historical_data" / self.today / "squeeze_alerts_sent"

            # Create new directory
            self.squeeze_alerts_sent_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"✅ Directory paths updated successfully for {new_date}")
            self.logger.info(f"📁 Squeeze alerts dir: {self.squeeze_alerts_sent_dir}")

        except Exception as e:
            self.logger.error(f"❌ Error updating directory paths: {e}")

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

        # Parse timestamp
        from dateutil import parser as date_parser
        timestamp = date_parser.parse(timestamp_str)

        # Log trade (suppress in test mode unless verbose)
        if not self.test_mode or self.verbose:
            self.logger.info(
                f"📊 {symbol:6s} @ ${price:8.2f} x {size:6,d} shares  "
                f"[{exchange}]  {timestamp_str}"
            )

        # Squeeze detection: Track prices in 10-second rolling window
        await self._detect_squeeze(symbol, timestamp, price, size)

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

        # Add current trade to history
        self.price_history[symbol].append((timestamp, price))

        # Remove trades older than 10 seconds
        from datetime import timedelta
        cutoff_time = timestamp - timedelta(seconds=10)

        while self.price_history[symbol] and self.price_history[symbol][0][0] < cutoff_time:
            self.price_history[symbol].popleft()

        # Need at least 2 prices to detect a squeeze
        if len(self.price_history[symbol]) < 2:
            return

        # Get lowest and highest prices in the window
        prices = [p for _, p in self.price_history[symbol]]
        low_price = min(prices)
        high_price = max(prices)

        # Calculate percent change from low to high
        if low_price > 0:
            percent_change = ((high_price - low_price) / low_price) * 100

            # Check if we have a squeeze
            if percent_change >= self.squeeze_percent:
                # Check if we already reported this squeeze recently
                # Only report once per symbol per 30-second window to avoid spam
                last_squeeze_time = getattr(self, '_last_squeeze_time', {})
                if symbol not in last_squeeze_time or (timestamp - last_squeeze_time[symbol]).total_seconds() > 30:
                    self._report_squeeze(symbol, low_price, high_price, percent_change, timestamp, size)

                    # Track last squeeze time
                    if not hasattr(self, '_last_squeeze_time'):
                        self._last_squeeze_time = {}
                    self._last_squeeze_time[symbol] = timestamp

    def _report_squeeze(self, symbol: str, low_price: float, high_price: float,
                       percent_change: float, timestamp: datetime, size: int):
        """
        Report a detected squeeze to stdout and Telegram.

        Args:
            symbol: Stock symbol
            low_price: Lowest price in 10-second window
            high_price: Highest price in 10-second window
            percent_change: Percent increase
            timestamp: Current timestamp
            size: Current trade size
        """
        self.squeeze_count += 1
        self.squeezes_by_symbol[symbol] = self.squeezes_by_symbol.get(symbol, 0) + 1

        # Print to stdout (not logger, so it's always visible)
        print(f"\n{'='*70}")
        print(f"🚀 SQUEEZE DETECTED - {symbol}")
        print(f"{'='*70}")
        print(f"Time:           {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Price Range:    ${low_price:.2f} → ${high_price:.2f}")
        print(f"Change:         +{percent_change:.2f}% in 10 seconds")
        print(f"Current Size:   {size:,} shares")
        print(f"Window Trades:  {len(self.price_history[symbol])} trades")
        print(f"Total Squeezes: {self.squeeze_count} ({self.squeezes_by_symbol[symbol]} for {symbol})")
        print(f"{'='*70}\n")

        # Also log it
        self.logger.info(f"🚀 SQUEEZE: {symbol} +{percent_change:.2f}% (${low_price:.2f} → ${high_price:.2f})")

        # Send Telegram alert to users with squeeze_alerts=true
        sent_users = self._send_telegram_alert(symbol, low_price, high_price, percent_change, timestamp, size)

        # Save squeeze alert to JSON file for scanner display
        self._save_squeeze_alert_sent(symbol, low_price, high_price, percent_change, timestamp, size, sent_users)

    def _send_telegram_alert(self, symbol: str, low_price: float, high_price: float,
                            percent_change: float, timestamp: datetime, size: int):
        """
        Send Telegram alert to users with squeeze_alerts=true.

        Args:
            symbol: Stock symbol
            low_price: Lowest price in 10-second window
            high_price: Highest price in 10-second window
            percent_change: Percent increase
            timestamp: Current timestamp
            size: Current trade size
        """
        try:
            # Get users with squeeze_alerts=true
            squeeze_users = self.user_manager.get_squeeze_alert_users()

            if not squeeze_users:
                self.logger.debug("No users with squeeze_alerts=true found")
                return

            # Format Telegram message
            message = (
                f"🚀 <b>SQUEEZE ALERT - {symbol}</b>\n\n"
                f"⏰ Time: {timestamp.strftime('%H:%M:%S ET')}\n"
                f"📈 Price: ${low_price:.2f} → ${high_price:.2f}\n"
                f"📊 Change: <b>+{percent_change:.2f}%</b> in 10 seconds\n"
                f"💰 Size: {size:,} shares\n"
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

    def _save_squeeze_alert_sent(self, symbol: str, low_price: float, high_price: float,
                                  percent_change: float, timestamp: datetime, size: int,
                                  sent_to_users: List[str]) -> None:
        """
        Save sent squeeze alert to JSON file for scanner display.

        Args:
            symbol: Stock symbol
            low_price: Lowest price in 10-second window
            high_price: Highest price in 10-second window
            percent_change: Percent increase
            timestamp: Current timestamp
            size: Current trade size
            sent_to_users: List of usernames who received the alert
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
                'low_price': float(low_price),
                'high_price': float(high_price),
                'percent_change': float(percent_change),
                'size': int(size),
                'window_trades': len(self.price_history.get(symbol, [])),
                'squeeze_threshold': float(self.squeeze_percent),
                'sent_to_users': sent_to_users,
                'sent_count': len(sent_to_users)
            }

            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(alert_json, f, indent=2)

            self.logger.debug(f"📝 Saved squeeze alert for {symbol} to {filename}")

        except Exception as e:
            self.logger.error(f"❌ Error saving squeeze alert: {e}")
            import traceback
            traceback.print_exc()

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
