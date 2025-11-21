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
from typing import List, Set, Dict
import websockets

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))  # cgi-bin/molecules/alpaca_molecules/
molecules_dir = os.path.dirname(script_dir)  # cgi-bin/molecules/
cgi_bin_dir = os.path.dirname(molecules_dir)  # cgi-bin/
project_root = os.path.dirname(cgi_bin_dir)  # project root
sys.path.insert(0, project_root)


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
        use_existing: bool = False
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
        """
        self.symbols_to_monitor: List[str] = []
        self.max_symbols = max_symbols
        self.websocket_url = websocket_url
        self.verbose = verbose
        self.test_duration = test_duration
        self.use_existing = use_existing

        # Setup logging
        self.logger = self._setup_logging(verbose)

        # Statistics tracking
        self.trade_count = 0
        self.trades_by_symbol: Dict[str, int] = {}
        self.active_subscriptions: Set[str] = set()
        self.start_time = None
        self.test_mode = test_duration is not None

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
                    self.logger.warning("⚠️  No existing subscriptions found, will use defaults")
                    self.symbols_to_monitor = ["AAPL", "TSLA", "MSFT"]
            else:
                self.logger.warning(f"⚠️  Unexpected response type: {data['type']}")

        except asyncio.TimeoutError:
            self.logger.error("❌ Timeout waiting for health response")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"❌ Error querying existing symbols: {e}")
            sys.exit(1)

    async def _subscribe_all_symbols(self, ws):
        """Subscribe to all symbols in the monitor list."""
        if not self.symbols_to_monitor:
            self.logger.warning("⚠️  No symbols to subscribe to")
            return

        self.logger.info(f"📤 Subscribing to {len(self.symbols_to_monitor)} symbols...")

        # Send subscription requests
        for symbol in self.symbols_to_monitor:
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
        max_messages = len(self.symbols_to_monitor) * 3  # Allow for 'connecting' + 'subscribed' + buffer

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
                    self.logger.info(f"   ✅ Subscribed to {symbol} ({confirmed}/{len(self.symbols_to_monitor)})")
                elif data['type'] == 'error':
                    self.logger.error(f"   ❌ Subscription error: {data.get('message')}")

                # Exit early if all subscriptions confirmed
                if confirmed == len(self.symbols_to_monitor):
                    break

            except asyncio.TimeoutError:
                timeout_count += 1
                if timeout_count > 5:  # Stop waiting after 5 consecutive timeouts
                    break

        self.logger.info(f"✅ Successfully subscribed to {confirmed}/{len(self.symbols_to_monitor)} symbols")
        if confirmed < len(self.symbols_to_monitor):
            self.logger.warning(f"⚠️  {len(self.symbols_to_monitor) - confirmed} symbols failed to subscribe")

    async def _monitor_trades(self, ws):
        """Monitor and log incoming trades (Phase 1: Eavesdropping)."""
        self.logger.info("="*70)
        if self.test_mode:
            self.logger.info(f"🧪 TEST MODE - Running for {self.test_duration} seconds")
        else:
            self.logger.info("🎧 EAVESDROPPING MODE - Monitoring live trades")
            self.logger.info("   Press Ctrl+C to stop")
        self.logger.info("="*70)

        # Track end time for test mode
        end_time = None
        if self.test_mode:
            end_time = asyncio.get_event_loop().time() + self.test_duration

        try:
            while True:
                # Check if test duration has elapsed
                if self.test_mode and asyncio.get_event_loop().time() >= end_time:
                    raise KeyboardInterrupt("Test duration completed")

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
        Handle incoming trade data (Phase 1: Just log).

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

        # Log trade (suppress in test mode unless verbose)
        if not self.test_mode or self.verbose:
            price = data['price']
            size = data['size']
            exchange = data.get('exchange', 'N/A')
            timestamp = data['timestamp']

            self.logger.info(
                f"📊 {symbol:6s} @ ${price:8.2f} x {size:6,d} shares  "
                f"[{exchange}]  {timestamp}"
            )

        # TODO Phase 2: Add squeeze detection logic here
        # - Track volume
        # - Detect price compression
        # - Monitor volatility changes
        # - Generate alerts on squeeze conditions

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
