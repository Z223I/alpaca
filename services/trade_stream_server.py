#!/usr/bin/env python3
"""
Real-time Trade Streaming WebSocket Server for Market Sentinel

This server maintains a persistent WebSocket connection to Alpaca's trade stream
and broadcasts real-time trades to connected web clients.

Architecture:
    Alpaca WebSocket → This Server → Browser Clients

Usage:
    python3 services/trade_stream_server.py

The server listens on port 8765 by default.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Set, Dict
import logging

# Third-party imports
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from alpaca.data.live import StockDataStream
from alpaca.data.models import Trade
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from repository root
# Script is in services/, .env is in parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
env_path = os.path.join(repo_root, '.env')
load_dotenv(env_path)

# Server configuration
WS_HOST = os.getenv('TRADE_STREAM_HOST', '0.0.0.0')
WS_PORT = int(os.getenv('TRADE_STREAM_PORT', '8765'))

# Alpaca credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("Missing Alpaca API credentials in .env file")
    sys.exit(1)


class TradeStreamServer:
    """WebSocket server that streams real-time trades from Alpaca to web clients."""

    def __init__(self):
        self.clients: Set[WebSocketServerProtocol] = set()
        self.subscriptions: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.alpaca_subscribed: Set[str] = set()  # Track which symbols are actually subscribed to Alpaca
        self.alpaca_stream: StockDataStream = None
        self.alpaca_task = None
        self.alpaca_connected = False
        self.reconnect_delay = 1  # Start with 1 second delay
        self.last_data_received = None  # Track when we last received data for health monitoring
        self.connection_established_at = None  # Track when connection was established
        self.connection_watchdog_task = None  # Watchdog task to monitor connection health
        self.max_time_without_initial_data = 60  # Max seconds to wait for first data after subscribing

    async def start(self):
        """Start the WebSocket server and Alpaca stream."""
        logger.info(f"Starting Trade Stream Server on {WS_HOST}:{WS_PORT}")

        # Initialize Alpaca data stream (but don't start it yet)
        # Use DataFeed.SIP for real-time data from all exchanges (requires paid subscription)
        # Use DataFeed.IEX for free IEX-only data
        self.alpaca_stream = StockDataStream(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            feed=DataFeed.SIP  # SIP feed includes all exchanges (NYSE, NASDAQ, FINRA, Cboe, etc.)
        )

        # DON'T start the stream yet - it will be started on first subscription
        # This avoids the busy loop in _run_forever() when there are no subscriptions
        logger.info("Alpaca stream initialized (will start on first subscription)")

        # Start WebSocket server
        async with websockets.serve(self.handle_client, WS_HOST, WS_PORT):
            logger.info(f"✅ Trade Stream Server running on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()  # Run forever

    async def _run_alpaca_stream(self):
        """Run the Alpaca WebSocket stream with automatic reconnection."""
        while True:
            try:
                logger.info("Connecting to Alpaca data stream...")
                self.alpaca_connected = False

                # Use the public run() method instead of private _run_forever()
                # Run the stream in a background task
                stream_task = asyncio.create_task(self.alpaca_stream._run_forever())

                # Mark as connected after successful start
                # CRITICAL FIX: Wait longer (5 seconds) to allow full WebSocket authentication
                # The Alpaca WebSocket needs time to: connect -> authenticate -> be ready for subscriptions
                try:
                    await asyncio.wait_for(asyncio.sleep(5), timeout=10)  # Add timeout wrapper
                except asyncio.TimeoutError:
                    logger.error("❌ Connection timeout during initial sleep!")
                    stream_task.cancel()
                    raise Exception("Connection timeout during initialization")

                # Verify the stream task is still running
                if stream_task.done():
                    logger.error("❌ Stream task died during connection!")
                    try:
                        stream_task.result()  # This will raise the exception if there was one
                    except Exception as e:
                        logger.error(f"Stream task exception: {e}")
                        raise
                    # If we get here, task completed without exception (shouldn't happen)
                    raise Exception("Stream task completed unexpectedly during connection")

                self.alpaca_connected = True
                self.connection_established_at = datetime.now()
                self.last_data_received = None  # Reset on new connection
                self.reconnect_delay = 1  # Reset delay on successful connection
                logger.info("✅ Alpaca stream connected successfully")

                # Subscribe to any pending symbols that were added before connection
                if self.subscriptions:
                    logger.info(f"Subscribing to {len(self.subscriptions)} pending symbols")
                    for symbol in list(self.subscriptions.keys()):
                        try:
                            # Add timeout to subscription attempts
                            await asyncio.wait_for(
                                self._subscribe_alpaca_symbol(symbol),
                                timeout=15
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout subscribing to {symbol}")
                        except Exception as e:
                            logger.error(f"Failed to subscribe to pending symbol {symbol}: {e}")

                # Start connection watchdog to monitor health
                self.connection_watchdog_task = asyncio.create_task(
                    self._connection_watchdog(stream_task, max_idle_time=120)
                )

                # Wait for the stream task to complete (or fail)
                # Use asyncio.wait with a timeout to prevent infinite blocking
                done, pending = await asyncio.wait(
                    [stream_task, self.connection_watchdog_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel any remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if stream_task failed
                if stream_task in done:
                    # Stream task completed - check for exceptions
                    try:
                        stream_task.result()
                    except Exception as e:
                        logger.error(f"Stream task failed: {e}")
                        raise

            except asyncio.CancelledError:
                logger.info("Alpaca stream task cancelled")
                self.alpaca_connected = False

                # Clean up watchdog task
                if self.connection_watchdog_task and not self.connection_watchdog_task.done():
                    self.connection_watchdog_task.cancel()
                    try:
                        await self.connection_watchdog_task
                    except asyncio.CancelledError:
                        pass
                raise
            except Exception as e:
                self.alpaca_connected = False
                logger.error(f"Alpaca stream error: {e}", exc_info=True)
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")

                # Clean up watchdog task
                if self.connection_watchdog_task and not self.connection_watchdog_task.done():
                    self.connection_watchdog_task.cancel()
                    try:
                        await self.connection_watchdog_task
                    except asyncio.CancelledError:
                        pass

                await asyncio.sleep(self.reconnect_delay)

                # Exponential backoff up to 60 seconds
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)

                # Recreate the stream object on error
                logger.info("Recreating Alpaca stream connection...")
                self.alpaca_stream = StockDataStream(
                    api_key=ALPACA_API_KEY,
                    secret_key=ALPACA_SECRET_KEY,
                    feed=DataFeed.SIP  # SIP feed includes all exchanges
                )

                # Re-subscribe to all active symbols
                await self._resubscribe_all_symbols()

    async def _connection_watchdog(self, stream_task, max_idle_time=120):
        """Monitor the connection health and force reconnect if stuck.

        Args:
            stream_task: The asyncio task running the Alpaca stream
            max_idle_time: Maximum seconds without data before forcing reconnect (default 120s)
        """
        logger.info(f"Starting connection watchdog (max idle time: {max_idle_time}s)")

        # Wait a bit after connection before starting health checks
        # This gives time for subscriptions to complete
        await asyncio.sleep(30)

        while not stream_task.done():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check if we have subscriptions
                if self.subscriptions:
                    now = datetime.now()

                    # Case 1: We have received data before - check idle time
                    if self.last_data_received:
                        time_since_data = (now - self.last_data_received).total_seconds()

                        if time_since_data > max_idle_time:
                            logger.error(f"⚠️  Connection appears stuck! No data received for {time_since_data:.0f}s")
                            logger.error(f"   Active subscriptions: {list(self.subscriptions.keys())}")
                            logger.error(f"   Forcing reconnection...")

                            # Cancel the stream task to trigger reconnection
                            stream_task.cancel()
                            return
                        elif time_since_data > max_idle_time / 2:
                            # Warning at half the timeout
                            logger.warning(f"⚠️  No data received for {time_since_data:.0f}s (warning threshold)")

                    # Case 2: Never received data since connection - check if waited too long
                    elif self.connection_established_at:
                        time_since_connection = (now - self.connection_established_at).total_seconds()

                        if time_since_connection > self.max_time_without_initial_data:
                            logger.error(f"⚠️  Never received any data since connection!")
                            logger.error(f"   Connected for {time_since_connection:.0f}s with no data")
                            logger.error(f"   Active subscriptions: {list(self.subscriptions.keys())}")
                            logger.error(f"   Alpaca subscribed symbols: {self.alpaca_subscribed}")
                            logger.error(f"   Forcing reconnection...")

                            # Cancel the stream task to trigger reconnection
                            stream_task.cancel()
                            return
                        elif time_since_connection > self.max_time_without_initial_data / 2:
                            logger.warning(f"⚠️  No initial data received for {time_since_connection:.0f}s since connection")

                # Also check if alpaca_connected flag is False but stream_task is still running
                # This could indicate a silent disconnection
                if not self.alpaca_connected:
                    logger.warning("⚠️  alpaca_connected is False but watchdog is running - potential silent disconnect")
                    logger.info("   Forcing reconnection due to inconsistent state...")
                    stream_task.cancel()
                    return

            except asyncio.CancelledError:
                logger.info("Watchdog cancelled")
                raise
            except Exception as e:
                logger.error(f"Watchdog error: {e}", exc_info=True)

        logger.info("Watchdog exiting - stream task completed")

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new WebSocket client connection."""
        try:
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        except:
            client_id = "unknown"
        logger.info(f"New client connected: {client_id}")

        self.clients.add(websocket)

        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up subscriptions (don't send messages to closed connection)
            await self.unsubscribe_all(websocket, send_confirmation=False)
            self.clients.discard(websocket)

            # CRITICAL: Explicitly close the websocket to prevent CLOSE_WAIT leak
            try:
                await websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket for {client_id}: {e}")

    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming message from a client."""
        try:
            data = json.loads(message)
            action = data.get('action')
            symbol = data.get('symbol', '').upper()

            if action == 'subscribe' and symbol:
                await self.subscribe_symbol(websocket, symbol)
            elif action == 'unsubscribe' and symbol:
                await self.unsubscribe_symbol(websocket, symbol)
            elif action == 'ping':
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'alpaca_connected': self.alpaca_connected
                }))
            elif action == 'health':
                # Check if alpaca_task is alive, restart if dead
                alpaca_task_status = "running"
                if self.alpaca_task is None:
                    alpaca_task_status = "not_started"
                elif self.alpaca_task.done():
                    alpaca_task_status = "dead"
                    # Auto-restart the alpaca task if it died and we have subscriptions
                    if self.subscriptions:
                        logger.warning("⚠️  Alpaca task died with active subscriptions - auto-restarting")
                        self.alpaca_task = asyncio.create_task(self._run_alpaca_stream())
                        alpaca_task_status = "restarting"

                await websocket.send(json.dumps({
                    'type': 'health',
                    'alpaca_connected': self.alpaca_connected,
                    'alpaca_task_status': alpaca_task_status,
                    'active_symbols': list(self.subscriptions.keys()),
                    'alpaca_subscribed': list(self.alpaca_subscribed),
                    'total_clients': len(self.clients),
                    'last_data_received': self.last_data_received.isoformat() if self.last_data_received else None,
                    'connection_established_at': self.connection_established_at.isoformat() if self.connection_established_at else None
                }))
            elif action == 'reconnect':
                # Force reconnection
                logger.info("🔄 Force reconnection requested via health check")
                await self._force_reconnect()
                await websocket.send(json.dumps({
                    'type': 'reconnect_initiated',
                    'message': 'Alpaca stream reconnection initiated'
                }))
            else:
                logger.warning(f"Unknown action: {action}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def subscribe_symbol(self, websocket: WebSocketServerProtocol, symbol: str):
        """Subscribe a client to trades for a specific symbol."""
        logger.info(f"🔵 [SUBSCRIBE] Client subscribing to {symbol}")
        logger.info(f"🔵 [SUBSCRIBE] Current state - Alpaca connected: {self.alpaca_connected}, Alpaca subscriptions: {self.alpaca_subscribed}")
        logger.info(f"🔵 [SUBSCRIBE] Existing subscriptions: {list(self.subscriptions.keys())}")

        # Start Alpaca stream on first subscription
        if self.alpaca_task is None:
            logger.info("🔵 [SUBSCRIBE] Starting Alpaca stream for first subscription...")
            self.alpaca_task = asyncio.create_task(self._run_alpaca_stream())

        # Check if Alpaca stream is connected
        # IMPORTANT: Don't block waiting for connection - just inform client if not ready
        if not self.alpaca_connected:
            # Send a "connecting" status instead of rejecting
            await websocket.send(json.dumps({
                'type': 'connecting',
                'symbol': symbol,
                'message': f'Connecting to Alpaca stream for {symbol}. Subscription will be activated when ready.'
            }))
            logger.info(f"🔵 [SUBSCRIBE] Alpaca stream not yet connected for {symbol} - subscription will activate when ready")
            # Don't return - let the subscription continue so it's ready when Alpaca connects

        # Add client to symbol subscriptions
        if symbol not in self.subscriptions:
            logger.info(f"🔵 [SUBSCRIBE] New symbol {symbol}, creating subscription set")
            self.subscriptions[symbol] = set()
            # Subscribe to Alpaca stream for this symbol (only if connected and not already subscribed)
            if self.alpaca_connected and symbol not in self.alpaca_subscribed:
                logger.info(f"🔵 [SUBSCRIBE] Alpaca connected and symbol new, subscribing to Alpaca for {symbol}")
                await self._subscribe_alpaca_symbol(symbol)
            elif symbol in self.alpaca_subscribed:
                logger.info(f"🔵 [SUBSCRIBE] Symbol {symbol} already subscribed in Alpaca, reusing existing subscription")
            else:
                logger.info(f"🔵 [SUBSCRIBE] Deferring Alpaca subscription for {symbol} until stream connects")
        else:
            logger.info(f"🔵 [SUBSCRIBE] Symbol {symbol} already has {len(self.subscriptions[symbol])} subscriber(s)")

        self.subscriptions[symbol].add(websocket)
        logger.info(f"🔵 [SUBSCRIBE] Added websocket to {symbol} subscription set")

        # Send confirmation
        await websocket.send(json.dumps({
            'type': 'subscribed',
            'symbol': symbol,
            'message': f'Subscribed to {symbol} trades'
        }))
        logger.info(f"🔵 [SUBSCRIBE] Sent subscription confirmation to client")

        logger.info(f"🔵 [SUBSCRIBE] ✅ Client subscribed to {symbol}. Total subscribers: {len(self.subscriptions[symbol])}")

    async def unsubscribe_symbol(self, websocket: WebSocketServerProtocol, symbol: str, send_confirmation: bool = True):
        """Unsubscribe a client from trades for a specific symbol.

        Args:
            websocket: The client websocket
            symbol: The symbol to unsubscribe from
            send_confirmation: If False, skip sending confirmation message (for cleanup of closed connections)
        """
        logger.info(f"🔴 [UNSUBSCRIBE] Client unsubscribing from {symbol}")
        logger.info(f"🔴 [UNSUBSCRIBE] Current subscriptions: {list(self.subscriptions.keys())}")

        if symbol in self.subscriptions:
            self.subscriptions[symbol].discard(websocket)
            logger.info(f"🔴 [UNSUBSCRIBE] Removed websocket from {symbol} subscription set")

            # If no more subscribers, clean up and unsubscribe from Alpaca
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]
                logger.info(f"🔴 [UNSUBSCRIBE] No more clients for {symbol}, unsubscribing from Alpaca")
                # Unsubscribe from Alpaca to stop receiving unwanted data
                await self._unsubscribe_alpaca_symbol(symbol)
                logger.info(f"🔴 [UNSUBSCRIBE] Remaining subscriptions: {list(self.subscriptions.keys())}")

            # Send confirmation (only if connection still open and confirmation requested)
            if send_confirmation:
                try:
                    await websocket.send(json.dumps({
                        'type': 'unsubscribed',
                        'symbol': symbol,
                        'message': f'Unsubscribed from {symbol} trades'
                    }))
                    logger.info(f"🔴 [UNSUBSCRIBE] Sent unsubscribe confirmation to client")
                except websockets.exceptions.ConnectionClosed:
                    logger.debug(f"🔴 [UNSUBSCRIBE] Could not send unsubscribe confirmation - connection already closed")
        else:
            logger.warning(f"🔴 [UNSUBSCRIBE] Symbol {symbol} not found in subscriptions")

    async def unsubscribe_all(self, websocket: WebSocketServerProtocol, send_confirmation: bool = True):
        """Unsubscribe a client from all symbols.

        Args:
            websocket: The client websocket
            send_confirmation: If False, skip sending confirmation messages (for cleanup of closed connections)
        """
        symbols_to_remove = []
        for symbol, subscribers in self.subscriptions.items():
            if websocket in subscribers:
                symbols_to_remove.append(symbol)

        for symbol in symbols_to_remove:
            await self.unsubscribe_symbol(websocket, symbol, send_confirmation=send_confirmation)

    async def _subscribe_alpaca_symbol(self, symbol: str):
        """Subscribe to Alpaca trade stream for a symbol."""
        logger.info(f"🟢 [ALPACA_SUB] Attempting to subscribe to Alpaca stream for {symbol}")

        # Only subscribe if Alpaca is connected
        if not self.alpaca_connected:
            logger.warning(f"🟢 [ALPACA_SUB] ❌ Cannot subscribe to {symbol} - Alpaca stream not connected yet")
            return

        logger.info(f"🟢 [ALPACA_SUB] Alpaca is connected, proceeding with subscription for {symbol}")

        async def trade_handler(trade: Trade):
            """Handler for incoming trades from Alpaca."""
            # CRITICAL FIX: Use trade.symbol from the Trade object instead of closure variable
            # The closure variable causes all handlers to use the last subscribed symbol
            actual_symbol = trade.symbol
            logger.info(f"🔔 Received trade from Alpaca for {actual_symbol}: ${trade.price:.2f} x {trade.size}")
            await self.broadcast_trade(actual_symbol, trade)

        # Subscribe to trades for this symbol
        # CRITICAL FIX: subscribe_trades() can block when stream is already running
        # Run it in a thread executor to prevent blocking the event loop
        try:
            logger.info(f"🟢 [ALPACA_SUB] Calling alpaca_stream.subscribe_trades for {symbol}")

            # Run the synchronous subscribe_trades in a thread executor to prevent blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.alpaca_stream.subscribe_trades,
                trade_handler,
                symbol
            )

            self.alpaca_subscribed.add(symbol)  # Track that we subscribed
            logger.info(f"🟢 [ALPACA_SUB] ✅ Trade handler registered for {symbol}")
            logger.info(f"🟢 [ALPACA_SUB] Current Alpaca subscriptions: {self.alpaca_subscribed}")
        except Exception as e:
            logger.error(f"🟢 [ALPACA_SUB] ❌ Failed to subscribe to {symbol}: {e}", exc_info=True)

    async def _unsubscribe_alpaca_symbol(self, symbol: str):
        """Unsubscribe from Alpaca trade stream for a symbol."""
        logger.info(f"🟠 [ALPACA_UNSUB] Attempting to unsubscribe from Alpaca stream for {symbol}")
        logger.info(f"🟠 [ALPACA_UNSUB] Current state - Alpaca connected: {self.alpaca_connected}, Alpaca subscriptions: {self.alpaca_subscribed}")

        # Only unsubscribe if we actually subscribed to Alpaca
        if symbol not in self.alpaca_subscribed:
            logger.info(f"🟠 [ALPACA_UNSUB] Symbol {symbol} was never subscribed to Alpaca, skipping unsubscribe")
            return

        # CRITICAL FIX: Don't unsubscribe if Alpaca isn't connected yet
        # This prevents sending invalid commands during authentication
        if not self.alpaca_connected:
            logger.warning(f"🟠 [ALPACA_UNSUB] Alpaca stream not connected, deferring unsubscribe for {symbol}")
            self.alpaca_subscribed.discard(symbol)  # Remove from tracking but don't send command
            logger.info(f"🟠 [ALPACA_UNSUB] Removed {symbol} from tracking (deferred). Remaining: {self.alpaca_subscribed}")
            return

        logger.info(f"🟠 [ALPACA_UNSUB] Alpaca is connected, proceeding with unsubscribe for {symbol}")
        try:
            logger.info(f"🟠 [ALPACA_UNSUB] Calling alpaca_stream.unsubscribe_trades for {symbol}...")
            # CRITICAL FIX: Run unsubscribe_trades in a thread executor to prevent blocking
            # (same pattern as subscribe_trades which was also blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.alpaca_stream.unsubscribe_trades,
                symbol
            )
            logger.info(f"🟠 [ALPACA_UNSUB] ✅ unsubscribe_trades call completed for {symbol}")
            self.alpaca_subscribed.discard(symbol)  # Remove from tracking
            logger.info(f"🟠 [ALPACA_UNSUB] ✅ Removed {symbol} from tracking. Remaining Alpaca subscriptions: {self.alpaca_subscribed}")
        except Exception as e:
            logger.error(f"🟠 [ALPACA_UNSUB] ❌ Error unsubscribing from {symbol}: {e}", exc_info=True)

    async def _resubscribe_all_symbols(self):
        """Re-subscribe to all active symbols after reconnection."""
        if not self.subscriptions:
            return

        logger.info(f"Re-subscribing to {len(self.subscriptions)} symbols after reconnection")
        for symbol in list(self.subscriptions.keys()):
            try:
                await self._subscribe_alpaca_symbol(symbol)
                logger.info(f"✅ Re-subscribed to {symbol}")
            except Exception as e:
                logger.error(f"Failed to re-subscribe to {symbol}: {e}")

    async def _force_reconnect(self):
        """Force a reconnection to Alpaca stream."""
        logger.info("🔄 Force reconnect initiated")

        # Cancel existing alpaca task if running
        if self.alpaca_task and not self.alpaca_task.done():
            logger.info("   Cancelling existing alpaca task...")
            self.alpaca_task.cancel()
            try:
                await self.alpaca_task
            except asyncio.CancelledError:
                pass

        # Reset connection state
        self.alpaca_connected = False
        self.connection_established_at = None
        self.last_data_received = None
        self.alpaca_subscribed.clear()

        # Recreate the stream object
        logger.info("   Recreating Alpaca stream connection...")
        self.alpaca_stream = StockDataStream(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            feed=DataFeed.SIP
        )

        # Start new alpaca task if we have subscriptions
        if self.subscriptions:
            logger.info(f"   Restarting with {len(self.subscriptions)} active subscriptions")
            self.alpaca_task = asyncio.create_task(self._run_alpaca_stream())
        else:
            logger.info("   No active subscriptions - will start on first subscription")
            self.alpaca_task = None

        logger.info("🔄 Force reconnect complete")

    async def broadcast_trade(self, symbol: str, trade: Trade):
        """Broadcast a trade to all subscribed clients."""
        if symbol not in self.subscriptions:
            logger.warning(f"Received trade for {symbol} but no subscribers")
            return

        # Update health check timestamp
        self.last_data_received = datetime.now()

        # Format trade data for frontend
        trade_data = {
            'type': 'trade',
            'symbol': symbol,
            'data': {
                'timestamp': trade.timestamp.isoformat(),
                'price': float(trade.price),
                'size': int(trade.size),
                'exchange': trade.exchange if hasattr(trade, 'exchange') else None,
                'conditions': trade.conditions if hasattr(trade, 'conditions') else None,
            }
        }

        message = json.dumps(trade_data)
        logger.info(f"📊 Broadcasting {symbol} trade: ${trade.price:.2f} x {trade.size} to {len(self.subscriptions[symbol])} clients")

        # Broadcast to all subscribers
        subscribers = list(self.subscriptions[symbol])  # Copy to avoid modification during iteration
        disconnected = []

        for client in subscribers:
            try:
                await client.send(message)
                logger.info(f"✅ Sent trade to client {client.remote_address}")
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
                logger.warning(f"Client {client.remote_address} disconnected during broadcast")
            except Exception as e:
                logger.error(f"Error broadcasting to client {client.remote_address}: {e}")
                disconnected.append(client)

        # Clean up disconnected clients
        for client in disconnected:
            await self.unsubscribe_all(client, send_confirmation=False)
            self.clients.discard(client)
            # Explicitly close the websocket to prevent CLOSE_WAIT leak
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"Error closing disconnected client: {e}")


async def main():
    """Main entry point."""
    server = TradeStreamServer()
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
