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
                await asyncio.sleep(5)  # Give it time to establish connection and authenticate
                self.alpaca_connected = True
                self.reconnect_delay = 1  # Reset delay on successful connection
                logger.info("✅ Alpaca stream connected successfully")

                # Subscribe to any pending symbols that were added before connection
                if self.subscriptions:
                    logger.info(f"Subscribing to {len(self.subscriptions)} pending symbols")
                    for symbol in list(self.subscriptions.keys()):
                        try:
                            await self._subscribe_alpaca_symbol(symbol)
                        except Exception as e:
                            logger.error(f"Failed to subscribe to pending symbol {symbol}: {e}")

                # Wait for the stream task to complete (or fail)
                await stream_task

            except asyncio.CancelledError:
                logger.info("Alpaca stream task cancelled")
                self.alpaca_connected = False
                raise
            except Exception as e:
                self.alpaca_connected = False
                logger.error(f"Alpaca stream error: {e}", exc_info=True)
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
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
                await websocket.send(json.dumps({
                    'type': 'health',
                    'alpaca_connected': self.alpaca_connected,
                    'active_symbols': list(self.subscriptions.keys()),
                    'total_clients': len(self.clients)
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
        # Only unsubscribe if we actually subscribed to Alpaca
        if symbol not in self.alpaca_subscribed:
            logger.info(f"Symbol {symbol} was never subscribed to Alpaca, skipping unsubscribe")
            return

        # CRITICAL FIX: Don't unsubscribe if Alpaca isn't connected yet
        # This prevents sending invalid commands during authentication
        if not self.alpaca_connected:
            logger.warning(f"Alpaca stream not connected, deferring unsubscribe for {symbol}")
            self.alpaca_subscribed.discard(symbol)  # Remove from tracking but don't send command
            return

        logger.info(f"Unsubscribing from Alpaca stream for {symbol}")
        try:
            self.alpaca_stream.unsubscribe_trades(symbol)
            self.alpaca_subscribed.discard(symbol)  # Remove from tracking
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")

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

    async def broadcast_trade(self, symbol: str, trade: Trade):
        """Broadcast a trade to all subscribed clients."""
        if symbol not in self.subscriptions:
            logger.warning(f"Received trade for {symbol} but no subscribers")
            return

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
