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
from websockets.server import WebSocketServerProtocol
from alpaca.data.live import StockDataStream
from alpaca.data.models import Trade
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
        self.alpaca_stream: StockDataStream = None
        self.alpaca_task = None

    async def start(self):
        """Start the WebSocket server and Alpaca stream."""
        logger.info(f"Starting Trade Stream Server on {WS_HOST}:{WS_PORT}")

        # Initialize Alpaca data stream
        self.alpaca_stream = StockDataStream(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )

        # Start Alpaca stream in background
        self.alpaca_task = asyncio.create_task(self._run_alpaca_stream())

        # Start WebSocket server
        async with websockets.serve(self.handle_client, WS_HOST, WS_PORT):
            logger.info(f"✅ Trade Stream Server running on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()  # Run forever

    async def _run_alpaca_stream(self):
        """Run the Alpaca WebSocket stream."""
        try:
            logger.info("Starting Alpaca data stream...")
            await self.alpaca_stream._run_forever()
        except Exception as e:
            logger.error(f"Alpaca stream error: {e}")
            raise

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
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
            await self.unsubscribe_all(websocket)
            self.clients.discard(websocket)

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
                await websocket.send(json.dumps({'type': 'pong'}))
            else:
                logger.warning(f"Unknown action: {action}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def subscribe_symbol(self, websocket: WebSocketServerProtocol, symbol: str):
        """Subscribe a client to trades for a specific symbol."""
        logger.info(f"Client subscribing to {symbol}")

        # Add client to symbol subscriptions
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
            # Subscribe to Alpaca stream for this symbol
            await self._subscribe_alpaca_symbol(symbol)

        self.subscriptions[symbol].add(websocket)

        # Send confirmation
        await websocket.send(json.dumps({
            'type': 'subscribed',
            'symbol': symbol,
            'message': f'Subscribed to {symbol} trades'
        }))

        logger.info(f"Client subscribed to {symbol}. Total subscribers: {len(self.subscriptions[symbol])}")

    async def unsubscribe_symbol(self, websocket: WebSocketServerProtocol, symbol: str):
        """Unsubscribe a client from trades for a specific symbol."""
        logger.info(f"Client unsubscribing from {symbol}")

        if symbol in self.subscriptions:
            self.subscriptions[symbol].discard(websocket)

            # If no more subscribers, unsubscribe from Alpaca
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]
                await self._unsubscribe_alpaca_symbol(symbol)

            # Send confirmation
            await websocket.send(json.dumps({
                'type': 'unsubscribed',
                'symbol': symbol,
                'message': f'Unsubscribed from {symbol} trades'
            }))

    async def unsubscribe_all(self, websocket: WebSocketServerProtocol):
        """Unsubscribe a client from all symbols."""
        symbols_to_remove = []
        for symbol, subscribers in self.subscriptions.items():
            if websocket in subscribers:
                symbols_to_remove.append(symbol)

        for symbol in symbols_to_remove:
            await self.unsubscribe_symbol(websocket, symbol)

    async def _subscribe_alpaca_symbol(self, symbol: str):
        """Subscribe to Alpaca trade stream for a symbol."""
        logger.info(f"Subscribing to Alpaca stream for {symbol}")

        async def trade_handler(trade: Trade):
            """Handler for incoming trades from Alpaca."""
            logger.info(f"🔔 Received trade from Alpaca for {symbol}: ${trade.price:.2f} x {trade.size}")
            await self.broadcast_trade(symbol, trade)

        # Subscribe to trades for this symbol
        self.alpaca_stream.subscribe_trades(trade_handler, symbol)
        logger.info(f"✅ Trade handler registered for {symbol}")

    async def _unsubscribe_alpaca_symbol(self, symbol: str):
        """Unsubscribe from Alpaca trade stream for a symbol."""
        logger.info(f"Unsubscribing from Alpaca stream for {symbol}")
        self.alpaca_stream.unsubscribe_trades(symbol)

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
            await self.unsubscribe_all(client)
            self.clients.discard(client)


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
