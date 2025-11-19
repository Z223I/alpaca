#!/usr/bin/env python3
"""
Browser-friendly WebSocket proxy for the trade stream server.

This proxy accepts connections from browsers on port 8766 and forwards
them to the actual trade stream server on port 8765 using the Python
websockets client (which works reliably).
"""

import asyncio
import json
import logging
from typing import Set

import websockets
from websockets.legacy.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROXY_PORT = 8766  # Browser connects here
BACKEND_URL = 'ws://localhost:8765'  # Real server


class ProxyServer:
    """Simple WebSocket proxy that forwards between browser and backend."""

    def __init__(self):
        self.active_connections: Set[WebSocketServerProtocol] = set()

    async def handle_browser_client(self, browser_ws: WebSocketServerProtocol):
        """Handle a browser client connection."""
        client_id = f"{browser_ws.remote_address[0]}:{browser_ws.remote_address[1]}"
        logger.info(f"Browser client connected: {client_id}")

        self.active_connections.add(browser_ws)
        backend_ws = None

        try:
            # Connect to backend server
            logger.info(f"Connecting to backend: {BACKEND_URL}")
            backend_ws = await websockets.connect(BACKEND_URL)
            logger.info(f"Connected to backend for client {client_id}")

            # Create tasks for bidirectional forwarding
            browser_to_backend = asyncio.create_task(
                self._forward_messages(browser_ws, backend_ws, f"{client_id} -> backend")
            )
            backend_to_browser = asyncio.create_task(
                self._forward_messages(backend_ws, browser_ws, f"backend -> {client_id}")
            )

            # Wait for either task to complete (one will complete when connection closes)
            done, pending = await asyncio.wait(
                [browser_to_backend, backend_to_browser],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the remaining task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
        finally:
            # Cleanup
            if backend_ws:
                await backend_ws.close()
            self.active_connections.discard(browser_ws)
            logger.info(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")

    async def _forward_messages(self, source, destination, direction: str):
        """Forward messages from source to destination WebSocket."""
        try:
            async for message in source:
                logger.debug(f"{direction}: {message[:100] if len(message) > 100 else message}")
                await destination.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"{direction}: Connection closed")
        except Exception as e:
            logger.error(f"{direction}: Error forwarding message: {e}")
            raise

    async def start(self):
        """Start the proxy server."""
        logger.info(f"Starting WebSocket proxy on port {PROXY_PORT}")
        logger.info(f"Forwarding to backend: {BACKEND_URL}")

        async with websockets.serve(self.handle_browser_client, '0.0.0.0', PROXY_PORT):
            logger.info(f"✅ Proxy server running on ws://0.0.0.0:{PROXY_PORT}")
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point."""
    proxy = ProxyServer()
    await proxy.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Proxy server stopped by user")
    except Exception as e:
        logger.error(f"Proxy server error: {e}")
