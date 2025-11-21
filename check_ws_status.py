#!/usr/bin/env python3
"""Check WebSocket server status and active subscriptions."""

import asyncio
import json
import websockets

async def check_status():
    """Query the backend server for status."""
    backend_url = 'ws://localhost:8765'

    print("=" * 70)
    print("WebSocket Server Status Check")
    print("=" * 70)

    try:
        async with websockets.connect(backend_url) as ws:
            print(f"✅ Connected to backend server at {backend_url}\n")

            # Send health check request
            health_request = json.dumps({'action': 'health'})
            await ws.send(health_request)

            # Wait for response
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)

            print("📊 Server Health Status:")
            print(f"  Response Type: {data.get('type', 'N/A')}")
            print(f"  Alpaca Connected: {data.get('alpaca_connected', False)}")
            print(f"  Total Clients: {data.get('total_clients', 0)}")
            print(f"\n📈 Active Symbol Subscriptions:")

            active_symbols = data.get('active_symbols', [])
            if active_symbols:
                for symbol in active_symbols:
                    print(f"  - {symbol}")
            else:
                print("  (No active subscriptions)")

            print("\n" + "=" * 70)

            if not data.get('alpaca_connected'):
                print("\n⚠️  WARNING: Alpaca stream is NOT connected!")
                print("The server is running but not receiving data from Alpaca.")
            else:
                print("\n✅ All systems operational")

            return data

    except asyncio.TimeoutError:
        print("❌ Timeout waiting for server response")
        print("Server may be unresponsive or not implementing health checks")
        return None
    except ConnectionRefusedError:
        print(f"❌ Cannot connect to {backend_url}")
        print("Is the trade_stream_server.py running?")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(check_status())
