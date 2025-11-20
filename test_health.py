#!/usr/bin/env python3
"""Check the health of the WebSocket backend."""

import asyncio
import json
import websockets

async def check_health():
    """Check backend health status."""
    url = "ws://localhost:8766"

    print("Checking WebSocket backend health...")
    print("-" * 60)

    try:
        async with websockets.connect(url) as ws:
            print("✅ Connected to proxy")

            # Send health check
            health_msg = json.dumps({"action": "health"})
            await ws.send(health_msg)
            print("📤 Sent health check request")

            # Wait for response
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)

            print("\n📊 Backend Status:")
            print(f"  Type: {data.get('type')}")
            print(f"  Alpaca Connected: {data.get('alpaca_connected')}")
            print(f"  Active Symbols: {data.get('active_symbols')}")
            print(f"  Total Clients: {data.get('total_clients')}")

            if not data.get('alpaca_connected'):
                print("\n⚠️  WARNING: Alpaca stream is NOT connected!")
                print("   This is the likely cause of reconnection issues.")
                print("   The backend is running but can't reach Alpaca.")
            else:
                print("\n✅ Alpaca stream is connected and healthy")

            return data

    except asyncio.TimeoutError:
        print("❌ Health check timeout - backend not responding")
        return None
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(check_health())
