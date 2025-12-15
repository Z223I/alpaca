#!/usr/bin/env python3
"""Test backend directly on port 8765."""

import asyncio
import json
import websockets

async def test_backend_direct():
    """Test direct connection to backend."""
    url = "ws://localhost:8765"

    print("Testing DIRECT connection to backend on port 8765...")
    print("-" * 60)

    try:
        async with websockets.connect(url) as ws:
            print("✅ Connected directly to backend")

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
    asyncio.run(test_backend_direct())
