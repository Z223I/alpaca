#!/usr/bin/env python3
"""
Test rapid reconnection scenario that mimics browser behavior.

This simulates:
1. Connect
2. Subscribe
3. Close immediately
4. Reconnect quickly (like clicking disconnect/connect rapidly)
"""

import asyncio
import json
import websockets
import time

async def connect_and_close(iteration):
    """Connect, subscribe, and close quickly."""
    url = "ws://localhost:8766"
    symbol = "AAPL"

    print(f"\n[Attempt {iteration}] Connecting...")
    try:
        # Set a connection timeout
        ws = await asyncio.wait_for(
            websockets.connect(url),
            timeout=10.0
        )
        print(f"  ✅ Connected")

        # Subscribe
        subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
        await ws.send(subscribe_msg)
        print(f"  📤 Subscribed to {symbol}")

        # Wait for at least one response
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            print(f"  📥 Got: {data.get('type')}")
        except asyncio.TimeoutError:
            print(f"  ⚠️  No response received")

        # Close
        await ws.close()
        print(f"  🔌 Closed")

        return True

    except asyncio.TimeoutError:
        print(f"  ❌ CONNECTION TIMEOUT!")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rapid_reconnections():
    """Test multiple rapid reconnections."""

    print("=" * 60)
    print("Rapid Reconnection Test (Browser-like behavior)")
    print("=" * 60)

    for i in range(1, 6):
        success = await connect_and_close(i)
        if not success:
            print(f"\n❌ FAILED on iteration {i}")
            return False

        # Small delay between reconnections (similar to browser)
        if i < 5:
            print(f"  ⏳ Waiting 0.5s before next attempt...")
            await asyncio.sleep(0.5)

    print("\n" + "=" * 60)
    print("✅ All 5 rapid reconnections successful!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_rapid_reconnections())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
        exit(1)
