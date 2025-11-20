#!/usr/bin/env python3
"""
Test WebSocket reconnection behavior for debugging.

This script simulates what the browser does:
1. Connect to the proxy
2. Subscribe to a symbol
3. Disconnect
4. Reconnect and subscribe again
5. Check if it works
"""

import asyncio
import json
import websockets
import time

async def test_reconnection():
    """Test connect -> disconnect -> reconnect cycle."""

    url = "ws://localhost:8766"
    symbol = "AAPL"

    print("=" * 60)
    print("WebSocket Reconnection Test")
    print("=" * 60)

    # FIRST CONNECTION
    print("\n[TEST 1] First connection and subscription")
    print("-" * 60)
    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Connected to {url}")

            # Subscribe
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Sent subscribe request for {symbol}")

            # Wait for responses
            print("⏳ Waiting for messages (5 seconds)...")
            try:
                for i in range(5):
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    print(f"📥 Received: {data.get('type', 'unknown')} - {data.get('message', data.get('symbol', ''))}")
            except asyncio.TimeoutError:
                print("⏱️  No more messages (timeout)")

            print("🔌 Closing first connection...")
            await ws.close()
            print("✅ First connection closed cleanly")
    except Exception as e:
        print(f"❌ First connection failed: {e}")
        return False

    # Wait a bit
    print("\n⏳ Waiting 2 seconds before reconnecting...")
    await asyncio.sleep(2)

    # SECOND CONNECTION (RECONNECT)
    print("\n[TEST 2] Reconnection and re-subscription")
    print("-" * 60)
    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Reconnected to {url}")

            # Subscribe again
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Sent subscribe request for {symbol}")

            # Wait for responses
            print("⏳ Waiting for messages (5 seconds)...")
            try:
                for i in range(5):
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    print(f"📥 Received: {data.get('type', 'unknown')} - {data.get('message', data.get('symbol', ''))}")
            except asyncio.TimeoutError:
                print("⏱️  No more messages (timeout)")

            print("🔌 Closing second connection...")
            await ws.close()
            print("✅ Second connection closed cleanly")
    except Exception as e:
        print(f"❌ RECONNECTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Wait a bit
    print("\n⏳ Waiting 2 seconds before third connection...")
    await asyncio.sleep(2)

    # THIRD CONNECTION
    print("\n[TEST 3] Third connection to verify stability")
    print("-" * 60)
    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Third connection established to {url}")

            # Subscribe again
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Sent subscribe request for {symbol}")

            # Wait for responses
            print("⏳ Waiting for messages (5 seconds)...")
            try:
                for i in range(5):
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    print(f"📥 Received: {data.get('type', 'unknown')} - {data.get('message', data.get('symbol', ''))}")
            except asyncio.TimeoutError:
                print("⏱️  No more messages (timeout)")

            print("🔌 Closing third connection...")
            await ws.close()
            print("✅ Third connection closed cleanly")
    except Exception as e:
        print(f"❌ THIRD CONNECTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Reconnection works correctly")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_reconnection())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
