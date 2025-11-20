#!/usr/bin/env python3
"""
Test a sustained connection that waits for Alpaca to connect.

This test:
1. Connects
2. Subscribes
3. Waits 10 seconds for Alpaca to connect and send trades
4. Disconnects
5. Reconnects
6. Verifies it still works
"""

import asyncio
import json
import websockets

async def test_sustained():
    """Test sustained connection with proper wait time."""

    url = "ws://localhost:8766"
    symbol = "AAPL"

    print("=" * 60)
    print("Sustained Connection Test")
    print("=" * 60)

    # FIRST CONNECTION
    print("\n[TEST 1] First connection - wait for Alpaca to connect")
    print("-" * 60)
    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Connected to {url}")

            # Subscribe
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Subscribed to {symbol}")

            # Wait for responses (including Alpaca connection)
            print("⏳ Waiting 10 seconds for Alpaca to connect...")
            messages_received = []
            try:
                for i in range(20):  # Try to receive up to 20 messages over 10 seconds
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(msg)
                    msg_type = data.get('type', 'unknown')
                    messages_received.append(msg_type)
                    print(f"  📥 {i+1}. {msg_type}: {data.get('message', data.get('symbol', ''))[:50]}")
            except asyncio.TimeoutError:
                print(f"  ⏱️  Timeout (received {len(messages_received)} messages)")

            print(f"✅ Received {len(messages_received)} messages: {set(messages_received)}")
            print("🔌 Disconnecting...")
            await ws.close()
    except Exception as e:
        print(f"❌ First connection failed: {e}")
        return False

    # Wait before reconnecting
    print("\n⏳ Waiting 3 seconds before reconnecting...")
    await asyncio.sleep(3)

    # SECOND CONNECTION (RECONNECT)
    print("\n[TEST 2] Reconnection test")
    print("-" * 60)
    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Reconnected to {url}")

            # Subscribe again
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Re-subscribed to {symbol}")

            # Wait for responses
            print("⏳ Waiting 5 seconds for messages...")
            messages_received = []
            try:
                for i in range(10):
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(msg)
                    msg_type = data.get('type', 'unknown')
                    messages_received.append(msg_type)
                    print(f"  📥 {i+1}. {msg_type}: {data.get('message', data.get('symbol', ''))[:50]}")
            except asyncio.TimeoutError:
                print(f"  ⏱️  Timeout (received {len(messages_received)} messages)")

            print(f"✅ Received {len(messages_received)} messages: {set(messages_received)}")
            print("🔌 Disconnecting...")
            await ws.close()

            if not messages_received:
                print("⚠️  WARNING: No messages received on reconnection")
                return False

    except Exception as e:
        print(f"❌ RECONNECTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ SUSTAINED CONNECTION TEST PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_sustained())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
        exit(1)
