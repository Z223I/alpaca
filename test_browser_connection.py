#!/usr/bin/env python3
"""
Diagnostic script to test browser WebSocket connection.
"""
import asyncio
import json
import websockets

async def test_connection():
    """Test connection to browser proxy."""
    uri = "ws://localhost:8766"

    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connected!")

        # Test ping
        print("\n1. Testing ping...")
        await websocket.send(json.dumps({"action": "ping"}))
        response = await websocket.recv()
        print(f"   Response: {response}")

        # Test health check
        print("\n2. Testing health check...")
        await websocket.send(json.dumps({"action": "health"}))
        response = await websocket.recv()
        print(f"   Response: {response}")

        # Test subscription to a real symbol
        print("\n3. Testing subscription to AAPL...")
        await websocket.send(json.dumps({"action": "subscribe", "symbol": "AAPL"}))
        response = await websocket.recv()
        print(f"   Response: {response}")

        # Wait for trades (with timeout)
        print("\n4. Waiting for trades (10 seconds)...")
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                if data.get('type') == 'trade':
                    print(f"   📊 Trade received: {data['symbol']} @ ${data['data']['price']:.2f} x {data['data']['size']}")
                    break
                else:
                    print(f"   Message: {response}")
        except asyncio.TimeoutError:
            print("   ⏱️ No trades received within 10 seconds (market may be closed)")

        print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_connection())
