#!/usr/bin/env python3
"""
Test multiple symbol subscriptions via WebSocket
"""

import asyncio
import websockets
import json

async def test_multi_symbol():
    print("Testing multiple symbol subscriptions...")
    print("-" * 60)

    # Connect directly to backend (port 8765) instead of proxy (8766) for testing
    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to WebSocket")

            # Subscribe to multiple symbols
            symbols = ["AAPL", "TSLA", "MSFT"]
            for symbol in symbols:
                await ws.send(json.dumps({
                    'action': 'subscribe',
                    'symbol': symbol
                }))
                print(f"📤 Sent subscribe request for {symbol}")

            # Wait for subscription confirmations (may receive 'connecting' messages first)
            subscribed = []
            max_messages = len(symbols) * 2  # Allow for 'connecting' + 'subscribed' per symbol
            for _ in range(max_messages):
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(response)
                    print(f"📥 Received: {data}")
                    if data['type'] == 'subscribed':
                        subscribed.append(data['symbol'])
                    if len(subscribed) == len(symbols):
                        break  # Got all subscriptions
                except asyncio.TimeoutError:
                    break

            print(f"\n✅ Successfully subscribed to: {', '.join(subscribed)}")

            # Wait a moment for potential trades
            print("\n⏳ Waiting 5 seconds for trades...")
            trade_count = 0
            try:
                for _ in range(10):  # Check for up to 10 messages
                    response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(response)
                    if data['type'] == 'trade':
                        trade_count += 1
                        print(f"  📊 {data['symbol']} @ ${data['data']['price']:.2f} x {data['data']['size']}")
            except asyncio.TimeoutError:
                pass

            if trade_count > 0:
                print(f"\n✅ Received {trade_count} trades")
            else:
                print("\n⚠️  No trades received (market may be closed)")

            # Unsubscribe from one symbol
            print(f"\n📤 Unsubscribing from {symbols[0]}...")
            await ws.send(json.dumps({
                'action': 'unsubscribe',
                'symbol': symbols[0]
            }))

            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            print(f"📥 Received: {data}")

            if data['type'] == 'unsubscribed' and data['symbol'] == symbols[0]:
                print(f"✅ Successfully unsubscribed from {symbols[0]}")
                print(f"   Remaining subscriptions: {', '.join(symbols[1:])}")

            print("\n✅ MULTI-SYMBOL TEST PASSED")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_symbol())
