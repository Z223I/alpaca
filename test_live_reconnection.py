#!/usr/bin/env python3
"""
Test reconnection with live market data.
This simulates the browser disconnect/reconnect scenario with real trades.
"""

import asyncio
import json
import websockets
from datetime import datetime

async def connect_and_receive_trades(connection_num, duration=10):
    """Connect, receive trades, then disconnect."""
    url = "ws://localhost:8766"
    symbol = "AAPL"

    print(f"\n[Connection #{connection_num}]")
    print("-" * 60)

    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Connected")

            # Subscribe
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Subscribed to {symbol}")

            # Listen for trades
            trade_count = 0
            start_time = datetime.now()

            try:
                while (datetime.now() - start_time).total_seconds() < duration:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.dumps(msg)
                    msg_type = data.get('type')

                    if msg_type == 'trade':
                        trade_count += 1
                        price = data.get('data', {}).get('price', 0)
                        print(f"  🔔 Trade #{trade_count}: ${price:.2f}")

            except asyncio.TimeoutError:
                pass

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"✅ Received {trade_count} trades in {elapsed:.1f}s")
            print(f"🔌 Disconnecting...")

            return trade_count > 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_live_reconnection():
    """Test multiple connect/disconnect cycles with live data."""

    print("=" * 70)
    print("LIVE RECONNECTION TEST (Market is Open)")
    print("=" * 70)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis test will:")
    print("1. Connect and receive trades")
    print("2. Disconnect")
    print("3. Reconnect and receive more trades")
    print("4. Repeat 3 times")

    success_count = 0

    for i in range(1, 4):
        result = await connect_and_receive_trades(i, duration=10)
        if result:
            success_count += 1

        # Wait between connections
        if i < 3:
            print(f"\n⏳ Waiting 2 seconds before reconnecting...")
            await asyncio.sleep(2)

    print("\n" + "=" * 70)
    if success_count == 3:
        print(f"✅ ALL 3 RECONNECTIONS SUCCESSFUL!")
        print(f"   Live trades received on every connection")
    else:
        print(f"⚠️  {success_count}/3 connections received trades")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_live_reconnection())
