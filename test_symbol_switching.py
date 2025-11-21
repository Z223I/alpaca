#!/usr/bin/env python3
"""
Test WebSocket reconnection with different symbols.
Tests: AAPL -> Disconnect -> Reconnect with SGBX
"""

import asyncio
import json
import websockets
from datetime import datetime

async def test_symbol(symbol, test_num, duration=12):
    """Connect, subscribe to a symbol, and wait for messages."""
    url = "ws://localhost:8766"

    print(f"\n{'='*70}")
    print(f"[TEST {test_num}] Testing with symbol: {symbol}")
    print(f"{'='*70}")

    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Connected to {url}")

            # Subscribe
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Subscribed to {symbol}")

            # Listen for messages
            print(f"⏳ Waiting {duration} seconds for messages...")
            print("-" * 70)

            messages_received = []
            trades_received = 0
            start_time = datetime.now()

            try:
                while (datetime.now() - start_time).total_seconds() < duration:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    msg_type = data.get('type', 'unknown')
                    messages_received.append(msg_type)

                    if msg_type == 'connecting':
                        print(f"🔄 {data.get('message', '')[:60]}")
                    elif msg_type == 'subscribed':
                        print(f"✅ {data.get('message', '')}")
                    elif msg_type == 'trade':
                        trades_received += 1
                        trade_data = data.get('data', {})
                        price = trade_data.get('price', 0)
                        size = trade_data.get('size', 0)
                        timestamp = trade_data.get('timestamp', '')

                        elapsed = (datetime.now() - start_time).total_seconds()
                        print(f"🔔 Trade #{trades_received} [{elapsed:5.1f}s]: {symbol} @ ${price:.2f} x {size:,} shares")

                        # Stop after 3 trades to save time
                        if trades_received >= 3:
                            print(f"\n   ✅ Received {trades_received} trades for {symbol}, stopping early")
                            break
                    else:
                        print(f"📥 {msg_type}: {str(data)[:80]}")

            except asyncio.TimeoutError:
                pass

            elapsed = (datetime.now() - start_time).total_seconds()

            print("-" * 70)
            print(f"Summary for {symbol}:")
            print(f"  Messages received: {len(messages_received)}")
            print(f"  Message types: {set(messages_received)}")
            print(f"  Trades received: {trades_received}")
            print(f"  Time elapsed: {elapsed:.1f}s")

            # Check if we got at least a subscription confirmation
            success = 'subscribed' in messages_received

            if success:
                print(f"  ✅ Subscription confirmed for {symbol}")
            else:
                print(f"  ⚠️  No subscription confirmation for {symbol}")

            if trades_received > 0:
                print(f"  ✅ Live trades received for {symbol}")
            else:
                print(f"  ℹ️  No trades for {symbol} (may be low volume or off-market hours)")

            print(f"\n🔌 Disconnecting from {symbol}...")
            await ws.close()
            print(f"✅ Disconnected")

            return success

    except Exception as e:
        print(f"\n❌ Error testing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Test AAPL then TSLA."""

    print("=" * 70)
    print("SYMBOL SWITCHING TEST: AAPL → TSLA")
    print("=" * 70)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis test will:")
    print("1. Connect and subscribe to AAPL")
    print("2. Wait for trades/confirmation")
    print("3. Disconnect")
    print("4. Wait 3 seconds")
    print("5. Reconnect and subscribe to TSLA")
    print("6. Wait for trades/confirmation")

    # Test 1: AAPL
    aapl_success = await test_symbol("AAPL", 1, duration=12)

    # Wait between tests
    print(f"\n{'='*70}")
    print("⏳ Waiting 3 seconds before reconnecting with different symbol...")
    print(f"{'='*70}")
    await asyncio.sleep(3)

    # Test 2: TSLA
    tsla_success = await test_symbol("TSLA", 2, duration=12)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"AAPL Test: {'✅ PASSED' if aapl_success else '❌ FAILED'}")
    print(f"TSLA Test: {'✅ PASSED' if tsla_success else '❌ FAILED'}")

    if aapl_success and tsla_success:
        print("\n✅ SYMBOL SWITCHING TEST PASSED")
        print("   Backend correctly handles reconnection with different symbols")
    else:
        print("\n⚠️  Some tests failed")

    print("=" * 70)

    return aapl_success and tsla_success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
