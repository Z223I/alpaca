#!/usr/bin/env python3
"""
Test real-time trade streaming with live market data.
Since the market is open, we should receive actual trades from Alpaca.
"""

import asyncio
import json
import websockets
from datetime import datetime

# Exchange code to name mapping (Source: Alpaca API Documentation)
EXCHANGE_NAMES = {
    'A': 'NYSE American (AMEX)',
    'B': 'NASDAQ OMX BX',
    'C': 'National Stock Exchange',
    'D': 'FINRA ADF',
    'E': 'Market Independent',
    'H': 'MIAX',
    'I': 'International Securities Exchange',
    'J': 'Cboe EDGA',
    'K': 'Cboe EDGX',
    'L': 'Long Term Stock Exchange',
    'M': 'Chicago Stock Exchange',
    'N': 'New York Stock Exchange',
    'P': 'NYSE Arca',
    'Q': 'NASDAQ OMX',
    'S': 'NASDAQ Small Cap',
    'T': 'NASDAQ Int',
    'U': 'Members Exchange',
    'V': 'IEX',
    'W': 'CBOE',
    'X': 'NASDAQ OMX PSX',
    'Y': 'Cboe BYX',
    'Z': 'Cboe BZX'
}

def get_exchange_name(code):
    """Convert exchange code to full name."""
    if not code or code == 'N/A':
        return 'N/A'
    return EXCHANGE_NAMES.get(code, code)

async def test_live_trades():
    """Connect and receive real-time trades."""

    url = "ws://localhost:8766"
    symbol = "AAPL"  # Highly liquid stock, should have frequent trades

    print("=" * 70)
    print("LIVE TRADE STREAMING TEST (Market is Open)")
    print("=" * 70)
    print(f"\nConnecting to {url}...")
    print(f"Subscribing to {symbol} trades...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    try:
        async with websockets.connect(url) as ws:
            print("✅ Connected!")

            # Subscribe to AAPL
            subscribe_msg = json.dumps({"action": "subscribe", "symbol": symbol})
            await ws.send(subscribe_msg)
            print(f"📤 Sent subscription request for {symbol}\n")

            # Listen for messages
            print("⏳ Waiting for trades (30 seconds)...")
            print("-" * 70)

            trade_count = 0
            start_time = datetime.now()

            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(msg)
                    msg_type = data.get('type', 'unknown')

                    if msg_type == 'connecting':
                        print(f"🔄 {data.get('message')}")
                    elif msg_type == 'subscribed':
                        print(f"✅ {data.get('message')}")
                        print(f"   Waiting for live trades from Alpaca...\n")
                    elif msg_type == 'trade':
                        trade_count += 1
                        trade_data = data.get('data', {})
                        price = trade_data.get('price', 0)
                        size = trade_data.get('size', 0)
                        timestamp = trade_data.get('timestamp', '')
                        exchange = trade_data.get('exchange', 'N/A')
                        exchange_name = get_exchange_name(exchange)

                        elapsed = (datetime.now() - start_time).total_seconds()

                        print(f"🔔 Trade #{trade_count} [{elapsed:5.1f}s]:")
                        print(f"   Symbol: {data.get('symbol')}")
                        print(f"   Price: ${price:.2f}")
                        print(f"   Size: {size:,} shares")
                        print(f"   Exchange: {exchange_name}")
                        print(f"   Time: {timestamp}")
                        print()

                        # Stop after receiving 5 trades
                        if trade_count >= 5:
                            print("-" * 70)
                            print(f"✅ Successfully received {trade_count} live trades!")
                            print(f"   Time elapsed: {elapsed:.1f} seconds")
                            print(f"   Average: {elapsed/trade_count:.2f}s per trade")
                            break
                    else:
                        print(f"📥 Other message: {msg_type}")

            except asyncio.TimeoutError:
                elapsed = (datetime.now() - start_time).total_seconds()
                print("-" * 70)
                if trade_count > 0:
                    print(f"✅ Received {trade_count} trades in {elapsed:.1f} seconds")
                    print("   (Stopped due to timeout)")
                else:
                    print("⚠️  No trades received in 30 seconds")
                    print("   Possible reasons:")
                    print("   - Market might be closed")
                    print("   - Symbol might not be trading")
                    print("   - Alpaca connection might not be established yet")

            print("\n" + "=" * 70)
            print("Test Complete")
            print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_live_trades())
