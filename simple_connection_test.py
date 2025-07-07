#!/usr/bin/env python3
"""
Simple connection test to check Alpaca websocket connection.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.websocket.alpaca_stream import AlpacaStreamClient

async def test_connection():
    """Test basic Alpaca websocket connection."""
    print(f"=== Simple Connection Test - {datetime.now()} ===")
    
    try:
        # Create stream client
        client = AlpacaStreamClient()
        print(f"Created stream client")
        print(f"Websocket URL: {client._get_websocket_url()}")
        print(f"Initial state: {client.state}")
        
        # Try to connect
        print("Attempting connection...")
        success = await asyncio.wait_for(client.connect(), timeout=15.0)
        
        if success:
            print(f"✅ Connected successfully! State: {client.state}")
            
            # Try to subscribe to a single symbol
            print("Attempting subscription to AAPL...")
            sub_success = await client.subscribe_bars(['AAPL'])
            
            if sub_success:
                print(f"✅ Subscribed successfully! State: {client.state}")
                print(f"Subscribed symbols: {client.subscribed_symbols}")
                
                # Listen for a few seconds
                print("Listening for 5 seconds...")
                try:
                    await asyncio.wait_for(client.listen(), timeout=5.0)
                except asyncio.TimeoutError:
                    print("Listening timed out (expected)")
                    
            else:
                print("❌ Subscription failed")
        else:
            print("❌ Connection failed")
            
        await client.disconnect()
        
    except asyncio.TimeoutError:
        print("❌ TIMEOUT - Connection failed within 15 seconds")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())