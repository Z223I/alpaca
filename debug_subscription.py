#!/usr/bin/env python3
"""
Debug script to check if symbols are properly subscribed and receiving data.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from molecules.orb_alert_engine import ORBAlertEngine
from atoms.config.symbol_manager import SymbolManager

async def debug_subscription():
    """Debug the symbol subscription process."""
    print(f"=== ORB Subscription Debug - {datetime.now()} ===")
    
    try:
        # Create symbol manager to get symbols
        symbol_manager = SymbolManager()
        symbols = symbol_manager.get_symbols()
        print(f"Found {len(symbols)} symbols to monitor: {symbols[:5]}..." if len(symbols) > 5 else f"Symbols: {symbols}")
        
        # Create alert engine 
        engine = ORBAlertEngine()
        print(f"Created ORB Alert Engine")
        
        # Check data buffer state
        print(f"Data buffer symbols tracked: {engine.data_buffer.get_tracked_symbols()}")
        print(f"Data buffer statistics: {engine.data_buffer.get_statistics()}")
        
        # Try to start and check subscription
        print("Starting engine...")
        await asyncio.wait_for(engine.start(), timeout=30.0)
        
        # Check if symbols are subscribed
        print(f"Engine is running: {engine.is_running}")
        print(f"Stream client state: {engine.stream_client.state}")
        print(f"Subscribed symbols: {engine.stream_client.subscribed_symbols}")
        
        # Wait a bit to see if data flows
        print("Waiting 10 seconds for data...")
        await asyncio.sleep(10)
        
        # Check data buffer again
        buffer_stats = engine.data_buffer.get_statistics()
        print(f"Data buffer after 10s: {buffer_stats}")
        
        if buffer_stats['total_messages_received'] == 0:
            print("❌ NO DATA RECEIVED - Subscription issue detected!")
        else:
            print(f"✅ Received {buffer_stats['total_messages_received']} messages")
            
        await engine.stop()
        
    except asyncio.TimeoutError:
        print("❌ TIMEOUT - Engine failed to start within 30 seconds")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(debug_subscription())