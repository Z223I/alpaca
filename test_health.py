#!/usr/bin/env python3
"""Check the health of the WebSocket backend."""

import asyncio
import json
import sys
import websockets

async def check_health(force_reconnect=False):
    """Check backend health status.

    Args:
        force_reconnect: If True, trigger a forced reconnection to Alpaca
    """
    url = "ws://localhost:8766"

    print("Checking WebSocket backend health...")
    print("-" * 60)

    try:
        async with websockets.connect(url) as ws:
            print("✅ Connected to proxy")

            # Send health check
            health_msg = json.dumps({"action": "health"})
            await ws.send(health_msg)
            print("📤 Sent health check request")

            # Wait for response
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)

            print("\n📊 Backend Status:")
            print(f"  Type: {data.get('type')}")
            print(f"  Alpaca Connected: {data.get('alpaca_connected')}")
            print(f"  Alpaca Task Status: {data.get('alpaca_task_status', 'unknown')}")
            print(f"  Active Symbols: {data.get('active_symbols')}")
            print(f"  Alpaca Subscribed: {data.get('alpaca_subscribed', 'unknown')}")
            print(f"  Total Clients: {data.get('total_clients')}")
            print(f"  Last Data Received: {data.get('last_data_received', 'never')}")
            print(f"  Connection Established: {data.get('connection_established_at', 'never')}")

            if not data.get('alpaca_connected'):
                print("\n⚠️  WARNING: Alpaca stream is NOT connected!")
                print("   This is the likely cause of reconnection issues.")
                print("   The backend is running but can't reach Alpaca.")

                task_status = data.get('alpaca_task_status', 'unknown')
                if task_status == 'dead':
                    print("\n⚠️  Alpaca task is DEAD - it should auto-restart on next subscription")
                elif task_status == 'restarting':
                    print("\n🔄 Alpaca task is being restarted...")
                elif task_status == 'not_started':
                    print("\n📋 Alpaca task not started yet (waiting for first subscription)")

                if force_reconnect:
                    print("\n🔄 Forcing reconnection...")
                    reconnect_msg = json.dumps({"action": "reconnect"})
                    await ws.send(reconnect_msg)
                    reconnect_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    reconnect_data = json.loads(reconnect_response)
                    print(f"   Response: {reconnect_data.get('message', reconnect_data)}")
                else:
                    print("\n💡 TIP: Run with --reconnect flag to force reconnection")
            else:
                print("\n✅ Alpaca stream is connected and healthy")

            return data

    except asyncio.TimeoutError:
        print("❌ Health check timeout - backend not responding")
        return None
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    force_reconnect = "--reconnect" in sys.argv or "-r" in sys.argv
    asyncio.run(check_health(force_reconnect=force_reconnect))
