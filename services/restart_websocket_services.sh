#!/bin/bash
# Restart WebSocket services for Market Sentinel

echo "Stopping existing services..."
pkill -f "trade_stream_server.py"
pkill -f "browser_proxy.py"
sleep 2

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpaca

echo "Starting Trade Stream Server..."
python3 services/trade_stream_server.py > /tmp/trade_stream.log 2>&1 &
BACKEND_PID=$!
sleep 3

echo "Starting Browser Proxy..."
python3 services/browser_proxy.py > /tmp/browser_proxy.log 2>&1 &
PROXY_PID=$!
sleep 2

echo ""
echo "✅ Services started:"
echo "   Backend PID: $BACKEND_PID"
echo "   Proxy PID: $PROXY_PID"
echo ""
echo "📊 Logs:"
echo "   Backend: tail -f /tmp/trade_stream.log"
echo "   Proxy: tail -f /tmp/browser_proxy.log"
echo ""
echo "🧪 Test:"
echo "   Browser: open public_html/test_live.html"
echo "   CLI: python3 test_health.py"
