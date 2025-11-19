#!/bin/bash
# Start WebSocket trade streaming services
# This script starts both the backend server and browser proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting WebSocket Trade Streaming Services...${NC}"

# Stop any existing instances
echo "Stopping existing servers..."
pkill -f "trade_stream_server.py" 2>/dev/null
pkill -f "browser_proxy.py" 2>/dev/null
sleep 2

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpaca

# Change to repo root
cd "$REPO_ROOT"

# Start backend server
echo -e "${GREEN}Starting backend server on port 8765...${NC}"
python3 services/trade_stream_server.py > /tmp/trade_backend.log 2>&1 &
BACKEND_PID=$!
sleep 3

# Check if backend started
if ! lsof -i :8765 > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Backend server failed to start${NC}"
    echo "Check /tmp/trade_backend.log for details"
    exit 1
fi

echo -e "${GREEN}✅ Backend server running (PID: $BACKEND_PID)${NC}"

# Start proxy server
echo -e "${GREEN}Starting proxy server on port 8766...${NC}"
python3 services/browser_proxy.py > /tmp/trade_proxy.log 2>&1 &
PROXY_PID=$!
sleep 2

# Check if proxy started
if ! lsof -i :8766 > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Proxy server failed to start${NC}"
    echo "Check /tmp/trade_proxy.log for details"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ Proxy server running (PID: $PROXY_PID)${NC}"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ WebSocket services started successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo "Backend Server:"
echo "  Port: 8765"
echo "  PID: $BACKEND_PID"
echo "  Log: /tmp/trade_backend.log"
echo ""
echo "Proxy Server (for browsers):"
echo "  Port: 8766"
echo "  PID: $PROXY_PID"
echo "  Log: /tmp/trade_proxy.log"
echo ""
echo "Browser should connect to: ws://localhost:8766"
echo ""
echo "To stop services:"
echo "  pkill -f trade_stream_server.py"
echo "  pkill -f browser_proxy.py"
echo ""
