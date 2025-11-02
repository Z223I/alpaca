#!/bin/bash
# Start the Market Sentinel Trade Stream Server

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpaca

# Change to repo root for .env file access
cd "$REPO_ROOT"

# Install required dependencies if needed
echo "Checking dependencies..."
pip install -q websockets python-dotenv alpaca-py

# Start the server
echo "Starting Trade Stream Server..."
python3 "$SCRIPT_DIR/trade_stream_server.py"
