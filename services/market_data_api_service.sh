#!/bin/bash
# Startup script for Market Data API Flask service
# This script is called by systemd to start the persistent Flask API server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
API_SCRIPT="$REPO_ROOT/cgi-bin/api/market_data.py"

# Load environment from conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate alpaca

# Change to repository root for proper .env loading
cd "$REPO_ROOT"

# Log file
LOG_FILE="/tmp/market_data_api.log"

echo "========================================" >> "$LOG_FILE"
echo "Market Data API Service Starting" >> "$LOG_FILE"
echo "Time: $(date)" >> "$LOG_FILE"
echo "Script: $API_SCRIPT" >> "$LOG_FILE"
echo "Working Directory: $(pwd)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run the Flask application
# The script already has host='0.0.0.0' and port=5000 configured
exec python3 "$API_SCRIPT" >> "$LOG_FILE" 2>&1
