#!/bin/bash
PROJECT_ROOT="/home/wilsonb/dl/github.com/Z223I/alpaca"
cd "$PROJECT_ROOT"

# Explicitly set the correct API keys (override any shell environment variables)
export APCA_API_KEY_ID=AKE3BC8FAL1SH0V2C1CE
export APCA_API_SECRET_KEY=QT8h5l8GJ6EakAwkSM9VsAM2XgIEtiq6xd6NM2Tb
export APCA_API_BASE_URL=https://api.alpaca.markets

# Also load any other vars from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source <(grep -v '^#' "$PROJECT_ROOT/.env" | grep -v '^$' | sed 's/#.*//' | grep -v APCA)
    set +a
fi

PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python "$PROJECT_ROOT/code/premarket_top_gainers.py" "$@"
