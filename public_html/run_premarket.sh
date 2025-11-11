#!/bin/bash
cd /home/wilsonb/dl/github.com/Z223I/alpaca

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

PYTHONPATH=$(pwd):$PYTHONPATH python code/premarket_top_gainers.py "$@"
