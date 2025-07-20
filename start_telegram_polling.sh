#!/bin/bash

# Telegram Polling Service Startup Script
# This script starts the automated Telegram user management service

echo "🚀 Starting Telegram Polling Service..."
echo "============================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "💡 Create .env file with BOT_TOKEN first"
    echo "   Copy from .env_telegram_sample and fill in your bot token"
    exit 1
fi

# Check if BOT_TOKEN is set
if ! grep -q "BOT_TOKEN=" .env; then
    echo "❌ Error: BOT_TOKEN not found in .env file"
    echo "💡 Add BOT_TOKEN=your_bot_token to .env file"
    exit 1
fi

# Check Python dependencies
echo "🔧 Checking dependencies..."
python3 -c "import requests; from dotenv import load_dotenv" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing Python dependencies"
    echo "💡 Install with: pip3 install requests python-dotenv"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the service
echo "✅ Dependencies OK"
echo "📊 Starting polling service..."
echo "💡 Press CTRL+C to stop"
echo ""

# Run the polling service with output to both console and log file
# Use stdbuf to disable buffering for real-time logging
stdbuf -oL -eL python3 molecules/telegram_polling.py 2>&1 | tee logs/telegram_polling_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "🛑 Telegram polling service stopped"
echo "📁 Logs saved in logs/ directory"