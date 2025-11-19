import os
import sys
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream

# Ensure output is not buffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# Load environment variables from .env file
# Use absolute path to .env file to ensure it's found regardless of working directory
import pathlib
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get API credentials
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

# Validate API credentials
if not api_key or not secret_key:
    error_msg = "[ERROR] API credentials not found in .env file"
    print(error_msg)
    with open('/tmp/wss_debug.log', 'a') as f:
        f.write(f"{error_msg}\n")
    sys.exit(1)

debug_msg = f"[DEBUG] API Key loaded: {api_key[:10]}..."
print(debug_msg)
with open('/tmp/wss_debug.log', 'a') as f:
    f.write(f"{debug_msg}\n")

debug_msg2 = f"[DEBUG] Secret Key loaded: {secret_key[:10]}..."
print(debug_msg2)
with open('/tmp/wss_debug.log', 'a') as f:
    f.write(f"{debug_msg2}\n")

# Initialize the stream client (requires API keys)
debug_msg3 = "[DEBUG] Initializing StockDataStream client..."
print(debug_msg3)
with open('/tmp/wss_debug.log', 'a') as f:
    f.write(f"{debug_msg3}\n")

try:
    wss_client = StockDataStream(api_key, secret_key)
    debug_msg4 = "[DEBUG] StockDataStream client initialized successfully"
    print(debug_msg4)
    with open('/tmp/wss_debug.log', 'a') as f:
        f.write(f"{debug_msg4}\n")
except Exception as e:
    error_msg = f"[ERROR] Failed to initialize StockDataStream: {e}"
    print(error_msg)
    with open('/tmp/wss_debug.log', 'a') as f:
        f.write(f"{error_msg}\n")
    sys.exit(1)

# Define an async handler for quote data
async def quote_data_handler(data):
    quote_msg = f"[QUOTE DATA] {data}"
    print(quote_msg)
    with open('/tmp/wss_debug.log', 'a') as f:
        f.write(f"{quote_msg}\n")

# Subscribe to quotes for specific symbols
debug_msg5 = "[DEBUG] Subscribing to AAPL quotes..."
print(debug_msg5)
with open('/tmp/wss_debug.log', 'a') as f:
    f.write(f"{debug_msg5}\n")

wss_client.subscribe_quotes(quote_data_handler, "AAPL")

debug_msg6 = "[DEBUG] Subscription registered"
print(debug_msg6)
with open('/tmp/wss_debug.log', 'a') as f:
    f.write(f"{debug_msg6}\n")

# Start receiving data
debug_msg7 = "[DEBUG] Starting WebSocket stream..."
print(debug_msg7)
with open('/tmp/wss_debug.log', 'a') as f:
    f.write(f"{debug_msg7}\n")

wss_client.run()
