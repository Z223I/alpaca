import os
import alpaca_trade_api as tradeapi


def init_alpaca_client() -> tradeapi.REST:
    """
    Initialize Alpaca API client with credentials from environment variables.
    
    Returns:
        Alpaca REST API client
    """
    # Get API credentials from environment variables
    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL')
    
    # Initialize and return REST API client
    return tradeapi.REST(key, secret, base_url)