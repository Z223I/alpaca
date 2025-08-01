import os
import sys
import alpaca_trade_api as tradeapi

# Add the code directory to the path to import the config
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'code'))

try:
    from alpaca_config import get_api_credentials
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def init_alpaca_client(provider: str = "alpaca", account: str = "Bruce", environment: str = "paper") -> tradeapi.REST:
    """
    Initialize Alpaca API client with credentials from config file.
    Falls back to environment variables if config is not available.

    Args:
        provider: Provider name (default: "alpaca")
        account: Account name (default: "Bruce")
        environment: Environment name (default: "paper")

    Returns:
        Alpaca REST API client
    """
    use_config = CONFIG_AVAILABLE
    key = None
    secret = None
    base_url = None

    if use_config:
        try:
            # Get API credentials from config file
            key, secret, base_url = get_api_credentials(provider, account, environment)
        except Exception as e:
            print(f"Warning: Could not load config file ({e}), falling back to environment variables")
            use_config = False

    if not use_config:
        # Fallback to environment variables
        key = os.getenv('ALPACA_API_KEY')
        secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL')

    # Initialize and return REST API client
    return tradeapi.REST(key, secret, base_url)