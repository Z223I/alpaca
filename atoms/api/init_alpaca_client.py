import os
import sys
import alpaca_trade_api as tradeapi

# Add the cgi-bin/molecules/alpaca_molecules directory to the path to import the config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cgi-bin', 'molecules', 'alpaca_molecules'))

try:
    from alpaca_config import get_api_credentials as _get_api_credentials
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    _get_api_credentials = None


def get_api_credentials(provider: str = "alpaca", account_name: str = "Bruce", account: str = "paper"):
    """
    Get API credentials from alpaca_config.py.

    Args:
        provider: Provider name (default: "alpaca")
        account_name: Account name (default: "Bruce")
        account: Environment name (default: "paper")

    Returns:
        Tuple of (api_key, secret_key, base_url)
    """
    if not CONFIG_AVAILABLE or _get_api_credentials is None:
        raise ImportError("alpaca_config module not available")

    return _get_api_credentials(provider, account_name, account)


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

    if use_config and _get_api_credentials is not None:
        try:
            # Get API credentials from config file
            key, secret, base_url = _get_api_credentials(provider, account, environment)
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