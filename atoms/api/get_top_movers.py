"""
Get top market movers (gainers and losers) from Alpaca screener API.

This atom provides functionality to fetch top gainers and losers
from the Alpaca screener endpoint for stocks or crypto markets.
"""

import requests
from typing import Any, Dict, Optional
from .api_error_handler import handle_requests_api_error


def get_top_movers(api_client: Any, market_type: str = 'stocks',
                   account_name: str = None, environment: str = None) -> Optional[Dict]:
    """
    Get top market movers (gainers and losers) from Alpaca screener API.

    Args:
        api_client: Alpaca API client instance
        market_type: Market type - 'stocks' or 'crypto' (default: 'stocks')
        account_name: Account name for debugging context (optional)
        environment: Environment for debugging context (optional)

    Returns:
        Dictionary containing 'gainers' and 'losers' lists, or None on error
        Example:
        {
            'gainers': [
                {'symbol': 'AAPL', 'price': 150.0, 'change': 5.0, 'percent_change': 3.45},
                ...
            ],
            'losers': [
                {'symbol': 'TSLA', 'price': 200.0, 'change': -10.0, 'percent_change': -4.76},
                ...
            ],
            'market_type': 'stocks',
            'last_updated': '2025-10-21T16:32:00Z'
        }
    """
    try:
        # Get API credentials from client
        api_key = api_client._key_id
        api_secret = api_client._secret_key

        # Build the endpoint URL
        base_url = "https://data.alpaca.markets"
        endpoint = f"/v1beta1/screener/{market_type}/movers"
        url = f"{base_url}{endpoint}"

        # Set up headers for authentication
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }

        # Make the API request
        response = requests.get(url, headers=headers, timeout=10)

        # Handle errors using the centralized error handler
        if response.status_code != 200:
            handle_requests_api_error(response, api_key, api_secret, base_url,
                                      account_name, environment)
            return None

        # Parse and return the JSON response
        return response.json()

    except requests.exceptions.Timeout:
        print("✗ Request timed out while fetching top movers")
        return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error while fetching top movers: {e}")
        return None
    except Exception as e:
        print(f"✗ Error fetching top movers: {e}")
        return None
