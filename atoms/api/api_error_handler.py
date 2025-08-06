"""
Common API Error Handler for Alpaca Trading System

This module provides centralized error handling for all Alpaca API interactions,
including detailed debugging information for 403 Forbidden errors.
"""

import sys
import os
from typing import Any, Callable, Optional

# Add the code directory to the path to import the config
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'code'))

try:
    from alpaca_config import get_api_credentials, get_current_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def display_403_debug_info(api_key: str, secret_key: str, base_url: str, 
                          account_name: str = None, environment: str = None) -> None:
    """
    Display detailed debugging information for 403 Forbidden errors.
    
    Args:
        api_key: The API key being used
        secret_key: The secret key being used  
        base_url: The base URL being used
        account_name: Account name from command line args
        environment: Environment from command line args
    """
    print("\n" + "="*60)
    print("🔒 403 FORBIDDEN ERROR - Account Configuration Check")
    print("="*60)
    print(f"API Key:      {api_key}")
    print(f"Secret Key:   {secret_key[:8]}...{secret_key[-4:] if len(secret_key) > 12 else '***'}")
    print(f"Base URL:     {base_url}")
    
    if account_name and environment:
        print(f"Account:      --account-name {account_name} --account {environment}")
    
    print("\nPlease verify:")
    print("• API Key matches your account configuration")
    print("• Secret Key is correct and not expired")
    print("• Base URL matches your account type (paper vs live)")
    print("• Account has proper permissions")
    print("• Configuration in code/alpaca_config.py is correct")
    print("="*60)


def handle_alpaca_api_error(func: Callable, *args, account_name: str = None, 
                           environment: str = None, **kwargs) -> Any:
    """
    Wrapper function to handle errors from alpaca_trade_api calls with enhanced debugging.
    
    Args:
        func: The API function to call
        *args: Arguments to pass to the function
        account_name: Account name for debugging context
        environment: Environment for debugging context  
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result from the API function call
        
    Raises:
        Exception: Re-raises the original exception after displaying debug info for 403 errors
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_message = str(e)
        
        # Check for 403 errors in alpaca_trade_api exceptions
        if "403" in error_message or "forbidden" in error_message.lower():
            # Try to extract credentials from the API client (first arg is usually the client)
            api_key = "Not found"
            secret_key = "Not found"
            base_url = "Not found"
            
            # First try to get credentials from config if available
            if CONFIG_AVAILABLE and account_name and environment:
                try:
                    api_key, secret_key, base_url = get_api_credentials("alpaca", account_name, environment)
                except Exception:
                    pass
            
            # If config failed, try to extract from client object via the bound method
            if api_key == "Not found":
                try:
                    # Check if func is a bound method and extract the client
                    if hasattr(func, '__self__'):
                        client = func.__self__
                        if hasattr(client, '_key_id') and hasattr(client, '_secret_key') and hasattr(client, '_base_url'):
                            api_key = client._key_id
                            secret_key = client._secret_key  
                            base_url = str(client._base_url)
                except Exception:
                    pass
            
            # Final fallback to environment variables
            if api_key == "Not found":
                api_key = os.getenv('ALPACA_API_KEY', 'Not found')
                secret_key = os.getenv('ALPACA_SECRET_KEY', 'Not found')  
                base_url = os.getenv('ALPACA_BASE_URL', 'Not found')
            
            display_403_debug_info(api_key, secret_key, base_url, account_name, environment)
            
            # For 403 errors, return None instead of crashing to allow graceful handling
            print("\n❌ Unable to access account due to authentication/authorization error.")
            print("   Please check your API credentials and account permissions.")
            return None
        
        # Re-raise non-403 exceptions
        raise


def handle_requests_api_error(response: Any, api_key: str, secret_key: str, base_url: str,
                             account_name: str = None, environment: str = None) -> None:
    """
    Handle HTTP response errors from requests library calls with enhanced debugging.
    
    Args:
        response: The HTTP response object
        api_key: The API key being used
        secret_key: The secret key being used
        base_url: The base URL being used
        account_name: Account name for debugging context
        environment: Environment for debugging context
        
    Raises:
        Exception: Raises an exception with the response error details
    """
    if response.status_code == 403:
        display_403_debug_info(api_key, secret_key, base_url, account_name, environment)
    
    raise Exception(f"API request failed: {response.status_code} - {response.text}")