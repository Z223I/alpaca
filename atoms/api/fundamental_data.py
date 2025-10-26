#!/usr/bin/env python3
"""
Fundamental Data Retrieval

Fetches fundamental stock data (shares outstanding, float, market cap) using:
1. Polygon.io API (primary - available through Alpaca)
2. Yahoo Finance (fallback)

Usage:
    from atoms.api.fundamental_data import get_fundamental_data

    data = get_fundamental_data("AAPL")
    print(data['shares_outstanding'])
    print(data['float_shares'])
    print(data['market_cap'])
"""

import os
import time
from typing import Dict, Optional
import requests


class FundamentalDataFetcher:
    """Fetches fundamental stock data from Polygon.io with Yahoo Finance fallback."""

    def __init__(self, polygon_api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize fundamental data fetcher.

        Args:
            polygon_api_key: Polygon.io API key (uses POLYGON_API_KEY env var if not provided)
            verbose: Enable verbose logging
        """
        self.polygon_api_key = polygon_api_key or os.getenv('POLYGON_API_KEY')
        self.verbose = verbose

        # Rate limiting
        self.last_polygon_call = 0
        self.polygon_rate_limit = 0.1  # 10 calls per second max

    def _rate_limit_polygon(self):
        """Implement rate limiting for Polygon API."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_polygon_call

        if time_since_last_call < self.polygon_rate_limit:
            sleep_time = self.polygon_rate_limit - time_since_last_call
            time.sleep(sleep_time)

        self.last_polygon_call = time.time()

    def get_from_polygon(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data from Polygon.io.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with shares_outstanding, float_shares, market_cap or None
        """
        if not self.polygon_api_key:
            if self.verbose:
                print("⚠️ No Polygon.io API key found")
            return None

        try:
            self._rate_limit_polygon()

            url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
            params = {'apiKey': self.polygon_api_key}

            if self.verbose:
                print(f"📊 Fetching fundamental data for {symbol} from Polygon.io...")

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if 'results' in data:
                    results = data['results']

                    # Extract fundamental data
                    shares_outstanding = results.get('share_class_shares_outstanding')
                    market_cap = results.get('market_cap')

                    # Polygon doesn't provide float directly, use weighted shares outstanding
                    # or shares_outstanding as approximation
                    weighted_shares = results.get('weighted_shares_outstanding')
                    float_shares = weighted_shares if weighted_shares else shares_outstanding

                    fundamental_data = {
                        'shares_outstanding': shares_outstanding,
                        'float_shares': float_shares,
                        'market_cap': market_cap,
                        'source': 'polygon'
                    }

                    if self.verbose:
                        print(f"✅ Polygon.io: {symbol} - Shares: {shares_outstanding:,} | "
                              f"Float: {float_shares:,} | Cap: ${market_cap:,}")

                    return fundamental_data
                else:
                    if self.verbose:
                        print(f"⚠️ Polygon.io: No results for {symbol}")
                    return None

            elif response.status_code == 429:
                if self.verbose:
                    print(f"⚠️ Polygon.io rate limit exceeded for {symbol}")
                return None

            else:
                if self.verbose:
                    print(f"⚠️ Polygon.io error {response.status_code} for {symbol}")
                return None

        except Exception as e:
            if self.verbose:
                print(f"❌ Polygon.io error for {symbol}: {e}")
            return None

    def get_from_yahoo(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with shares_outstanding, float_shares, market_cap or None
        """
        try:
            if self.verbose:
                print(f"📊 Fetching fundamental data for {symbol} from Yahoo Finance...")

            # Use yfinance library for cleaner access
            try:
                import yfinance as yf
            except ImportError:
                if self.verbose:
                    print("⚠️ yfinance library not installed. Install with: pip install yfinance")
                return None

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                if self.verbose:
                    print(f"⚠️ Yahoo Finance: No data for {symbol}")
                return None

            shares_outstanding = info.get('sharesOutstanding')
            float_shares = info.get('floatShares')
            market_cap = info.get('marketCap')

            # If float not available, use shares outstanding as approximation
            if not float_shares and shares_outstanding:
                float_shares = shares_outstanding

            fundamental_data = {
                'shares_outstanding': shares_outstanding,
                'float_shares': float_shares,
                'market_cap': market_cap,
                'source': 'yahoo'
            }

            if self.verbose:
                shares_str = f"{shares_outstanding:,}" if shares_outstanding else "N/A"
                float_str = f"{float_shares:,}" if float_shares else "N/A"
                cap_str = f"${market_cap:,}" if market_cap else "N/A"
                print(f"✅ Yahoo Finance: {symbol} - Shares: {shares_str} | "
                      f"Float: {float_str} | Cap: {cap_str}")

            return fundamental_data

        except Exception as e:
            if self.verbose:
                print(f"❌ Yahoo Finance error for {symbol}: {e}")
            return None

    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Get fundamental data with Polygon.io primary and Yahoo Finance fallback.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with shares_outstanding, float_shares, market_cap
            Returns None values if data not available
        """
        # Try Polygon.io first
        data = self.get_from_polygon(symbol)

        # Fallback to Yahoo Finance if Polygon fails
        if not data or not all([data.get('shares_outstanding'),
                                data.get('market_cap')]):
            if self.verbose:
                print(f"⚠️ Polygon.io data incomplete or unavailable, trying Yahoo Finance...")
            data = self.get_from_yahoo(symbol)

        # Return data or empty dict with None values
        if data:
            return {
                'shares_outstanding': data.get('shares_outstanding'),
                'float_shares': data.get('float_shares'),
                'market_cap': data.get('market_cap'),
                'source': data.get('source', 'unknown')
            }
        else:
            return {
                'shares_outstanding': None,
                'float_shares': None,
                'market_cap': None,
                'source': 'none'
            }


# Convenience function for simple usage
def get_fundamental_data(symbol: str, polygon_api_key: Optional[str] = None,
                        verbose: bool = False) -> Dict:
    """
    Get fundamental data for a symbol.

    Args:
        symbol: Stock ticker symbol
        polygon_api_key: Optional Polygon.io API key
        verbose: Enable verbose logging

    Returns:
        Dictionary with shares_outstanding, float_shares, market_cap
    """
    fetcher = FundamentalDataFetcher(polygon_api_key=polygon_api_key, verbose=verbose)
    return fetcher.get_fundamental_data(symbol)


if __name__ == "__main__":
    # Test the module
    import sys

    test_symbols = ['AAPL', 'TSLA', 'NVDA']

    print("Testing Fundamental Data Fetcher")
    print("=" * 80)

    for symbol in test_symbols:
        print(f"\n{symbol}:")
        print("-" * 80)
        data = get_fundamental_data(symbol, verbose=True)
        print(f"\nResult:")
        for key, value in data.items():
            if value and key != 'source':
                if key == 'market_cap':
                    print(f"  {key}: ${value:,.0f}")
                else:
                    print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
