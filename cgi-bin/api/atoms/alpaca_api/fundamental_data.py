#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
"""
Fundamental Data Retrieval Using yfinance

This module fetches fundamental stock data (shares outstanding, float, market cap)
using Yahoo Finance via the yfinance library.

NOTE: alpaca-py does NOT provide fundamental data. The alpaca.trading.models.Asset
class only contains trading-related fields (tradable, marginable, shortable, etc.)

This module uses yfinance to retrieve:
- Float shares
- Shares outstanding
- Market cap

Usage:
    from cgi_bin.api.atoms.alpaca_api.fundamental_data import FundamentalDataFetcher

    fetcher = FundamentalDataFetcher(verbose=True)
    data = fetcher.get_fundamental_data("AAPL")
    # Returns: {'float_shares': 15204934656, 'shares_outstanding': 15204934656, 'market_cap': 3500000000000}
"""

import os
from typing import Dict, Optional


class FundamentalDataFetcher:
    """Fetches fundamental stock data from Yahoo Finance using yfinance."""

    def __init__(self, verbose: bool = False):
        """
        Initialize fundamental data fetcher.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Get fundamental data from Yahoo Finance using yfinance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with fundamental data:
            - float_shares: Number of shares in the float
            - shares_outstanding: Total shares outstanding
            - market_cap: Market capitalization
            - source: 'yahoo'
        """
        try:
            if self.verbose:
                print(f"📊 Fetching fundamental data for {symbol} from Yahoo Finance...")

            # Use yfinance library
            try:
                import yfinance as yf
            except ImportError:
                if self.verbose:
                    print("⚠️ yfinance library not installed. Install with: pip install yfinance")
                return {
                    'float_shares': None,
                    'shares_outstanding': None,
                    'market_cap': None,
                    'source': 'error-yfinance-not-installed'
                }

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                if self.verbose:
                    print(f"⚠️ Yahoo Finance: No data for {symbol}")
                return {
                    'float_shares': None,
                    'shares_outstanding': None,
                    'market_cap': None,
                    'source': 'yahoo-no-data'
                }

            shares_outstanding = info.get('sharesOutstanding')
            float_shares = info.get('floatShares')
            market_cap = info.get('marketCap')

            # If float not available, use shares outstanding as approximation
            if not float_shares and shares_outstanding:
                float_shares = shares_outstanding

            if self.verbose:
                shares_str = f"{shares_outstanding:,}" if shares_outstanding else "N/A"
                float_str = f"{float_shares:,}" if float_shares else "N/A"
                cap_str = f"${market_cap:,}" if market_cap else "N/A"
                print(f"✅ Yahoo Finance: {symbol} - Shares: {shares_str} | "
                      f"Float: {float_str} | Cap: {cap_str}")

            return {
                'float_shares': float_shares,
                'shares_outstanding': shares_outstanding,
                'market_cap': market_cap,
                'source': 'yahoo'
            }

        except Exception as e:
            if self.verbose:
                print(f"❌ Yahoo Finance error for {symbol}: {e}")
            return {
                'float_shares': None,
                'shares_outstanding': None,
                'market_cap': None,
                'source': 'yahoo-error'
            }


# Convenience function for simple usage
def get_fundamental_data(symbol: str, verbose: bool = False) -> Dict:
    """
    Get fundamental data for a symbol using Yahoo Finance.

    Args:
        symbol: Stock ticker symbol
        verbose: Enable verbose logging

    Returns:
        Dictionary with fundamental data:
        - float_shares: Number of shares in the float
        - shares_outstanding: Total shares outstanding
        - market_cap: Market capitalization
        - source: 'yahoo'
    """
    fetcher = FundamentalDataFetcher(verbose=verbose)
    return fetcher.get_fundamental_data(symbol)


if __name__ == "__main__":
    # Test the module
    import sys

    test_symbols = ['AAPL', 'TSLA', 'NVDA']

    print("=" * 80)
    print("Fundamental Data Fetcher Test (using yfinance)")
    print("=" * 80)
    print()

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
