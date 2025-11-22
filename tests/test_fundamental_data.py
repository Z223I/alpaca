"""
Unit tests for Fundamental Data Fetcher

Tests the fundamental data retrieval functionality using yfinance.
This includes testing both the standalone FundamentalDataFetcher class
and the integration with MomentumAlertsSystem.

Recent Issue: Company fundamentals stopped working - these tests help identify
whether the issue is with yfinance API, data structure changes, or our code.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict
import sys
import os

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the fetcher from the cgi-bin location
cgi_bin_dir = os.path.join(project_root, 'cgi-bin')
sys.path.insert(0, cgi_bin_dir)

# Import using direct module path
import importlib.util
cgi_api_atoms_dir = os.path.join(cgi_bin_dir, 'api', 'atoms', 'alpaca_api')
yfinance_fetcher_path = os.path.join(cgi_api_atoms_dir, 'fundamental_data.py')
spec = importlib.util.spec_from_file_location("yfinance_fundamental_data", yfinance_fetcher_path)
yfinance_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yfinance_module)
FundamentalDataFetcher = yfinance_module.FundamentalDataFetcher


class TestFundamentalDataFetcherUnit:
    """Unit tests for FundamentalDataFetcher class with mocked yfinance."""

    def setup_method(self):
        """Setup test fixtures."""
        self.fetcher = FundamentalDataFetcher(verbose=False)

    def test_fetcher_initialization(self):
        """Test fetcher initialization."""
        fetcher = FundamentalDataFetcher(verbose=True)
        assert fetcher.verbose is True

        fetcher = FundamentalDataFetcher(verbose=False)
        assert fetcher.verbose is False

    def test_get_fundamental_data_success(self):
        """Test successful fundamental data retrieval."""
        # Mock yfinance module and Ticker class
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': 15000000000,
            'floatShares': 14500000000,
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        assert result is not None
        assert result['shares_outstanding'] == 15000000000
        assert result['float_shares'] == 14500000000
        assert result['market_cap'] == 3500000000000
        assert result['source'] == 'yahoo'

        # Verify yfinance was called correctly
        mock_yf.Ticker.assert_called_once_with('AAPL')

    def test_get_fundamental_data_missing_float(self):
        """Test handling of missing float shares (uses shares outstanding as fallback)."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': 15000000000,
            'floatShares': None,  # Missing float shares
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        assert result is not None
        assert result['shares_outstanding'] == 15000000000
        assert result['float_shares'] == 15000000000  # Should fallback to shares_outstanding
        assert result['market_cap'] == 3500000000000
        assert result['source'] == 'yahoo'

    def test_get_fundamental_data_missing_all_data(self):
        """Test handling of missing data."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': None,
            'floatShares': None,
            'marketCap': None
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('INVALID')

        assert result is not None
        assert result['shares_outstanding'] is None
        assert result['float_shares'] is None
        assert result['market_cap'] is None
        assert result['source'] == 'yahoo'

    def test_get_fundamental_data_empty_info(self):
        """Test handling of empty info dictionary."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('INVALID')

        assert result is not None
        assert result['shares_outstanding'] is None
        assert result['float_shares'] is None
        assert result['market_cap'] is None
        # Empty info dict is falsy, so it returns 'yahoo-no-data'
        assert result['source'] == 'yahoo-no-data'

    def test_get_fundamental_data_no_info(self):
        """Test handling of None info."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = None
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('INVALID')

        assert result is not None
        assert result['shares_outstanding'] is None
        assert result['float_shares'] is None
        assert result['market_cap'] is None
        assert result['source'] == 'yahoo-no-data'

    def test_get_fundamental_data_exception(self):
        """Test handling of yfinance exceptions."""
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("Network error")

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        assert result is not None
        assert result['shares_outstanding'] is None
        assert result['float_shares'] is None
        assert result['market_cap'] is None
        assert result['source'] == 'yahoo-error'

    def test_get_fundamental_data_yfinance_not_installed(self):
        """Test handling when yfinance is not installed."""
        # Temporarily hide yfinance module
        with patch.dict('sys.modules', {'yfinance': None}):
            # This will cause ImportError when trying to import yfinance
            result = self.fetcher.get_fundamental_data('AAPL')

            assert result is not None
            assert result['shares_outstanding'] is None
            assert result['float_shares'] is None
            assert result['market_cap'] is None
            assert result['source'] == 'error-yfinance-not-installed'

    def test_verbose_output(self):
        """Test that verbose mode produces expected output."""
        # Create verbose fetcher
        verbose_fetcher = FundamentalDataFetcher(verbose=True)

        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': 15000000000,
            'floatShares': 14500000000,
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        # Capture print output
        with patch('builtins.print') as mock_print:
            with patch.dict('sys.modules', {'yfinance': mock_yf}):
                result = verbose_fetcher.get_fundamental_data('AAPL')

            # Verify print was called (verbose output)
            assert mock_print.call_count >= 1

            # Check that one of the calls contains the symbol
            calls = [str(call) for call in mock_print.call_args_list]
            assert any('AAPL' in call for call in calls)


class TestFundamentalDataFetcherIntegration:
    """Integration tests with real yfinance API.

    These tests make real API calls to verify the integration works.
    They may be slower and could fail due to network issues.
    Use pytest -m "not integration" to skip these tests.
    """

    pytestmark = pytest.mark.integration

    def setup_method(self):
        """Setup test fixtures."""
        self.fetcher = FundamentalDataFetcher(verbose=False)

    @pytest.mark.slow
    def test_get_fundamental_data_real_api_aapl(self):
        """Test real API call for AAPL."""
        result = self.fetcher.get_fundamental_data('AAPL')

        assert result is not None
        assert result['source'] in ['yahoo', 'yahoo-no-data', 'yahoo-error']

        # If data was successfully retrieved
        if result['source'] == 'yahoo':
            # AAPL should have market cap (it's a large company)
            assert result['market_cap'] is not None
            assert result['market_cap'] > 0

            # AAPL should have shares outstanding
            assert result['shares_outstanding'] is not None
            assert result['shares_outstanding'] > 0

            # Float shares might be None, but if present should be positive
            if result['float_shares'] is not None:
                assert result['float_shares'] > 0

    @pytest.mark.slow
    def test_get_fundamental_data_real_api_invalid_symbol(self):
        """Test real API call for invalid symbol."""
        result = self.fetcher.get_fundamental_data('INVALID_SYMBOL_XYZ123')

        assert result is not None
        # Invalid symbols should return None values or error source
        assert result['source'] in ['yahoo', 'yahoo-no-data', 'yahoo-error']

    @pytest.mark.slow
    def test_get_fundamental_data_real_api_multiple_symbols(self):
        """Test real API calls for multiple symbols."""
        symbols = ['AAPL', 'TSLA', 'NVDA']

        for symbol in symbols:
            result = self.fetcher.get_fundamental_data(symbol)
            assert result is not None
            assert 'source' in result
            assert 'market_cap' in result
            assert 'shares_outstanding' in result
            assert 'float_shares' in result

    @pytest.mark.slow
    def test_get_fundamental_data_real_api_mndr(self):
        """Test real API call for MNDR - a low float stock that was reported as failing."""
        result = self.fetcher.get_fundamental_data('MNDR')

        assert result is not None
        print(f"\nMNDR Result: {result}")

        # MNDR should return data (even if it's a small cap stock)
        assert result['source'] in ['yahoo', 'yahoo-no-data', 'yahoo-error']

        # If data was retrieved successfully
        if result['source'] == 'yahoo':
            # MNDR is a very small cap stock
            assert result['shares_outstanding'] is not None or result['float_shares'] is not None or result['market_cap'] is not None

            # Log the values for debugging
            print(f"  Shares Outstanding: {result['shares_outstanding']}")
            print(f"  Float Shares: {result['float_shares']}")
            print(f"  Market Cap: {result['market_cap']}")


class TestFundamentalDataStructure:
    """Test the structure and types of returned data."""

    def setup_method(self):
        """Setup test fixtures."""
        self.fetcher = FundamentalDataFetcher(verbose=False)

    def test_return_data_structure(self):
        """Test that returned data has correct structure."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': 15000000000,
            'floatShares': 14500000000,
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        # Check that all required keys are present
        assert 'float_shares' in result
        assert 'shares_outstanding' in result
        assert 'market_cap' in result
        assert 'source' in result

    def test_return_data_types(self):
        """Test that returned data has correct types."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': 15000000000,
            'floatShares': 14500000000,
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        # Check types
        assert isinstance(result, dict)
        assert isinstance(result['shares_outstanding'], (int, type(None)))
        assert isinstance(result['float_shares'], (int, type(None)))
        assert isinstance(result['market_cap'], (int, type(None)))
        assert isinstance(result['source'], str)

    def test_data_consistency(self):
        """Test that data is internally consistent."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': 15000000000,
            'floatShares': 14500000000,
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        # Float should be less than or equal to shares outstanding
        if result['float_shares'] and result['shares_outstanding']:
            assert result['float_shares'] <= result['shares_outstanding']


class TestYFinanceAPIChanges:
    """Test potential API changes or data structure issues.

    These tests help diagnose issues when fundamentals stop working,
    which often happens when Yahoo Finance changes their API or data structure.
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.fetcher = FundamentalDataFetcher(verbose=False)

    def test_info_dict_keys_changed(self):
        """Test handling of changed key names in info dict."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'shares_outstanding': 15000000000,  # Different key name
            'float_shares': 14500000000,  # Different key name
            'market_cap': 3500000000000  # Different key name
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        # With current implementation, this should return None values
        # because keys don't match
        assert result['shares_outstanding'] is None
        assert result['float_shares'] is None
        assert result['market_cap'] is None

    def test_info_dict_nested_structure(self):
        """Test handling of nested data structures."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'defaultKeyStatistics': {
                'sharesOutstanding': 15000000000,
                'floatShares': 14500000000
            },
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        # With current implementation, nested values won't be found
        assert result['shares_outstanding'] is None
        assert result['float_shares'] is None
        assert result['market_cap'] == 3500000000000  # This is at top level

    def test_info_returns_different_types(self):
        """Test handling of different data types in response."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': '15000000000',  # String instead of int
            'floatShares': 14500000000,
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = self.fetcher.get_fundamental_data('AAPL')

        # Current implementation should handle strings
        assert result['shares_outstanding'] == '15000000000'
        assert result['float_shares'] == 14500000000

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_api_response_structure(self):
        """Test the actual structure of real API responses.

        This test helps diagnose what changed when fundamentals stop working.
        """
        try:
            import yfinance as yf

            # Get real data
            ticker = yf.Ticker('AAPL')
            info = ticker.info

            # Print debug info (will show in pytest output with -v)
            print(f"\nReal API response keys: {list(info.keys())}")
            print(f"sharesOutstanding: {info.get('sharesOutstanding')}")
            print(f"floatShares: {info.get('floatShares')}")
            print(f"marketCap: {info.get('marketCap')}")

            # Verify expected keys exist
            # This test will FAIL if Yahoo Finance changed their API
            assert 'sharesOutstanding' in info or 'shares_outstanding' in info, \
                "Shares outstanding key not found - API may have changed"
            assert 'marketCap' in info or 'market_cap' in info, \
                "Market cap key not found - API may have changed"

        except ImportError:
            pytest.skip("yfinance not installed")
        except Exception as e:
            pytest.fail(f"Real API test failed: {e}")


class TestConvenienceFunction:
    """Test the convenience function get_fundamental_data()."""

    def test_convenience_function(self):
        """Test the module-level convenience function."""
        mock_yf = MagicMock()
        mock_ticker = Mock()
        mock_ticker.info = {
            'sharesOutstanding': 15000000000,
            'floatShares': 14500000000,
            'marketCap': 3500000000000
        }
        mock_yf.Ticker.return_value = mock_ticker

        # Import the convenience function
        get_fundamental_data = yfinance_module.get_fundamental_data

        with patch.dict('sys.modules', {'yfinance': mock_yf}):
            result = get_fundamental_data('AAPL', verbose=False)

        assert result is not None
        assert result['shares_outstanding'] == 15000000000
        assert result['source'] == 'yahoo'


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
