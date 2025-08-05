"""
Tests for PnL (Profit and Loss) calculation functionality.
"""

import pytest
from unittest.mock import Mock, patch
from atoms.api.pnl import AlpacaDailyPnL
from code.alpaca_config import get_api_credentials


class TestAlpacaDailyPnL:
    """Test suite for AlpacaDailyPnL class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use Primary paper account credentials for testing
        self.api_key, self.secret_key, self.base_url = get_api_credentials()
        self.pnl_client = AlpacaDailyPnL(self.api_key, self.secret_key, self.base_url)

    def test_init(self):
        """Test PnL client initialization."""
        assert self.pnl_client.api_key == self.api_key
        assert self.pnl_client.secret_key == self.secret_key
        assert self.pnl_client.base_url == self.base_url
        assert self.pnl_client.headers['APCA-API-KEY-ID'] == self.api_key
        assert self.pnl_client.headers['APCA-API-SECRET-KEY'] == self.secret_key
        assert self.pnl_client.headers['Content-Type'] == 'application/json'

    @patch('atoms.api.pnl.requests.get')
    def test_get_account_info_success(self, mock_get):
        """Test successful account info retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'equity': '100000.00',
            'cash': '50000.00'
        }
        mock_get.return_value = mock_response

        result = self.pnl_client.get_account_info()

        assert result['equity'] == '100000.00'
        assert result['cash'] == '50000.00'
        mock_get.assert_called_once_with(
            f"{self.base_url}/v2/account",
            headers=self.pnl_client.headers
        )

    @patch('atoms.api.pnl.requests.get')
    def test_get_account_info_error(self, mock_get):
        """Test account info retrieval error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            self.pnl_client.get_account_info()

        assert "Error getting account info: 401 - Unauthorized" in str(exc_info.value)

    @patch('atoms.api.pnl.requests.get')
    def test_get_portfolio_history_success(self, mock_get):
        """Test successful portfolio history retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'equity': [100000.0, 100500.0, 101000.0],
            'timestamp': [1625097600, 1625098200, 1625098800]
        }
        mock_get.return_value = mock_response

        result = self.pnl_client.get_portfolio_history()

        assert len(result['equity']) == 3
        assert result['equity'][0] == 100000.0
        assert result['equity'][-1] == 101000.0
        mock_get.assert_called_once()

    @patch('atoms.api.pnl.requests.get')
    def test_get_portfolio_history_with_params(self, mock_get):
        """Test portfolio history with custom parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'equity': [], 'timestamp': []}
        mock_get.return_value = mock_response

        self.pnl_client.get_portfolio_history(
            period="1W", timeframe="1Hour", extended_hours=True
        )

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]['params']['period'] == "1W"
        assert call_args[1]['params']['timeframe'] == "1Hour"
        assert call_args[1]['params']['extended_hours'] is True

    @patch('atoms.api.pnl.requests.get')
    def test_get_positions_success(self, mock_get):
        """Test successful positions retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'symbol': 'AAPL',
                'qty': '10',
                'market_value': '1500.00'
            }
        ]
        mock_get.return_value = mock_response

        result = self.pnl_client.get_positions()

        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'
        mock_get.assert_called_once_with(
            f"{self.base_url}/v2/positions",
            headers=self.pnl_client.headers
        )

    @patch('atoms.api.pnl.AlpacaDailyPnL.get_portfolio_history')
    @patch('atoms.api.pnl.AlpacaDailyPnL.get_account_info')
    def test_calculate_daily_pnl_success(self, mock_account, mock_portfolio):
        """Test successful daily P&L calculation."""
        # Mock account info
        mock_account.return_value = {'equity': '101000.00'}
        
        # Mock portfolio history
        mock_portfolio.return_value = {
            'equity': [100000.0, 100500.0, 101000.0],
            'timestamp': [1625097600, 1625098200, 1625098800]
        }

        result = self.pnl_client.calculate_daily_pnl()

        assert result is not None
        assert result['current_equity'] == 101000.0
        assert result['starting_equity'] == 100000.0
        assert result['daily_pnl'] == 1000.0
        assert result['daily_pnl_percentage'] == 1.0
        assert 'timestamp' in result

    @patch('atoms.api.pnl.AlpacaDailyPnL.get_portfolio_history')
    @patch('atoms.api.pnl.AlpacaDailyPnL.get_account_info')
    def test_calculate_daily_pnl_insufficient_data(self, mock_account, mock_portfolio):
        """Test P&L calculation with insufficient data."""
        mock_account.return_value = {'equity': '100000.00'}
        mock_portfolio.return_value = {'equity': [100000.0], 'timestamp': []}

        with patch('builtins.print') as mock_print:
            result = self.pnl_client.calculate_daily_pnl()

        assert result is None
        mock_print.assert_called_with("Insufficient data to calculate daily P&L")

    @patch('atoms.api.pnl.AlpacaDailyPnL.get_account_info')
    def test_calculate_daily_pnl_exception_handling(self, mock_account):
        """Test P&L calculation exception handling."""
        mock_account.side_effect = Exception("API Error")

        with patch('builtins.print') as mock_print:
            result = self.pnl_client.calculate_daily_pnl()

        assert result is None
        mock_print.assert_called_with("Error calculating daily P&L: API Error")

    @patch('atoms.api.pnl.AlpacaDailyPnL.calculate_daily_pnl')
    def test_display_daily_summary_with_profit(self, mock_calculate):
        """Test display of daily summary with profit."""
        mock_calculate.return_value = {
            'starting_equity': 100000.0,
            'current_equity': 101000.0,
            'daily_pnl': 1000.0,
            'daily_pnl_percentage': 1.0,
            'timestamp': '2023-07-01T10:00:00'
        }

        with patch('builtins.print') as mock_print:
            self.pnl_client.display_daily_summary()

        # Check that profit status is displayed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        profit_status_found = any("📈 PROFIT" in call for call in print_calls)
        assert profit_status_found

    @patch('atoms.api.pnl.AlpacaDailyPnL.calculate_daily_pnl')
    def test_display_daily_summary_with_loss(self, mock_calculate):
        """Test display of daily summary with loss."""
        mock_calculate.return_value = {
            'starting_equity': 100000.0,
            'current_equity': 99000.0,
            'daily_pnl': -1000.0,
            'daily_pnl_percentage': -1.0,
            'timestamp': '2023-07-01T10:00:00'
        }

        with patch('builtins.print') as mock_print:
            self.pnl_client.display_daily_summary()

        # Check that loss status is displayed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        loss_status_found = any("📉 LOSS" in call for call in print_calls)
        assert loss_status_found

    @patch('atoms.api.pnl.AlpacaDailyPnL.calculate_daily_pnl')
    def test_display_daily_summary_break_even(self, mock_calculate):
        """Test display of daily summary at break even."""
        mock_calculate.return_value = {
            'starting_equity': 100000.0,
            'current_equity': 100000.0,
            'daily_pnl': 0.0,
            'daily_pnl_percentage': 0.0,
            'timestamp': '2023-07-01T10:00:00'
        }

        with patch('builtins.print') as mock_print:
            self.pnl_client.display_daily_summary()

        # Check that break even status is displayed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        break_even_found = any("➡️ BREAK EVEN" in call for call in print_calls)
        assert break_even_found

    @patch('atoms.api.pnl.AlpacaDailyPnL.calculate_daily_pnl')
    def test_display_daily_summary_no_data(self, mock_calculate):
        """Test display when no P&L data is available."""
        mock_calculate.return_value = None

        with patch('builtins.print') as mock_print:
            self.pnl_client.display_daily_summary()

        mock_print.assert_called_with("Unable to calculate daily P&L")

    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_SECRET_KEY': 'test_secret',
        'ALPACA_BASE_URL': 'https://test-api.alpaca.markets'
    })
    @patch('atoms.api.pnl.AlpacaDailyPnL.display_daily_summary')
    def test_create_pnl_with_env_vars(self, mock_display):
        """Test create_pnl method with environment variables."""
        client = AlpacaDailyPnL("dummy", "dummy")
        client.create_pnl()

        mock_display.assert_called_once()

    @patch.dict('os.environ', {}, clear=True)
    def test_create_pnl_missing_env_vars(self):
        """Test create_pnl method with missing environment variables."""
        client = AlpacaDailyPnL("dummy", "dummy")

        with patch('builtins.print') as mock_print:
            client.create_pnl()

        mock_print.assert_called_with(
            "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables"
        )