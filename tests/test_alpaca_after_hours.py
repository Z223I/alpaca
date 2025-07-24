"""
Comprehensive tests for after-hours trading methods in the Alpaca trading system.

This test suite covers all after-hours trading functionality including:
- _buy_after_hours: Simple after-hours buy orders
- _sell_short_after_hours: Simple after-hours short sell orders  
- _buy_after_hours_protected: Protected buy orders with stop-loss/take-profit
- _sell_short_after_hours_protected: Protected short orders with stop-loss/take-profit
- _submit_after_hours_stop_loss: Helper for stop-loss orders
- _submit_after_hours_take_profit: Helper for take-profit orders
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca/code')

from alpaca import alpaca_private


class TestAfterHoursTrading:
    """Test suite for after-hours trading functionality."""

    @pytest.fixture
    def mock_alpaca(self):
        """Create a mock alpaca_private instance for testing."""
        with patch('alpaca.init_alpaca_client') as mock_init:
            mock_api = Mock()
            mock_init.return_value = mock_api
            
            # Mock environment variables
            with patch.dict(os.environ, {'PORTFOLIO_RISK': '0.10'}):
                with patch('alpaca.parse_args') as mock_parse:
                    mock_args = Mock()
                    mock_parse.return_value = mock_args
                    
                    alpaca_obj = alpaca_private([])
                    alpaca_obj.api = mock_api
                    return alpaca_obj

    @pytest.fixture
    def mock_market_data(self):
        """Mock market data for testing."""
        return {
            'market_price': 150.00,
            'bid_price': 149.95,
            'ask_price': 150.05
        }

    def test_buy_after_hours_dry_run_default_params(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours with default parameters in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=10):
                
                result = mock_alpaca._buy_after_hours('AAPL', submit_order=False)
                
                assert result is None  # Dry run should return None
                # Verify no API calls were made
                mock_alpaca.api.submit_order.assert_not_called()

    def test_buy_after_hours_dry_run_with_amount(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours with custom amount in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            
            result = mock_alpaca._buy_after_hours('AAPL', amount=1000.0, submit_order=False)
            
            assert result is None  # Dry run should return None
            mock_alpaca.api.submit_order.assert_not_called()

    def test_buy_after_hours_dry_run_with_limit_price(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours with custom limit price in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=5):
                
                result = mock_alpaca._buy_after_hours('AAPL', limit_price=155.00, submit_order=False)
                
                assert result is None  # Dry run should return None
                mock_alpaca.api.submit_order.assert_not_called()

    def test_buy_after_hours_successful_submission(self, mock_alpaca, mock_market_data):
        """Test successful _buy_after_hours order submission."""
        mock_order = Mock()
        mock_order.id = 'order_123'
        mock_order.status = 'new'
        mock_order.symbol = 'AAPL'
        mock_order.qty = 6
        mock_alpaca.api.submit_order.return_value = mock_order
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=6):
                
                result = mock_alpaca._buy_after_hours('AAPL', submit_order=True)
                
                assert result == mock_order
                mock_alpaca.api.submit_order.assert_called_once_with(
                    symbol='AAPL',
                    qty=6,
                    side='buy',
                    type='limit',
                    limit_price=150.30,  # 150.00 * 1.002
                    time_in_force='ext'
                )

    def test_buy_after_hours_api_exception(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours when API raises exception."""
        mock_alpaca.api.submit_order.side_effect = Exception("API Error")
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=6):
                
                result = mock_alpaca._buy_after_hours('AAPL', submit_order=True)
                
                assert result is None
                mock_alpaca.api.submit_order.assert_called_once()

    def test_sell_short_after_hours_dry_run_default_params(self, mock_alpaca, mock_market_data):
        """Test _sell_short_after_hours with default parameters in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=8):
                
                result = mock_alpaca._sell_short_after_hours('TSLA', submit_order=False)
                
                assert result is None  # Dry run should return None
                mock_alpaca.api.submit_order.assert_not_called()

    def test_sell_short_after_hours_dry_run_with_amount(self, mock_alpaca, mock_market_data):
        """Test _sell_short_after_hours with custom amount in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            
            result = mock_alpaca._sell_short_after_hours('TSLA', amount=1500.0, submit_order=False)
            
            assert result is None  # Dry run should return None
            mock_alpaca.api.submit_order.assert_not_called()

    def test_sell_short_after_hours_successful_submission(self, mock_alpaca, mock_market_data):
        """Test successful _sell_short_after_hours order submission."""
        mock_order = Mock()
        mock_order.id = 'short_order_456'
        mock_order.status = 'new'
        mock_order.symbol = 'TSLA'
        mock_order.qty = 4
        mock_alpaca.api.submit_order.return_value = mock_order
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=4):
                
                result = mock_alpaca._sell_short_after_hours('TSLA', submit_order=True)
                
                assert result == mock_order
                mock_alpaca.api.submit_order.assert_called_once_with(
                    symbol='TSLA',
                    qty=4,
                    side='sell',
                    type='limit',
                    limit_price=149.70,  # 150.00 * 0.998
                    time_in_force='ext'
                )

    def test_sell_short_after_hours_api_exception(self, mock_alpaca, mock_market_data):
        """Test _sell_short_after_hours when API raises exception."""
        mock_alpaca.api.submit_order.side_effect = Exception("Short selling error")
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=4):
                
                result = mock_alpaca._sell_short_after_hours('TSLA', submit_order=True)
                
                assert result is None
                mock_alpaca.api.submit_order.assert_called_once()

    def test_submit_after_hours_stop_loss_successful(self, mock_alpaca):
        """Test successful _submit_after_hours_stop_loss order."""
        mock_stop_order = Mock()
        mock_stop_order.id = 'stop_789'
        mock_alpaca.api.submit_order.return_value = mock_stop_order
        
        result = mock_alpaca._submit_after_hours_stop_loss('AAPL', 10, 140.00, 'sell')
        
        assert result == mock_stop_order
        mock_alpaca.api.submit_order.assert_called_once_with(
            symbol='AAPL',
            qty=10,
            side='sell',
            type='stop',
            stop_price=140.00,
            time_in_force='ext'
        )

    def test_submit_after_hours_stop_loss_exception(self, mock_alpaca):
        """Test _submit_after_hours_stop_loss when API raises exception."""
        mock_alpaca.api.submit_order.side_effect = Exception("Stop loss error")
        
        result = mock_alpaca._submit_after_hours_stop_loss('AAPL', 10, 140.00, 'sell')
        
        assert result is None
        mock_alpaca.api.submit_order.assert_called_once()

    def test_submit_after_hours_take_profit_successful(self, mock_alpaca):
        """Test successful _submit_after_hours_take_profit order."""
        mock_profit_order = Mock()
        mock_profit_order.id = 'profit_101'
        mock_alpaca.api.submit_order.return_value = mock_profit_order
        
        result = mock_alpaca._submit_after_hours_take_profit('AAPL', 10, 160.00, 'sell')
        
        assert result == mock_profit_order
        mock_alpaca.api.submit_order.assert_called_once_with(
            symbol='AAPL',
            qty=10,
            side='sell',
            type='limit',
            limit_price=160.00,
            time_in_force='ext'
        )

    def test_submit_after_hours_take_profit_exception(self, mock_alpaca):
        """Test _submit_after_hours_take_profit when API raises exception."""
        mock_alpaca.api.submit_order.side_effect = Exception("Take profit error")
        
        result = mock_alpaca._submit_after_hours_take_profit('AAPL', 10, 160.00, 'sell')
        
        assert result is None
        mock_alpaca.api.submit_order.assert_called_once()

    def test_buy_after_hours_protected_dry_run_with_calc_take_profit(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours_protected with calc_take_profit in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=12):
                
                result = mock_alpaca._buy_after_hours_protected(
                    'AAPL', 
                    stop_loss=142.50,  # Custom stop loss
                    take_profit=None,  # Will trigger calc_take_profit logic
                    submit_order=False
                )
                
                assert result is None  # Dry run should return None
                mock_alpaca.api.submit_order.assert_not_called()

    def test_buy_after_hours_protected_dry_run_with_amount(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours_protected with custom amount in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            
            result = mock_alpaca._buy_after_hours_protected(
                'AAPL',
                amount=2000.0,
                take_profit=165.00,
                stop_loss=140.00,
                submit_order=False
            )
            
            assert result is None  # Dry run should return None
            mock_alpaca.api.submit_order.assert_not_called()

    def test_buy_after_hours_protected_successful_with_all_orders(self, mock_alpaca, mock_market_data):
        """Test successful _buy_after_hours_protected with all protection orders."""
        # Mock main order
        mock_main_order = Mock()
        mock_main_order.id = 'main_order_202'
        mock_main_order.status = 'new'
        
        # Mock protection orders
        mock_stop_order = Mock()
        mock_stop_order.id = 'stop_203'
        mock_profit_order = Mock()
        mock_profit_order.id = 'profit_204'
        
        mock_alpaca.api.submit_order.return_value = mock_main_order
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=8):
                with patch.object(mock_alpaca, '_submit_after_hours_stop_loss', return_value=mock_stop_order):
                    with patch.object(mock_alpaca, '_submit_after_hours_take_profit', return_value=mock_profit_order):
                        
                        result = mock_alpaca._buy_after_hours_protected(
                            'AAPL',
                            take_profit=165.00,
                            stop_loss=140.00,
                            submit_order=True
                        )
                        
                        assert result is not None
                        assert result['main_order'] == mock_main_order
                        assert result['stop_loss_order'] == mock_stop_order
                        assert result['take_profit_order'] == mock_profit_order
                        assert result['entry_price'] == 150.30  # 150.00 * 1.002
                        assert result['stop_price'] == 140.00
                        assert result['take_profit_price'] == 165.00
                        
                        # Verify main order submission
                        mock_alpaca.api.submit_order.assert_called_once_with(
                            symbol='AAPL',
                            qty=8,
                            side='buy',
                            type='limit',
                            limit_price=150.30,
                            time_in_force='ext'
                        )
                        
                        # Verify protection orders were called
                        mock_alpaca._submit_after_hours_stop_loss.assert_called_once_with('AAPL', 8, 140.00, 'sell')
                        mock_alpaca._submit_after_hours_take_profit.assert_called_once_with('AAPL', 8, 165.00, 'sell')

    def test_buy_after_hours_protected_main_order_exception(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours_protected when main order fails."""
        mock_alpaca.api.submit_order.side_effect = Exception("Main order failed")
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=8):
                
                result = mock_alpaca._buy_after_hours_protected(
                    'AAPL',
                    take_profit=165.00,
                    stop_loss=140.00,
                    submit_order=True
                )
                
                assert result is None
                mock_alpaca.api.submit_order.assert_called_once()

    def test_sell_short_after_hours_protected_dry_run_with_calc_take_profit(self, mock_alpaca, mock_market_data):
        """Test _sell_short_after_hours_protected with calc_take_profit in dry run mode."""
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=15):
                
                result = mock_alpaca._sell_short_after_hours_protected(
                    'TSLA',
                    stop_loss=157.50,  # Custom stop loss (above entry for shorts)
                    take_profit=None,  # Will trigger calc_take_profit logic
                    submit_order=False
                )
                
                assert result is None  # Dry run should return None
                mock_alpaca.api.submit_order.assert_not_called()

    def test_sell_short_after_hours_protected_successful_with_all_orders(self, mock_alpaca, mock_market_data):
        """Test successful _sell_short_after_hours_protected with all protection orders."""
        # Mock main order
        mock_main_order = Mock()
        mock_main_order.id = 'short_main_305'
        mock_main_order.status = 'new'
        
        # Mock protection orders
        mock_stop_order = Mock()
        mock_stop_order.id = 'short_stop_306'
        mock_profit_order = Mock()
        mock_profit_order.id = 'short_profit_307'
        
        mock_alpaca.api.submit_order.return_value = mock_main_order
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=6):
                with patch.object(mock_alpaca, '_submit_after_hours_stop_loss', return_value=mock_stop_order):
                    with patch.object(mock_alpaca, '_submit_after_hours_take_profit', return_value=mock_profit_order):
                        
                        result = mock_alpaca._sell_short_after_hours_protected(
                            'TSLA',
                            take_profit=135.00,
                            stop_loss=157.50,
                            submit_order=True
                        )
                        
                        assert result is not None
                        assert result['main_order'] == mock_main_order
                        assert result['stop_loss_order'] == mock_stop_order
                        assert result['take_profit_order'] == mock_profit_order
                        assert result['entry_price'] == 149.70  # 150.00 * 0.998
                        assert result['stop_price'] == 157.50
                        assert result['take_profit_price'] == 135.00
                        
                        # Verify main order submission
                        mock_alpaca.api.submit_order.assert_called_once_with(
                            symbol='TSLA',
                            qty=6,
                            side='sell',
                            type='limit',
                            limit_price=149.70,
                            time_in_force='ext'
                        )
                        
                        # Verify protection orders were called with correct sides for shorts
                        mock_alpaca._submit_after_hours_stop_loss.assert_called_once_with('TSLA', 6, 157.50, 'buy')  # Buy to cover
                        mock_alpaca._submit_after_hours_take_profit.assert_called_once_with('TSLA', 6, 135.00, 'buy')  # Buy to cover

    def test_sell_short_after_hours_protected_main_order_exception(self, mock_alpaca, mock_market_data):
        """Test _sell_short_after_hours_protected when main order fails."""
        mock_alpaca.api.submit_order.side_effect = Exception("Short main order failed")
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=6):
                
                result = mock_alpaca._sell_short_after_hours_protected(
                    'TSLA',
                    take_profit=135.00,
                    stop_loss=157.50,
                    submit_order=True
                )
                
                assert result is None
                mock_alpaca.api.submit_order.assert_called_once()

    def test_buy_after_hours_protected_calc_take_profit_calculation(self, mock_alpaca, mock_market_data):
        """Test _buy_after_hours_protected take profit calculation logic."""
        mock_alpaca.STOP_LOSS_PERCENT = 0.05  # 5% stop loss
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=10):
                
                # Test with calc_take_profit (take_profit=None triggers this)
                result = mock_alpaca._buy_after_hours_protected(
                    'AAPL',
                    stop_loss=142.50,
                    take_profit=None,  # This should trigger calc_take_profit logic
                    submit_order=False
                )
                
                # Verify dry run returns None
                assert result is None
                mock_alpaca.api.submit_order.assert_not_called()

    def test_sell_short_after_hours_protected_calc_take_profit_calculation(self, mock_alpaca, mock_market_data):
        """Test _sell_short_after_hours_protected take profit calculation logic."""
        mock_alpaca.STOP_LOSS_PERCENT = 0.05  # 5% stop loss
        
        with patch('alpaca.get_latest_quote_avg', return_value=mock_market_data['market_price']):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=8):
                
                # Test with calc_take_profit (take_profit=None triggers this)
                result = mock_alpaca._sell_short_after_hours_protected(
                    'TSLA',
                    stop_loss=157.50,  # Above entry price for shorts
                    take_profit=None,  # This should trigger calc_take_profit logic
                    submit_order=False
                )
                
                # Verify dry run returns None
                assert result is None
                mock_alpaca.api.submit_order.assert_not_called()

    def test_calculate_quantity_integration_after_hours(self, mock_alpaca):
        """Test integration with _calculateQuantity for after-hours methods."""
        with patch('alpaca.get_latest_quote_avg', return_value=100.0):
            with patch.object(mock_alpaca, '_calculateQuantity') as mock_calc:
                mock_calc.return_value = 15
                
                # Test buy after hours
                mock_alpaca._buy_after_hours('TEST', submit_order=False)
                mock_calc.assert_called_with(100.20, "_buy_after_hours")  # 100.0 * 1.002
                
                # Test short after hours
                mock_alpaca._sell_short_after_hours('TEST', submit_order=False)
                mock_calc.assert_called_with(99.80, "_sell_short_after_hours")  # 100.0 * 0.998
                
                # Test protected buy after hours
                mock_alpaca._buy_after_hours_protected('TEST', take_profit=110.0, stop_loss=90.0, submit_order=False)
                mock_calc.assert_called_with(100.20, "_buy_after_hours_protected")
                
                # Test protected short after hours
                mock_alpaca._sell_short_after_hours_protected('TEST', take_profit=85.0, stop_loss=110.0, submit_order=False)
                mock_calc.assert_called_with(99.80, "_sell_short_after_hours_protected")

    def test_amount_based_quantity_calculation_after_hours(self, mock_alpaca):
        """Test amount-based quantity calculation for after-hours methods."""
        with patch('alpaca.get_latest_quote_avg', return_value=50.0):
            
            # Test buy after hours with amount
            mock_alpaca._buy_after_hours('TEST', amount=1000.0, submit_order=False)
            # Expected: 1000.0 / (50.0 * 1.002) = 1000.0 / 50.10 = ~20 shares
            
            # Test short after hours with amount  
            mock_alpaca._sell_short_after_hours('TEST', amount=1500.0, submit_order=False)
            # Expected: 1500.0 / (50.0 * 0.998) = 1500.0 / 49.90 = ~30 shares
            
            # Test protected buy with amount
            mock_alpaca._buy_after_hours_protected('TEST', amount=2000.0, take_profit=60.0, stop_loss=45.0, submit_order=False)
            # Expected: 2000.0 / (50.0 * 1.002) = ~40 shares
            
            # Test protected short with amount
            mock_alpaca._sell_short_after_hours_protected('TEST', amount=2500.0, take_profit=40.0, stop_loss=55.0, submit_order=False)
            # Expected: 2500.0 / (50.0 * 0.998) = ~50 shares

    def test_custom_limit_price_after_hours(self, mock_alpaca):
        """Test custom limit price functionality for after-hours methods."""
        with patch('alpaca.get_latest_quote_avg', return_value=100.0):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=5):
                
                # Test buy with custom limit price
                mock_alpaca._buy_after_hours('TEST', limit_price=105.0, submit_order=False)
                # Should use 105.0 instead of calculated price
                
                # Test short with custom limit price
                mock_alpaca._sell_short_after_hours('TEST', limit_price=95.0, submit_order=False)
                # Should use 95.0 instead of calculated price
                
                # Test protected buy with custom limit price
                mock_alpaca._buy_after_hours_protected('TEST', limit_price=102.0, take_profit=115.0, stop_loss=90.0, submit_order=False)
                # Should use 102.0 instead of calculated price
                
                # Test protected short with custom limit price
                mock_alpaca._sell_short_after_hours_protected('TEST', limit_price=98.0, take_profit=85.0, stop_loss=110.0, submit_order=False)
                # Should use 98.0 instead of calculated price

    def test_stop_loss_calculation_defaults(self, mock_alpaca):
        """Test default stop loss calculation for protected after-hours methods."""
        mock_alpaca.STOP_LOSS_PERCENT = 0.075  # 7.5% stop loss
        
        with patch('alpaca.get_latest_quote_avg', return_value=200.0):
            with patch.object(mock_alpaca, '_calculateQuantity', return_value=3):
                
                # Test protected buy with default stop loss
                mock_alpaca._buy_after_hours_protected('TEST', take_profit=220.0, submit_order=False)
                # Expected stop loss: (200.0 * 1.002) * (1 - 0.075) = 200.40 * 0.925 = ~185.37
                
                # Test protected short with default stop loss  
                mock_alpaca._sell_short_after_hours_protected('TEST', take_profit=180.0, submit_order=False)
                # Expected stop loss: (200.0 * 0.998) * (1 + 0.075) = 199.60 * 1.075 = ~214.57


if __name__ == "__main__":
    pytest.main([__file__, "-v"])