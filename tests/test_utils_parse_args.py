"""
Comprehensive tests for command line argument parsing utility.
"""

import pytest
import argparse
from unittest.mock import patch
from atoms.api.parse_args import parse_args


class TestParseArgs:
    """Test suite for parse_args function."""
    
    def test_parse_args_no_arguments(self):
        """Test parsing with no arguments."""
        args = parse_args([])
        
        assert not args.bracket_order
        assert not args.future_bracket_order
        assert not args.get_latest_quote
        assert not args.buy
        assert not args.submit
        assert args.symbol is None
        assert args.quantity is None
        assert args.market_price is None
        assert args.limit_price is None
        assert args.stop_price is None
        assert args.take_profit is None
    
    def test_parse_args_bracket_order_valid(self):
        """Test parsing valid bracket order arguments."""
        args = parse_args([
            '--bracket_order',
            '--symbol', 'AAPL',
            '--quantity', '100',
            '--market_price', '150.50',
            '--take_profit', '160.00'
        ])
        
        assert args.bracket_order is True
        assert args.symbol == 'AAPL'
        assert args.quantity == 100
        assert args.market_price == 150.50
        assert args.take_profit == 160.00
        assert not args.submit  # Default False
    
    def test_parse_args_bracket_order_with_submit(self):
        """Test parsing bracket order with submit flag."""
        args = parse_args([
            '--bracket_order',
            '--symbol', 'AAPL',
            '--quantity', '100',
            '--market_price', '150.50',
            '--take_profit', '160.00',
            '--submit'
        ])
        
        assert args.bracket_order is True
        assert args.submit is True
    
    def test_parse_args_bracket_order_missing_symbol(self):
        """Test bracket order validation with missing symbol."""
        with pytest.raises(SystemExit):
            parse_args([
                '--bracket_order',
                '--quantity', '100',
                '--market_price', '150.50',
                '--take_profit', '160.00'
            ])
    
    def test_parse_args_bracket_order_missing_quantity(self):
        """Test bracket order validation with missing quantity."""
        with pytest.raises(SystemExit):
            parse_args([
                '--bracket_order',
                '--symbol', 'AAPL',
                '--market_price', '150.50',
                '--take_profit', '160.00'
            ])
    
    def test_parse_args_bracket_order_missing_market_price(self):
        """Test bracket order validation with missing market price."""
        with pytest.raises(SystemExit):
            parse_args([
                '--bracket_order',
                '--symbol', 'AAPL',
                '--quantity', '100',
                '--take_profit', '160.00'
            ])
    
    def test_parse_args_bracket_order_missing_take_profit(self):
        """Test bracket order validation with missing take profit."""
        with pytest.raises(SystemExit):
            parse_args([
                '--bracket_order',
                '--symbol', 'AAPL',
                '--quantity', '100',
                '--market_price', '150.50'
            ])
    
    def test_parse_args_future_bracket_order_valid(self):
        """Test parsing valid future bracket order arguments."""
        args = parse_args([
            '--future_bracket_order',
            '--symbol', 'TSLA',
            '--limit_price', '200.00',
            '--stop_price', '190.00',
            '--take_profit', '220.00'
        ])
        
        assert args.future_bracket_order is True
        assert args.symbol == 'TSLA'
        assert args.limit_price == 200.00
        assert args.stop_price == 190.00
        assert args.take_profit == 220.00
        assert args.quantity == 0  # Default auto-calculation
    
    def test_parse_args_future_bracket_order_with_quantity(self):
        """Test future bracket order with explicit quantity."""
        args = parse_args([
            '--future_bracket_order',
            '--symbol', 'TSLA',
            '--quantity', '50',
            '--limit_price', '200.00',
            '--stop_price', '190.00',
            '--take_profit', '220.00'
        ])
        
        assert args.quantity == 50
    
    def test_parse_args_future_bracket_order_missing_symbol(self):
        """Test future bracket order validation with missing symbol."""
        with pytest.raises(SystemExit):
            parse_args([
                '--future_bracket_order',
                '--limit_price', '200.00',
                '--stop_price', '190.00',
                '--take_profit', '220.00'
            ])
    
    def test_parse_args_future_bracket_order_missing_limit_price(self):
        """Test future bracket order validation with missing limit price."""
        with pytest.raises(SystemExit):
            parse_args([
                '--future_bracket_order',
                '--symbol', 'TSLA',
                '--stop_price', '190.00',
                '--take_profit', '220.00'
            ])
    
    def test_parse_args_future_bracket_order_missing_stop_price(self):
        """Test future bracket order validation with missing stop price."""
        with pytest.raises(SystemExit):
            parse_args([
                '--future_bracket_order',
                '--symbol', 'TSLA',
                '--limit_price', '200.00',
                '--take_profit', '220.00'
            ])
    
    def test_parse_args_future_bracket_order_missing_take_profit(self):
        """Test future bracket order validation with missing take profit."""
        with pytest.raises(SystemExit):
            parse_args([
                '--future_bracket_order',
                '--symbol', 'TSLA',
                '--limit_price', '200.00',
                '--stop_price', '190.00'
            ])
    
    def test_parse_args_get_latest_quote_valid(self):
        """Test parsing valid get latest quote arguments."""
        args = parse_args([
            '--get_latest_quote',
            '--symbol', 'MSFT'
        ])
        
        assert args.get_latest_quote is True
        assert args.symbol == 'MSFT'
    
    def test_parse_args_get_latest_quote_missing_symbol(self):
        """Test get latest quote validation with missing symbol."""
        with pytest.raises(SystemExit):
            parse_args(['--get_latest_quote'])
    
    def test_parse_args_buy_order_valid(self):
        """Test parsing valid buy order arguments."""
        args = parse_args([
            '--buy',
            '--symbol', 'AMZN',
            '--take_profit', '140.00'
        ])
        
        assert args.buy is True
        assert args.symbol == 'AMZN'
        assert args.take_profit == 140.00
    
    def test_parse_args_buy_order_missing_symbol(self):
        """Test buy order validation with missing symbol."""
        with pytest.raises(SystemExit):
            parse_args([
                '--buy',
                '--take_profit', '140.00'
            ])
    
    def test_parse_args_buy_order_missing_take_profit(self):
        """Test buy order validation with missing take profit."""
        with pytest.raises(SystemExit):
            parse_args([
                '--buy',
                '--symbol', 'AMZN'
            ])
    
    def test_parse_args_short_flags(self):
        """Test parsing with short flags."""
        args = parse_args([
            '-b',  # --bracket_order
            '--symbol', 'AAPL',
            '--quantity', '100',
            '--market_price', '150.50',
            '--take_profit', '160.00'
        ])
        
        assert args.bracket_order is True
        
        args2 = parse_args([
            '-f',  # --future_bracket_order
            '--symbol', 'TSLA',
            '--limit_price', '200.00',
            '--stop_price', '190.00',
            '--take_profit', '220.00'
        ])
        
        assert args2.future_bracket_order is True
        
        args3 = parse_args([
            '-q',  # --get_latest_quote
            '--symbol', 'MSFT'
        ])
        
        assert args3.get_latest_quote is True
    
    def test_parse_args_data_type_conversion(self):
        """Test that arguments are converted to correct data types."""
        args = parse_args([
            '--bracket_order',
            '--symbol', 'AAPL',
            '--quantity', '100',
            '--market_price', '150.50',
            '--take_profit', '160.75'
        ])
        
        assert isinstance(args.quantity, int)
        assert isinstance(args.market_price, float)
        assert isinstance(args.take_profit, float)
        assert isinstance(args.symbol, str)
    
    def test_parse_args_invalid_data_types(self):
        """Test parsing with invalid data types."""
        # Invalid quantity (should be int)
        with pytest.raises(SystemExit):
            parse_args([
                '--bracket_order',
                '--symbol', 'AAPL',
                '--quantity', 'not_a_number',
                '--market_price', '150.50',
                '--take_profit', '160.00'
            ])
        
        # Invalid price (should be float)
        with pytest.raises(SystemExit):
            parse_args([
                '--bracket_order',
                '--symbol', 'AAPL',
                '--quantity', '100',
                '--market_price', 'not_a_price',
                '--take_profit', '160.00'
            ])
    
    def test_parse_args_multiple_conflicting_actions(self):
        """Test parsing with multiple action flags (should be allowed)."""
        args = parse_args([
            '--bracket_order',
            '--get_latest_quote',
            '--symbol', 'AAPL',
            '--quantity', '100',
            '--market_price', '150.50',
            '--take_profit', '160.00'
        ])
        
        # Both flags should be set
        assert args.bracket_order is True
        assert args.get_latest_quote is True
    
    def test_parse_args_negative_values(self):
        """Test parsing with negative numeric values."""
        args = parse_args([
            '--bracket_order',
            '--symbol', 'AAPL',
            '--quantity', '-100',  # Negative quantity
            '--market_price', '150.50',
            '--take_profit', '160.00'
        ])
        
        assert args.quantity == -100
    
    def test_parse_args_zero_values(self):
        """Test parsing with zero values."""
        # Zero values should still be valid for bracket orders
        with pytest.raises(SystemExit):  # Validation may fail with 0 market price
            parse_args([
                '--bracket_order',
                '--symbol', 'AAPL',
                '--quantity', '0',
                '--market_price', '0.0',
                '--take_profit', '160.00'
            ])
        
        # Test zero values in a context that should work
        args = parse_args([
            '--symbol', 'AAPL',
            '--quantity', '0',
            '--market_price', '0.0'
        ])
        
        assert args.quantity == 0
        assert args.market_price == 0.0
    
    def test_parse_args_very_large_values(self):
        """Test parsing with very large numeric values."""
        args = parse_args([
            '--bracket_order',
            '--symbol', 'AAPL',
            '--quantity', '999999',
            '--market_price', '999999.99',
            '--take_profit', '1000000.00'
        ])
        
        assert args.quantity == 999999
        assert args.market_price == 999999.99
        assert args.take_profit == 1000000.00
    
    def test_parse_args_special_symbols(self):
        """Test parsing with special symbol formats."""
        # Symbol with dot
        args1 = parse_args([
            '--get_latest_quote',
            '--symbol', 'BRK.A'
        ])
        assert args1.symbol == 'BRK.A'
        
        # Symbol with hyphen
        args2 = parse_args([
            '--get_latest_quote',
            '--symbol', 'BRK-B'
        ])
        assert args2.symbol == 'BRK-B'
        
        # Symbol with numbers
        args3 = parse_args([
            '--get_latest_quote',
            '--symbol', 'STOCK123'
        ])
        assert args3.symbol == 'STOCK123'
    
    def test_parse_args_none_input(self):
        """Test parsing with None as input."""
        # When None is passed, it uses sys.argv which includes test runner arguments
        # This will cause a SystemExit due to unrecognized arguments
        with pytest.raises(SystemExit):
            parse_args(None)
    
    def test_parse_args_empty_string_symbol(self):
        """Test parsing with empty string symbol."""
        # Empty string symbol should fail validation
        with pytest.raises(SystemExit):
            parse_args([
                '--get_latest_quote',
                '--symbol', ''
            ])
    
    def test_parse_args_whitespace_symbol(self):
        """Test parsing with whitespace in symbol."""
        args = parse_args([
            '--get_latest_quote',
            '--symbol', ' AAPL '
        ])
        
        assert args.symbol == ' AAPL '
    
    def test_parse_args_decimal_precision(self):
        """Test parsing with high decimal precision."""
        args = parse_args([
            '--bracket_order',
            '--symbol', 'AAPL',
            '--quantity', '100',
            '--market_price', '150.123456789',
            '--take_profit', '160.987654321'
        ])
        
        assert args.market_price == 150.123456789
        assert args.take_profit == 160.987654321
    
    def test_parse_args_scientific_notation(self):
        """Test parsing with scientific notation."""
        args = parse_args([
            '--bracket_order',
            '--symbol', 'AAPL',
            '--quantity', '1000',
            '--market_price', '1.5e2',  # 150.0
            '--take_profit', '1.6e2'    # 160.0
        ])
        
        assert args.market_price == 150.0
        assert args.take_profit == 160.0
    
    def test_parse_args_all_parameters_present(self):
        """Test parsing with all possible parameters."""
        args = parse_args([
            '--bracket_order',
            '--future_bracket_order',
            '--get_latest_quote',
            '--buy',
            '--submit',
            '--symbol', 'AAPL',
            '--quantity', '100',
            '--market_price', '150.50',
            '--limit_price', '149.00',
            '--stop_price', '145.00',
            '--take_profit', '160.00'
        ])
        
        assert args.bracket_order is True
        assert args.future_bracket_order is True
        assert args.get_latest_quote is True
        assert args.buy is True
        assert args.submit is True
        assert args.symbol == 'AAPL'
        assert args.quantity == 100
        assert args.market_price == 150.50
        assert args.limit_price == 149.00
        assert args.stop_price == 145.00
        assert args.take_profit == 160.00
    
    def test_parse_args_help_option(self):
        """Test that help option works (should exit)."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(['--help'])
        
        # Help should exit with code 0
        assert exc_info.value.code == 0
    
    def test_parse_args_unknown_argument(self):
        """Test parsing with unknown argument."""
        with pytest.raises(SystemExit):
            parse_args(['--unknown_argument'])
    
    def test_parse_args_argument_parser_properties(self):
        """Test that the argument parser has correct properties."""
        # This test verifies the parser configuration indirectly
        # by checking that valid arguments are recognized
        
        # Test that all expected arguments are defined
        try:
            args = parse_args([
                '--bracket_order', '--symbol', 'TEST', '--quantity', '1', 
                '--market_price', '1.0', '--take_profit', '2.0'
            ])
            assert hasattr(args, 'bracket_order')
            assert hasattr(args, 'future_bracket_order')
            assert hasattr(args, 'get_latest_quote')
            assert hasattr(args, 'buy')
            assert hasattr(args, 'submit')
            assert hasattr(args, 'symbol')
            assert hasattr(args, 'quantity')
            assert hasattr(args, 'market_price')
            assert hasattr(args, 'limit_price')
            assert hasattr(args, 'stop_price')
            assert hasattr(args, 'take_profit')
        except Exception:
            pytest.fail("Expected arguments are not properly defined")
    
    def test_parse_args_return_type(self):
        """Test that parse_args returns correct type."""
        args = parse_args([])
        assert isinstance(args, argparse.Namespace)