import argparse
from typing import Optional, List


def parse_args(userArgs: Optional[List[str]]) -> argparse.Namespace:
    """
    Parse command line arguments for the trading bot.
    
    Args:
        userArgs: List of command line arguments to parse
        
    Returns:
        Parsed arguments namespace
        
    Raises:
        SystemExit: If required arguments are missing for bracket orders
    """
    parser = argparse.ArgumentParser(description='Alpaca Trading Bot')
    parser.add_argument('-b', '--bracket_order', action='store_true',
                      help='Execute bracket order')
    parser.add_argument('-f', '--future_bracket_order', action='store_true',
                      help='Execute future bracket order with limit entry')
    parser.add_argument('--symbol', type=str, required=False,
                      help='Stock symbol for bracket order')
    parser.add_argument('--quantity', type=int, required=False,
                      help='Number of shares for bracket order')
    parser.add_argument('--market_price', type=float, required=False,
                      help='Current market price for bracket order')
    parser.add_argument('--limit_price', type=float, required=False,
                      help='Limit price for future bracket order entry')
    parser.add_argument('--stop_price', type=float, required=False,
                      help='Stop loss price for future bracket order')
    parser.add_argument('--take_profit', type=float, required=False,
                      help='Take profit price for bracket orders')
    parser.add_argument('--submit', action='store_true',
                      help='Actually submit the bracket order (default: False)')
    parser.add_argument('-q', '--get_latest_quote', action='store_true',
                      help='Get latest quote for a symbol')
    parser.add_argument('--buy', action='store_true',
                      help='Execute a buy order for the specified symbol')
    parser.add_argument('--stop_loss', type=float, required=False,
                      help='Custom stop loss price for buy orders')
    parser.add_argument('--calc_take_profit', action='store_true',
                      help='Calculate take profit as (latest_quote - stop_loss) * 1.5')
    parser.add_argument('--amount', type=float, required=False,
                      help='Dollar amount to invest (will calculate quantity automatically)')

    args = parser.parse_args(userArgs)

    # Validate bracket order arguments
    if args.bracket_order:
        if not all([args.symbol, args.quantity, args.market_price, args.take_profit]):
            parser.error("--bracket_order requires --symbol, --quantity, --market_price, and --take_profit")
    
    # Validate future bracket order arguments
    if args.future_bracket_order:
        if not all([args.symbol, args.limit_price, args.stop_price, args.take_profit]):
            parser.error("--future_bracket_order requires --symbol, --limit_price, --stop_price, and --take_profit")
        # Set default quantity to 0 for auto-calculation if not provided
        if args.quantity is None:
            args.quantity = 0
    
    # Validate quote arguments
    if args.get_latest_quote:
        if not args.symbol:
            parser.error("--get_latest_quote requires --symbol")
    
    # Validate buy arguments
    if args.buy:
        # Check for calc_take_profit usage warnings
        if args.calc_take_profit and not args.stop_loss:
            parser.error("--calc_take_profit requires --stop_loss")
        if args.calc_take_profit and args.take_profit:
            print("Warning: --calc_take_profit used with --take_profit. --take_profit will be ignored.")
        
        # Require symbol and either take_profit or calc_take_profit
        if not args.symbol:
            parser.error("--buy requires --symbol")
        if not args.take_profit and not args.calc_take_profit:
            parser.error("--buy requires either --take_profit or --calc_take_profit")

    return args
