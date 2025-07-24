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
    parser.add_argument('-b', '--bracket-order', action='store_true',
                      help='Execute bracket order')
    parser.add_argument('-f', '--future-bracket-order', action='store_true',
                      help='Execute future bracket order with limit entry')
    parser.add_argument('--symbol', type=str, required=False,
                      help='Stock symbol for bracket order')
    parser.add_argument('--quantity', type=int, required=False,
                      help='Number of shares for bracket order')
    parser.add_argument('--market-price', type=float, required=False,
                      help='Current market price for bracket order')
    parser.add_argument('--limit-price', type=float, required=False,
                      help='Limit price for future bracket order entry')
    parser.add_argument('--stop-price', type=float, required=False,
                      help='Stop loss price for future bracket order')
    parser.add_argument('--take-profit', type=float, required=False,
                      help='Take profit price for bracket orders')
    parser.add_argument('--submit', action='store_true',
                      help='Actually submit the bracket order (default: False)')
    parser.add_argument('-q', '--get-latest-quote', action='store_true',
                      help='Get latest quote for a symbol')
    parser.add_argument('--buy', action='store_true',
                      help='Execute a buy order for the specified symbol')
    parser.add_argument('--sell-short', action='store_true',
                      help='Execute a short sell order for bearish predictions')
    parser.add_argument('--after-hours', action='store_true',
                      help='Execute order for after-hours/extended hours trading (limit orders only)')
    parser.add_argument('--custom-limit-price', type=float, required=False,
                      help='Custom limit price for after-hours orders')
    parser.add_argument('--stop-loss', type=float, required=False,
                      help='Custom stop loss price for buy/short orders')
    parser.add_argument('--calc-take-profit', action='store_true',
                      help='Calculate take profit as (latest_quote - stop_loss) * 1.5')
    parser.add_argument('--amount', type=float, required=False,
                      help='Dollar amount to invest (will calculate quantity automatically)')

    args = parser.parse_args(userArgs)

    # Validate bracket order arguments
    if args.bracket_order:
        if not all([args.symbol, args.quantity, args.market_price, args.take_profit]):
            parser.error("--bracket-order requires --symbol, --quantity, --market-price, and --take-profit")
    
    # Validate future bracket order arguments
    if args.future_bracket_order:
        if not all([args.symbol, args.limit_price, args.stop_price, args.take_profit]):
            parser.error("--future-bracket-order requires --symbol, --limit-price, --stop-price, and --take-profit")
        # Set default quantity to 0 for auto-calculation if not provided
        if args.quantity is None:
            args.quantity = 0
    
    # Validate quote arguments
    if args.get_latest_quote:
        if not args.symbol:
            parser.error("--get-latest-quote requires --symbol")
    
    # Validate buy arguments
    if args.buy:
        # Check for calc_take_profit usage warnings
        if args.calc_take_profit and not args.stop_loss:
            parser.error("--calc-take-profit requires --stop-loss")
        if args.calc_take_profit and args.take_profit:
            print("Warning: --calc-take-profit used with --take-profit. --take-profit will be ignored.")
        
        # Require symbol and either take_profit or calc_take_profit
        if not args.symbol:
            parser.error("--buy requires --symbol")
        if not args.take_profit and not args.calc_take_profit:
            parser.error("--buy requires either --take-profit or --calc-take-profit")
    
    # Validate sell_short arguments
    if args.sell_short:
        # Check for calc_take_profit usage warnings
        if args.calc_take_profit and not args.stop_loss:
            parser.error("--calc-take-profit requires --stop-loss")
        if args.calc_take_profit and args.take_profit:
            print("Warning: --calc-take-profit used with --take-profit. --take-profit will be ignored.")
        
        # Require symbol and either take_profit or calc_take_profit
        if not args.symbol:
            parser.error("--sell-short requires --symbol")
        if not args.take_profit and not args.calc_take_profit:
            parser.error("--sell-short requires either --take-profit or --calc-take-profit")
    
    # Ensure buy and sell_short are mutually exclusive
    if args.buy and args.sell_short:
        parser.error("--buy and --sell-short cannot be used together")
    
    # Validate after_hours arguments
    if args.after_hours:
        if not (args.buy or args.sell_short):
            parser.error("--after-hours requires either --buy or --sell-short")
        if not args.symbol:
            parser.error("--after-hours requires --symbol")

    return args
