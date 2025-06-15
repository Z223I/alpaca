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
    parser.add_argument('--symbol', type=str, required=False,
                      help='Stock symbol for bracket order')
    parser.add_argument('--quantity', type=int, required=False,
                      help='Number of shares for bracket order')
    parser.add_argument('--market_price', type=float, required=False,
                      help='Current market price for bracket order')
    parser.add_argument('--submit', action='store_true',
                      help='Actually submit the bracket order (default: False)')
    parser.add_argument('-q', '--get_latest_quote', action='store_true',
                      help='Get latest quote for a symbol')
    parser.add_argument('--buy', action='store_true',
                      help='Execute a buy order for the specified symbol')

    args = parser.parse_args(userArgs)

    # Validate bracket order arguments
    if args.bracket_order:
        if not all([args.symbol, args.quantity, args.market_price]):
            parser.error("--bracket_order requires --symbol, --quantity, and --market_price")
    
    # Validate quote arguments
    if args.get_latest_quote:
        if not args.symbol:
            parser.error("--get_latest_quote requires --symbol")
    
    # Validate buy arguments
    if args.buy:
        if not args.symbol:
            parser.error("--buy requires --symbol")

    return args
