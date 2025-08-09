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
    parser.add_argument('--take-profit-percent', type=float, required=False,
                      help='Take profit percentage above current market price (requires --symbol and --quantity)')
    parser.add_argument('--submit', action='store_true',
                      help='Actually submit the bracket order (default: False)')
    parser.add_argument('-q', '--get-latest-quote', action='store_true',
                      help='Get latest quote for a symbol')
    parser.add_argument('--buy', action='store_true',
                      help='Execute a buy order for the specified symbol')
    parser.add_argument('--buy-market', action='store_true',
                      help='Execute a simple market buy order for the specified symbol')
    parser.add_argument('--buy-market-trailing-sell', action='store_true',
                      help='Execute market buy then automatic trailing sell when filled')
    parser.add_argument('--sell-trailing', action='store_true',
                      help='Execute a trailing sell order for the specified symbol')
    parser.add_argument('--trailing-percent', type=float, required=False,
                      help='Trailing percentage for trailing sell order (default: 7.5)')
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

    # Liquidation arguments
    parser.add_argument('--liquidate', action='store_true',
                      help='Liquidate position for a specific symbol (requires --symbol)')
    parser.add_argument('--liquidate-all', action='store_true',
                      help='Liquidate all open positions and optionally cancel all orders')
    parser.add_argument('--cancel-orders', action='store_true',
                      help='Cancel all open orders (used with --liquidate-all)')

    # Cancel all orders argument
    parser.add_argument('--cancel-all-orders', action='store_true',
                      help='Cancel all open orders immediately (cannot be combined with other arguments)')

    # Display-only arguments
    parser.add_argument('--positions', action='store_true',
                      help='Display current positions only')
    parser.add_argument('--cash', action='store_true',
                      help='Display cash balance only')
    parser.add_argument('--active-order', action='store_true',
                      help='Display active orders only')

    # PNL report argument
    parser.add_argument('--PNL', action='store_true',
                      help='Display daily profit/loss summary (standalone use only)')

    # Account configuration arguments
    parser.add_argument('--account-name', type=str, default='Primary',
                      help='Account name to use (default: Primary)')
    parser.add_argument('--account', type=str, default='paper',
                      help='Account environment to use: paper, live, cash (default: paper)')

    args = parser.parse_args(userArgs)

    # Validate account configuration arguments
    # Import here to avoid circular imports
    try:
        from alpaca_config import get_current_config
        config = get_current_config()
        # Validate that the account configuration exists
        try:
            config.get_environment_config("alpaca", args.account_name, args.account)
        except (KeyError, AttributeError) as e:
            parser.error(f"Invalid account configuration {args.account_name}:{args.account} - {e}")
    except ImportError:
        parser.error("Account validation requires alpaca_config.py to be available")

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

    # Validate buy-market arguments
    if args.buy_market:
        if not args.symbol:
            parser.error("--buy-market requires --symbol")

    # Validate buy-market-trailing-sell arguments
    if args.buy_market_trailing_sell:
        if not args.symbol:
            parser.error("--buy-market-trailing-sell requires --symbol")

    # Validate sell-trailing arguments
    if args.sell_trailing:
        if not args.symbol:
            parser.error("--sell-trailing requires --symbol")
        if not args.quantity:
            parser.error("--sell-trailing requires --quantity")

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

    # Ensure buy, sell-trailing, sell_short, buy-market, and buy-market-trailing-sell are mutually exclusive
    buy_operations = [args.buy, args.buy_market, args.buy_market_trailing_sell]
    sell_operations = [args.sell_trailing, args.sell_short]

    if sum(buy_operations) > 1:
        parser.error("--buy, --buy-market, and --buy-market-trailing-sell cannot be used together")
    if sum(sell_operations) > 1:
        parser.error("--sell-trailing and --sell-short cannot be used together")
    if any(buy_operations) and any(sell_operations):
        parser.error("Buy and sell operations cannot be used together")

    # Validate liquidation arguments
    if args.liquidate:
        if not args.symbol:
            parser.error("--liquidate requires --symbol")
        if args.liquidate_all:
            parser.error("--liquidate and --liquidate-all cannot be used together")

    if args.liquidate_all:
        if args.symbol:
            parser.error("--liquidate-all cannot be used with --symbol")

    if args.cancel_orders and not args.liquidate_all:
        parser.error("--cancel-orders can only be used with --liquidate-all")

    # Ensure liquidation is mutually exclusive with other operations
    if args.liquidate or args.liquidate_all:
        conflicting_args = [args.buy, args.buy_market, args.buy_market_trailing_sell, args.sell_trailing, args.sell_short, args.bracket_order, args.future_bracket_order, args.get_latest_quote]
        if any(conflicting_args):
            parser.error("Liquidation operations cannot be combined with other trading operations")

    # Validate after_hours arguments
    if args.after_hours:
        if not (args.buy or args.buy_market or args.buy_market_trailing_sell or args.sell_trailing or args.sell_short):
            parser.error("--after-hours requires either --buy, --buy-market, --buy-market-trailing-sell, --sell-trailing, or --sell-short")
        if not args.symbol:
            parser.error("--after-hours requires --symbol")

    # Validate cancel-all-orders argument (must be standalone)
    if args.cancel_all_orders:
        # Check if any other arguments are present
        for arg_name, arg_value in vars(args).items():
            if arg_name != 'cancel_all_orders':
                # Check if argument has been set to non-default value
                if isinstance(arg_value, bool) and arg_value:
                    parser.error("--cancel-all-orders cannot be combined with any other arguments")
                elif arg_value is not None and not isinstance(arg_value, bool):
                    parser.error("--cancel-all-orders cannot be combined with any other arguments")

    # Validate PNL argument (must be standalone except for account configuration)
    if args.PNL:
        # Check if any other arguments are present (excluding account configuration)
        for arg_name, arg_value in vars(args).items():
            if arg_name not in ['PNL', 'account_name', 'account']:
                # Check if argument has been set to non-default value
                if isinstance(arg_value, bool) and arg_value:
                    parser.error("--PNL cannot be combined with any other arguments (except account configuration)")
                elif arg_value is not None and not isinstance(arg_value, bool):
                    parser.error("--PNL cannot be combined with any other arguments (except account configuration)")

    # Validate take-profit-percent arguments
    if args.take_profit_percent:
        if not args.symbol:
            parser.error("--take-profit-percent requires --symbol")
        if not args.quantity:
            parser.error("--take-profit-percent requires --quantity")
        if args.quantity <= 0:
            parser.error("--take-profit-percent requires --quantity to be greater than 0")
        if args.take_profit_percent <= 0:
            parser.error("--take-profit-percent must be greater than 0")

    # Validate display-only arguments
    display_args = [args.positions, args.cash, args.active_order]
    if any(display_args):
        # Check if any non-display arguments are present (excluding account configuration)
        for arg_name, arg_value in vars(args).items():
            if arg_name not in ['positions', 'cash', 'active_order', 'account_name', 'account']:
                # Check if argument has been set to non-default value
                if isinstance(arg_value, bool) and arg_value:
                    parser.error("Display-only arguments (--positions, --cash, --active-order) cannot be combined with other operations")
                elif arg_value is not None and not isinstance(arg_value, bool):
                    parser.error("Display-only arguments (--positions, --cash, --active-order) cannot be combined with other operations")

    # Normalize symbol to uppercase if provided
    if args.symbol:
        args.symbol = args.symbol.upper()

    return args
