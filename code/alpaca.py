#import requests
# import json
# import math
# import time
import sys
import os
import math
from typing import Optional, List, Dict, Any

import alpaca_trade_api as tradeapi   # pip3 install alpaca-trade-api -U
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.api.get_cash import get_cash
# from atoms.api.get_active_orders import get_active_orders
from atoms.api.get_positions import get_positions
from atoms.api.get_latest_quote import get_latest_quote
from atoms.display.print_cash import print_cash
from atoms.display.print_orders import print_active_orders
from atoms.display.print_positions import print_positions
from atoms.display.print_quote import print_quote
from atoms.utils.delay import delay
from atoms.utils.parse_args import parse_args

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")


class alpaca_private:
    """
    Alpaca trading API wrapper for automated trading operations.

    This class provides methods for interacting with the Alpaca trading API,
    including order management, position tracking, and bracket order execution.
    """

    STOP_LOSS_PERCENT = 0.10  # 10% stop loss constant

    def __init__(self, userArgs: Optional[List[str]] = None) -> None:
        """
        Initialize the Alpaca trading client.

        Args:
            userArgs: Optional command line arguments for configuration
        """
        self.history = {}

        # Parse arguments
        self.args = parse_args(userArgs)

        # Set portfolio risk from environment variable or use default
        self.PORTFOLIO_RISK = float(os.getenv('PORTFOLIO_RISK', '0.10'))

        self.key = os.getenv('ALPACA_API_KEY')
        self.secret = os.getenv('ALPACA_SECRET_KEY')
        self.headers = {'APCA-API-KEY-ID':self.key, 'APCA-API-SECRET-KEY':self.secret}

        self.baseURL = os.getenv('ALPACA_BASE_URL')
        self.accountURL = "{}/v2/account".format(self.baseURL)
        self.ordersURL = "{}/v2/orders".format(self.baseURL)

        self.api =  tradeapi.REST(self.key, self.secret, self.baseURL)
        self.active_orders = []


    def _buy(self, symbol: str, take_profit: float, submit_order: bool = False) -> None:
        """
        Execute a buy order with bracket order protection.

        This method retrieves the latest quote, calculates position size based on
        available cash and existing positions, and submits a bracket order with
        stop loss protection.

        Args:
            symbol: The stock symbol to buy
            take_profit: The take profit price for the bracket order
            submit_order: Whether to actually submit the order (default: False for dry run)
        """
        # Get current market data for the symbol
        latest_quote = get_latest_quote(self.api, symbol)
        market_price = latest_quote.ask_price

        # Calculate stop loss price
        stop_price = round(market_price * (1 - self.STOP_LOSS_PERCENT), 2)

        # Get current account information
        cash = get_cash(self.api)
        positions = get_positions(self.api)

        # TODO: Update logic to properly handle different portfolio risk values
        if self.PORTFOLIO_RISK != 0.50:
            print("_buy() logic must be changed to use the new portfolio risk value")

        # Calculate quantity based on portfolio state
        if not positions:
            # First position: use portfolio risk percentage of available cash
            quantity = math.floor(cash * self.PORTFOLIO_RISK / market_price)
        else:
            # Subsequent positions: use all remaining cash
            quantity = math.floor(cash / market_price)

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol={symbol},\n"
                f"    qty={quantity},\n"
                f"    side='buy',\n"
                f"    type='market',\n"
                f"    time_in_force='gtc',\n"
                f"    order_class='bracket',\n"
                f"    stop_loss={{'stop_price': {stop_price}}},\n"
                f"    take_profit={{'limit_price': {take_profit}}}\n"
                f")")

        # Submit the actual order if requested
        if submit_order:
            self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',  # Market order for immediate execution
                time_in_force='gtc',  # Good till cancelled
                order_class='bracket',  # Bracket order with stop loss
                stop_loss={
                    'stop_price': stop_price,  # Triggers a stop order at 10% loss
                },
                take_profit={
                    'limit_price': take_profit
                }
            )


    def _bracketOrder(self, symbol: str, quantity: int, market_price: float, take_profit: float, submit_order: bool = False) -> None:
        """
        Create a bracket order with stop loss protection.

        Args:
            symbol: The stock symbol to trade
            quantity: Number of shares to buy
            market_price: Current market price of the stock
            take_profit: The take profit price for the bracket order
            submit_order: Whether to actually submit the order (default: False)
        """
        stop_price = round(market_price * (1 - self.STOP_LOSS_PERCENT), 2)

        print(f"submit_order(\n"
              f"    symbol={symbol},\n"
              f"    qty={quantity},\n"
              f"    side='buy',\n"
              f"    type='market',\n"
              f"    time_in_force='gtc',\n"
              f"    order_class='bracket',\n"
              f"    stop_loss={{'stop_price': {stop_price}}},\n"
              f"    take_profit={{'limit_price': {take_profit}}}\n"
              f")")

        if submit_order:
            self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',  # or 'limit'
                time_in_force='gtc',
                order_class='bracket',
                stop_loss={
                    'stop_price': stop_price,  # Triggers a stop order
                },
                take_profit={
                    'limit_price': take_profit  # Required for take-profit
                }
            )



    def Exec(self) -> int:
        """
        Execute the main trading logic.

        Prints current positions, cash balance, and active orders.
        If bracket order arguments are provided, executes a bracket order.

        Returns:
            Exit code (0 for success)
        """

        print_positions(self.api)
        print_cash(self.api)
        print_active_orders(self.api)

        # Handle bracket order if requested
        if self.args.bracket_order:
            self._bracketOrder(
                symbol=self.args.symbol,
                quantity=self.args.quantity,
                market_price=self.args.market_price,
                take_profit=self.args.take_profit,
                submit_order=self.args.submit
            )

        # Handle quote request if requested
        if self.args.get_latest_quote:
            print_quote(self.api, self.args.symbol)

        # Handle buy order if requested
        if self.args.buy:
            self._buy(symbol=self.args.symbol, take_profit=self.args.take_profit, submit_order=self.args.submit)

        return 0


def execMain(userArgs: Optional[List[str]] = None) -> int:
    """
    Main execution function for the Alpaca trading script.

    Args:
        userArgs: Optional command line arguments

    Returns:
        Exit code from the trading execution
    """
    # sourcery skip: inline-immediately-returned-variable

    alpacaObj = alpaca_private(userArgs)

    exitValue = alpacaObj.Exec()

    return exitValue

if __name__ == '__main__':
   try:
      retVal = execMain(sys.argv[1:])
   except KeyboardInterrupt:
      print('Received <Ctrl+c>')
      sys.exit(-1)

   sys.exit(retVal)

"""
python3 -m pdb code/alpaca.py
"""


"""
Can use <F5> or <Ctrl+F5> by doing the following:
conda activate alpaca
"""
