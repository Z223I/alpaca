#import requests
import json
import math
import time
import sys
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

import alpaca_trade_api as tradeapi   # pip3 install alpaca-trade-api -U
import argparse

import parseArgs

# Load environment variables from .env file
load_dotenv()


class alpaca_private:
    """
    Alpaca trading API wrapper for automated trading operations.
    
    This class provides methods for interacting with the Alpaca trading API,
    including order management, position tracking, and bracket order execution.
    """

    RISK = 0.10  # 10% risk constant

    def __init__(self, userArgs: Optional[List[str]] = None) -> None:
        """
        Initialize the Alpaca trading client.
        
        Args:
            userArgs: Optional command line arguments for configuration
        """
        self.history = {}

        # Parse arguments
        self.args = self._parse_args(userArgs)

        self.key = os.getenv('ALPACA_API_KEY')
        self.secret = os.getenv('ALPACA_SECRET_KEY')
        self.headers = {'APCA-API-KEY-ID':self.key, 'APCA-API-SECRET-KEY':self.secret}

        self.baseURL = os.getenv('ALPACA_BASE_URL')
        self.accountURL = "{}/v2/account".format(self.baseURL)
        self.ordersURL = "{}/v2/orders".format(self.baseURL)

        self.api =  tradeapi.REST(self.key, self.secret, self.baseURL)
        self.active_orders = []

    def _parse_args(self, userArgs: Optional[List[str]]) -> argparse.Namespace:
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

        args = parser.parse_args(userArgs)

        # Validate bracket order arguments
        if args.bracket_order:
            if not all([args.symbol, args.quantity, args.market_price]):
                parser.error("--bracket_order requires --symbol, --quantity, and --market_price")

        return args

    def delay(self) -> None:
        """
        Wait until all active orders are completed.
        
        Continuously polls for active orders and sleeps until none remain.
        """
        while len(self.getActiveOrders()) > 0:
            time.sleep(1)


    def getActiveOrders(self) -> List[Any]:
        """
        Retrieve all active (open) orders from Alpaca.
        
        Returns:
            List of active order objects, or empty dict if an error occurs
        """
        try:
            return self.api.list_orders(
                status='open',
                limit=100)
        except:
            return {}

    def printActiveOrders(self) -> None:
        """
        Print details of all active orders to console.
        
        Displays order information including symbol, quantity, side, status,
        filled quantity, remaining quantity, and timestamps.
        """
        orders = self.getActiveOrders()

        if orders:
            print(f"orders: {orders}")

            for order in orders:
                print(f"order: {order}")
                print(f"symbol: {order.symbol}")
                print(f"qty: {order.qty}")
                print(f"side: {order.side}")
                print(f"status: {order.status}")
                print(f"filled_qty: {order.filled_qty}")
                print(f"remaining_qty: {order.remaining_qty}")
                print(f"created_at: {order.created_at}")
                print(f"updated_at: {order.updated_at}")
        else:
            print("No current orders")

    def getCash(self) -> float:
        """
        Get the current cash balance in the trading account.
        
        Returns:
            Current cash balance as a float
        """
        return self.api.get_account().cash

    def printCash(self) -> None:
        """
        Print the current cash balance to console.
        """
        cash = self.getCash()
        print(f"cash: {cash}")

    def _getPositions(self) -> List[Any]:
        """
        Get all current positions in the trading account.
        
        Returns:
            List of position objects from Alpaca API
        """
        return self.api.list_positions()

    def printPositions(self) -> None:
        """
        Print all current positions to console.
        """
        positions = self._getPositions()
        print(f"positions: {positions}")

    def _bracketOrder(self, symbol: str, quantity: int, market_price: float, submit_order: bool = False) -> None:
        """
        Create a bracket order with stop loss protection.

        Args:
            symbol: The stock symbol to trade
            quantity: Number of shares to buy
            market_price: Current market price of the stock
            submit_order: Whether to actually submit the order (default: False)
        """
        stop_price = market_price * (1 - self.RISK)

        print(f"submit_order(\n"
              f"    symbol={symbol},\n"
              f"    qty={quantity},\n"
              f"    side='buy',\n"
              f"    type='market',\n"
              f"    time_in_force='gtc',\n"
              f"    order_class='bracket',\n"
              f"    stop_loss={{'stop_price': {stop_price}}}\n"
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
                }
                #,
                # take_profit={
                #     'limit_price': 160.00  # Required for take-profit
                # }
            )



    def Exec(self) -> int:
        """
        Execute the main trading logic.
        
        Prints current positions, cash balance, and active orders.
        If bracket order arguments are provided, executes a bracket order.
        
        Returns:
            Exit code (0 for success)
        """

        self.printPositions()
        self.printCash()
        self.printActiveOrders()

        # Handle bracket order if requested
        if self.args.bracket_order:
            self._bracketOrder(
                symbol=self.args.symbol,
                quantity=self.args.quantity,
                market_price=self.args.market_price,
                submit_order=self.args.submit
            )

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
