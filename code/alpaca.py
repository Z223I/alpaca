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
from atoms.api.get_latest_quote_avg import get_latest_quote_avg
from atoms.api.init_alpaca_client import init_alpaca_client
from atoms.display.print_cash import print_cash
from atoms.display.print_orders import print_active_orders
from atoms.display.print_positions import print_positions
from atoms.display.print_quote import print_quote
from atoms.utils.delay import delay
from atoms.api.parse_args import parse_args

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

    STOP_LOSS_PERCENT = 0.05 # Default stop loss percentage (5.0%)

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


        # Initialize Alpaca API client using atom
        self.api = init_alpaca_client()


        self.active_orders = []


    def _calculateQuantity(self, price: float, method_name: str = "method") -> int:
        """
        Calculate the quantity of shares to buy based on portfolio risk and available cash.

        Args:
            price: The price per share to use for calculation
            method_name: Name of the calling method for logging purposes

        Returns:
            The calculated quantity of shares to buy
        """
        # Get current account information
        cash = get_cash(self.api)
        positions = get_positions(self.api)

        # TODO: Update logic to properly handle different portfolio risk values
        if self.PORTFOLIO_RISK != 0.50:
            print(f"{method_name}() logic must be changed to use the new portfolio risk value")

        # Calculate quantity based on portfolio state
        if not positions:
            # First position: use portfolio risk percentage of available cash
            quantity = math.floor(cash * self.PORTFOLIO_RISK / price)
        else:
            # Subsequent positions: use all remaining cash
            quantity = math.floor(cash / price)

        return quantity

    def _buy(self, symbol: str, take_profit: Optional[float] = None, stop_loss: Optional[float] = None, amount: Optional[float] = None, submit_order: bool = False) -> Optional[Any]:
        """
        Execute a buy order with bracket order protection.

        This method retrieves the latest quote, calculates position size based on
        available cash and existing positions, and submits a bracket order with
        stop loss protection.

        Args:
            symbol: The stock symbol to buy
            take_profit: The take profit price for the bracket order (optional if calc_take_profit is used)
            stop_loss: Custom stop loss price (optional, uses default percentage if not provided)
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)
        """
        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol)

        # Use custom stop loss or calculate default
        if stop_loss is not None:
            stop_price = stop_loss
        else:
            stop_price = round(market_price * (1 - self.STOP_LOSS_PERCENT), 2)

        # Calculate take profit if using calc_take_profit
        if take_profit is None:
            # This means calc_take_profit is being used (validation ensures this)
            take_profit = round(market_price + (market_price - stop_price) * 1.5, 2)

        # Calculate quantity based on amount or use portfolio risk logic
        if amount is not None:
            # Use specified dollar amount to calculate shares
            quantity = round(amount / market_price)
        else:
            # Use existing portfolio risk calculation
            quantity = self._calculateQuantity(market_price, "_buy")

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

        if not submit_order:
            print("[DRY RUN] Order not submitted (use --submit to execute)")
            return None

        # Submit the actual order if requested
        if submit_order:
            try:
                order_response = self.api.submit_order(
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
                print(f"✓ Order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: {order_response.qty}")
                print(f"  Order Class: {order_response.order_class}")
                return order_response
            except Exception as e:
                print(f"✗ Order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}")
                return None

    def _buy_after_hours(self, symbol: str, amount: Optional[float] = None, limit_price: Optional[float] = None, submit_order: bool = False) -> Optional[Any]:
        """
        Execute a buy order for after-hours trading (extended hours).

        After-hours trading has restrictions:
        - Only limit orders are allowed (no market orders)
        - Bracket orders are not supported
        - Must use time_in_force='day' for extended hours

        Args:
            symbol: The stock symbol to buy
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            limit_price: Custom limit price (optional, calculates slightly above market if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)
        """
        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol)

        # Calculate limit price if not provided (slightly above market for better fill probability)
        if limit_price is None:
            limit_price = round(market_price * 1.01, 2)  # 1% above market price

        # Calculate quantity based on amount or use portfolio risk logic
        if amount is not None:
            # Use specified dollar amount to calculate shares
            quantity = round(amount / limit_price)
        else:
            # Use existing portfolio risk calculation
            quantity = self._calculateQuantity(limit_price, "_buy_after_hours")

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol={symbol},\n"
                f"    qty={quantity},\n"
                f"    side='buy',\n"
                f"    type='limit',\n"
                f"    limit_price={limit_price},\n"
                f"    time_in_force='day'\n"
                f")")
        print(f"  NOTE: After-hours order - no bracket/stop-loss protection available")

        if not submit_order:
            print("[DRY RUN] After-hours order not submitted (use --submit to execute)")
            return None

        # Submit the actual order if requested
        if submit_order:
            try:
                order_response = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='limit',  # Only limit orders allowed after-hours
                    limit_price=limit_price,
                    time_in_force='day'  # Good for day including extended hours
                )
                print(f"✓ After-hours order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: {order_response.qty}")
                print(f"  Limit Price: ${limit_price:.2f}")
                print(f"  Market Price: ~${market_price:.2f}")
                print(f"  ⚠️  No automatic stop-loss protection - monitor manually")
                return order_response
            except Exception as e:
                print(f"✗ After-hours order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}, Limit Price: {limit_price}")
                print(f"  Note: Extended hours trading may not be enabled on your account")
                return None

    def _sell_short(self, symbol: str, take_profit: Optional[float] = None, stop_loss: Optional[float] = None, amount: Optional[float] = None, submit_order: bool = False) -> Optional[Any]:
        """
        Execute a short sell order with bracket order protection for bearish predictions.

        This method retrieves the latest quote, calculates position size based on
        available cash and existing positions, and submits a short bracket order with
        stop loss protection above the entry price.

        Args:
            symbol: The stock symbol to short sell
            take_profit: The take profit price for the bracket order (optional if calc_take_profit is used)
            stop_loss: Custom stop loss price (optional, uses default percentage if not provided)
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)
        """
        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol)

        # Use custom stop loss or calculate default (ABOVE entry price for shorts)
        if stop_loss is not None:
            stop_price = stop_loss
        else:
            stop_price = round(market_price * (1 + self.STOP_LOSS_PERCENT), 2)

        # Calculate take profit if using calc_take_profit (BELOW entry price for shorts)
        if take_profit is None:
            # This means calc_take_profit is being used (validation ensures this)
            # For shorts: profit when price falls, so take_profit = entry - (stop_loss - entry) * 1.5
            take_profit = round(market_price - (stop_price - market_price) * 1.5, 2)

        # Calculate quantity based on amount or use portfolio risk logic
        if amount is not None:
            # Use specified dollar amount to calculate shares
            quantity = round(amount / market_price)
        else:
            # Use existing portfolio risk calculation
            quantity = self._calculateQuantity(market_price, "_sell_short")

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol={symbol},\n"
                f"    qty={quantity},\n"
                f"    side='sell',\n"
                f"    type='market',\n"
                f"    time_in_force='gtc',\n"
                f"    order_class='bracket',\n"
                f"    stop_loss={{'stop_price': {stop_price}}},\n"
                f"    take_profit={{'limit_price': {take_profit}}}\n"
                f")")

        if not submit_order:
            print("[DRY RUN] Short order not submitted (use --submit to execute)")
            return None

        # Submit the actual order if requested
        if submit_order:
            try:
                order_response = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',  # Short sell
                    type='market',  # Market order for immediate execution
                    time_in_force='gtc',  # Good till cancelled
                    order_class='bracket',  # Bracket order with stop loss
                    stop_loss={
                        'stop_price': stop_price,  # Triggers stop order ABOVE entry price
                    },
                    take_profit={
                        'limit_price': take_profit  # Take profit BELOW entry price
                    }
                )
                print(f"✓ Short order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: -{order_response.qty}")  # Negative to indicate short
                print(f"  Order Class: {order_response.order_class}")
                print(f"  Entry Price: ~${market_price:.2f}")
                print(f"  Stop Loss: ${stop_price:.2f} (above entry)")
                print(f"  Take Profit: ${take_profit:.2f} (below entry)")
                return order_response
            except Exception as e:
                print(f"✗ Short order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}")
                return None

    def _sell_short_after_hours(self, symbol: str, amount: Optional[float] = None, limit_price: Optional[float] = None, submit_order: bool = False) -> Optional[Any]:
        """
        Execute a short sell order for after-hours trading (extended hours).

        After-hours trading has restrictions:
        - Only limit orders are allowed (no market orders)
        - Bracket orders are not supported
        - Must use time_in_force='day' for extended hours

        Args:
            symbol: The stock symbol to short sell
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            limit_price: Custom limit price (optional, calculates slightly below market if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)
        """
        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol)

        # Calculate limit price if not provided (slightly below market for better fill probability)
        if limit_price is None:
            limit_price = round(market_price * 0.998, 2)  # 0.2% below market price

        # Calculate quantity based on amount or use portfolio risk logic
        if amount is not None:
            # Use specified dollar amount to calculate shares
            quantity = round(amount / limit_price)
        else:
            # Use existing portfolio risk calculation
            quantity = self._calculateQuantity(limit_price, "_sell_short_after_hours")

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol={symbol},\n"
                f"    qty={quantity},\n"
                f"    side='sell',\n"
                f"    type='limit',\n"
                f"    limit_price={limit_price},\n"
                f"    time_in_force='day'\n"
                f")")
        print(f"  NOTE: After-hours short order - no bracket/stop-loss protection available")

        if not submit_order:
            print("[DRY RUN] After-hours short order not submitted (use --submit to execute)")
            return None

        # Submit the actual order if requested
        if submit_order:
            try:
                order_response = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',  # Short sell
                    type='limit',  # Only limit orders allowed after-hours
                    limit_price=limit_price,
                    time_in_force='day'  # Good for day including extended hours
                )
                print(f"✓ After-hours short order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: -{order_response.qty}")  # Negative to indicate short
                print(f"  Limit Price: ${limit_price:.2f}")
                print(f"  Market Price: ~${market_price:.2f}")
                print(f"  ⚠️  No automatic stop-loss protection - monitor manually")
                return order_response
            except Exception as e:
                print(f"✗ After-hours short order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}, Limit Price: {limit_price}")
                print(f"  Note: Extended hours trading may not be enabled on your account")
                return None

    def _submit_after_hours_stop_loss(self, symbol: str, quantity: int, stop_price: float, side: str) -> Optional[Any]:
        """
        Submit a stop-loss order for after-hours trading.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            stop_price: Stop loss trigger price
            side: 'buy' for covering shorts, 'sell' for closing longs
        """
        try:
            stop_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='stop',
                stop_price=stop_price,
                time_in_force='day'
            )
            print(f"  ✓ Stop-loss order submitted: {stop_order.id}")
            print(f"    Stop Price: ${stop_price:.2f}")
            return stop_order
        except Exception as e:
            print(f"  ✗ Stop-loss order failed: {str(e)}")
            return None

    def _submit_after_hours_take_profit(self, symbol: str, quantity: int, limit_price: float, side: str) -> Optional[Any]:
        """
        Submit a take-profit order for after-hours trading.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            limit_price: Take profit limit price
            side: 'sell' for closing longs, 'buy' for covering shorts
        """
        try:
            profit_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='limit',
                limit_price=limit_price,
                time_in_force='day'
            )
            print(f"  ✓ Take-profit order submitted: {profit_order.id}")
            print(f"    Limit Price: ${limit_price:.2f}")
            return profit_order
        except Exception as e:
            print(f"  ✗ Take-profit order failed: {str(e)}")
            return None

    def _buy_after_hours_protected(self, symbol: str, take_profit: Optional[float] = None, stop_loss: Optional[float] = None, amount: Optional[float] = None, limit_price: Optional[float] = None, submit_order: bool = False) -> Optional[Dict]:
        """
        Execute a protected buy order for after-hours trading with separate stop-loss and take-profit orders.

        Since bracket orders aren't allowed after-hours, this method:
        1. Submits the main buy order
        2. If filled, submits separate stop-loss and take-profit orders

        Args:
            symbol: The stock symbol to buy
            take_profit: The take profit price (optional if calc_take_profit is used)
            stop_loss: Custom stop loss price (optional, uses default percentage if not provided)
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            limit_price: Custom limit price (optional, calculates slightly above market if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)
        """
        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol)

        # Calculate limit price if not provided (slightly above market for better fill probability)
        if limit_price is None:
            limit_price = round(market_price * 1.01, 2)  # 1% above market price

        # Use custom stop loss or calculate default (BELOW entry price for longs)
        if stop_loss is not None:
            stop_price = stop_loss
        else:
            stop_price = round(limit_price * (1 - self.STOP_LOSS_PERCENT), 2)

        # Calculate take profit if using calc_take_profit (ABOVE entry price for longs)
        if take_profit is None:
            # This means calc_take_profit is being used (validation ensures this)
            take_profit = round(limit_price + (limit_price - stop_price) * 1.5, 2)

        # Calculate quantity based on amount or use portfolio risk logic
        if amount is not None:
            # Use specified dollar amount to calculate shares
            quantity = round(amount / limit_price)
        else:
            # Use existing portfolio risk calculation
            quantity = self._calculateQuantity(limit_price, "_buy_after_hours_protected")

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol={symbol},\n"
                f"    qty={quantity},\n"
                f"    side='buy',\n"
                f"    type='limit',\n"
                f"    limit_price={limit_price},\n"
                f"    time_in_force='day'\n"
                f")")
        print(f"  Protected after-hours order with stop-loss: ${stop_price:.2f}, take-profit: ${take_profit:.2f}")

        if not submit_order:
            print("[DRY RUN] Protected after-hours order not submitted (use --submit to execute)")
            return None

        # Submit the main buy order
        try:
            main_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='limit',
                limit_price=limit_price,
                time_in_force='day'
            )
            print(f"✓ Main buy order submitted: {main_order.id}")
            print(f"  Status: {main_order.status}")
            print(f"  Limit Price: ${limit_price:.2f}")

            # Submit protection orders
            stop_order = self._submit_after_hours_stop_loss(symbol, quantity, stop_price, 'sell')
            profit_order = self._submit_after_hours_take_profit(symbol, quantity, take_profit, 'sell')

            return {
                'main_order': main_order,
                'stop_loss_order': stop_order,
                'take_profit_order': profit_order,
                'entry_price': limit_price,
                'stop_price': stop_price,
                'take_profit_price': take_profit
            }

        except Exception as e:
            print(f"✗ Protected after-hours buy order failed: {str(e)}")
            print(f"  Symbol: {symbol}, Quantity: {quantity}, Limit Price: {limit_price}")
            return None

    def _sell_short_after_hours_protected(self, symbol: str, take_profit: Optional[float] = None, stop_loss: Optional[float] = None, amount: Optional[float] = None, limit_price: Optional[float] = None, submit_order: bool = False) -> Optional[Dict]:
        """
        Execute a protected short sell order for after-hours trading with separate stop-loss and take-profit orders.

        Since bracket orders aren't allowed after-hours, this method:
        1. Submits the main short sell order
        2. If filled, submits separate stop-loss and take-profit orders

        Args:
            symbol: The stock symbol to short sell
            take_profit: The take profit price (optional if calc_take_profit is used)
            stop_loss: Custom stop loss price (optional, uses default percentage if not provided)
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            limit_price: Custom limit price (optional, calculates slightly below market if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)
        """
        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol)

        # Calculate limit price if not provided (slightly below market for better fill probability)
        if limit_price is None:
            limit_price = round(market_price * 0.998, 2)  # 0.2% below market price

        # Use custom stop loss or calculate default (ABOVE entry price for shorts)
        if stop_loss is not None:
            stop_price = stop_loss
        else:
            stop_price = round(limit_price * (1 + self.STOP_LOSS_PERCENT), 2)

        # Calculate take profit if using calc_take_profit (BELOW entry price for shorts)
        if take_profit is None:
            # This means calc_take_profit is being used (validation ensures this)
            take_profit = round(limit_price - (stop_price - limit_price) * 1.5, 2)

        # Calculate quantity based on amount or use portfolio risk logic
        if amount is not None:
            # Use specified dollar amount to calculate shares
            quantity = round(amount / limit_price)
        else:
            # Use existing portfolio risk calculation
            quantity = self._calculateQuantity(limit_price, "_sell_short_after_hours_protected")

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol={symbol},\n"
                f"    qty={quantity},\n"
                f"    side='sell',\n"
                f"    type='limit',\n"
                f"    limit_price={limit_price},\n"
                f"    time_in_force='day'\n"
                f")")
        print(f"  Protected after-hours short with stop-loss: ${stop_price:.2f}, take-profit: ${take_profit:.2f}")

        if not submit_order:
            print("[DRY RUN] Protected after-hours short order not submitted (use --submit to execute)")
            return None

        # Submit the main short sell order
        try:
            main_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='limit',
                limit_price=limit_price,
                time_in_force='day'
            )
            print(f"✓ Main short order submitted: {main_order.id}")
            print(f"  Status: {main_order.status}")
            print(f"  Limit Price: ${limit_price:.2f}")

            # Submit protection orders (note: sides are reversed for short positions)
            stop_order = self._submit_after_hours_stop_loss(symbol, quantity, stop_price, 'buy')  # Buy to cover
            profit_order = self._submit_after_hours_take_profit(symbol, quantity, take_profit, 'buy')  # Buy to cover

            return {
                'main_order': main_order,
                'stop_loss_order': stop_order,
                'take_profit_order': profit_order,
                'entry_price': limit_price,
                'stop_price': stop_price,
                'take_profit_price': take_profit
            }

        except Exception as e:
            print(f"✗ Protected after-hours short order failed: {str(e)}")
            print(f"  Symbol: {symbol}, Quantity: {quantity}, Limit Price: {limit_price}")
            return None


    def _bracketOrder(self, symbol: str, quantity: int, market_price: float, take_profit: float, submit_order: bool = False) -> Optional[Any]:
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

        if not submit_order:
            print("[DRY RUN] Bracket order not submitted (use --submit to execute)")
            return None

        if submit_order:
            try:
                order_response = self.api.submit_order(
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
                print(f"✓ Bracket order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: {order_response.qty}")
                print(f"  Order Class: {order_response.order_class}")
                return order_response
            except Exception as e:
                print(f"✗ Bracket order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}")
                return None



    def _futureBracketOrder(self, symbol: str, quantity: int, limit_price: float, stop_price: float, take_profit: float, submit_order: bool = False) -> Optional[Any]:
        """
        Create a future bracket order with limit entry and stop loss protection.

        Args:
            symbol: The stock symbol to trade
            quantity: Number of shares to buy (if 0, calculates automatically based on portfolio risk)
            limit_price: The limit price for the entry order
            stop_price: The stop loss price for the bracket order
            take_profit: The take profit price for the bracket order
            submit_order: Whether to actually submit the order (default: False)
        """
        # Calculate quantity if not provided (quantity == 0)
        if quantity == 0:
            quantity = self._calculateQuantity(limit_price, "_futureBracketOrder")

        print(f"submit_order(\n"
              f"    symbol='{symbol}',\n"
              f"    qty={quantity},\n"
              f"    side='buy',\n"
              f"    type='limit',\n"
              f"    time_in_force='day',\n"
              f"    limit_price={limit_price},\n"
              f"    order_class='bracket',\n"
              f"    stop_loss={{'stop_price': {stop_price}}},\n"
              f"    take_profit={{'limit_price': {take_profit}}}\n"
              f")")

        if not submit_order:
            print("[DRY RUN] Future bracket order not submitted (use --submit to execute)")
            return None

        if submit_order:
            try:
                order_response = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=str(limit_price),
                    order_class='bracket',
                    stop_loss={
                        'stop_price': stop_price,
                    },
                    take_profit={
                        'limit_price': take_profit
                    }
                )
                print(f"✓ Future bracket order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: {order_response.qty}")
                print(f"  Limit Price: {limit_price}")
                print(f"  Order Class: {order_response.order_class}")
                return order_response
            except Exception as e:
                print(f"✗ Future bracket order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}, Limit Price: {limit_price}")
                return None

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
            order_result = self._bracketOrder(
                symbol=self.args.symbol,
                quantity=self.args.quantity,
                market_price=self.args.market_price,
                take_profit=self.args.take_profit,
                submit_order=self.args.submit
            )
            if order_result is None and self.args.submit:
                print("Failed to submit bracket order")
                return 1

        # Handle future bracket order if requested
        if self.args.future_bracket_order:
            order_result = self._futureBracketOrder(
                symbol=self.args.symbol,
                quantity=self.args.quantity,
                limit_price=self.args.limit_price,
                stop_price=self.args.stop_price,
                take_profit=self.args.take_profit,
                submit_order=self.args.submit
            )
            if order_result is None and self.args.submit:
                print("Failed to submit future bracket order")
                return 1

        # Handle quote request if requested
        if self.args.get_latest_quote:
            print_quote(self.api, self.args.symbol)

        # Handle buy order if requested
        if self.args.buy:
            if self.args.after_hours:
                # Check if protection is requested
                if self.args.stop_loss or self.args.take_profit or self.args.calc_take_profit:
                    # Use protected after-hours method
                    order_result = self._buy_after_hours_protected(
                        symbol=self.args.symbol,
                        take_profit=self.args.take_profit,
                        stop_loss=self.args.stop_loss,
                        amount=self.args.amount,
                        limit_price=self.args.custom_limit_price,
                        submit_order=self.args.submit
                    )
                else:
                    # Use simple after-hours method
                    order_result = self._buy_after_hours(
                        symbol=self.args.symbol,
                        amount=self.args.amount,
                        limit_price=self.args.custom_limit_price,
                        submit_order=self.args.submit
                    )
                if order_result is None and self.args.submit:
                    print("Failed to submit after-hours buy order")
                    return 1
            else:
                # Use regular buy method with bracket order protection
                order_result = self._buy(
                    symbol=self.args.symbol,
                    take_profit=self.args.take_profit,
                    stop_loss=self.args.stop_loss,
                    amount=self.args.amount,
                    submit_order=self.args.submit
                )
                if order_result is None and self.args.submit:
                    print("Failed to submit buy order")
                    return 1

        # Handle sell-short order if requested
        if self.args.sell_short:
            if self.args.after_hours:
                # Check if protection is requested
                if self.args.stop_loss or self.args.take_profit or self.args.calc_take_profit:
                    # Use protected after-hours method
                    order_result = self._sell_short_after_hours_protected(
                        symbol=self.args.symbol,
                        take_profit=self.args.take_profit,
                        stop_loss=self.args.stop_loss,
                        amount=self.args.amount,
                        limit_price=self.args.custom_limit_price,
                        submit_order=self.args.submit
                    )
                else:
                    # Use simple after-hours method
                    order_result = self._sell_short_after_hours(
                        symbol=self.args.symbol,
                        amount=self.args.amount,
                        limit_price=self.args.custom_limit_price,
                        submit_order=self.args.submit
                    )
                if order_result is None and self.args.submit:
                    print("Failed to submit after-hours short order")
                    return 1
            else:
                # Use regular short method with bracket order protection
                order_result = self._sell_short(
                    symbol=self.args.symbol,
                    take_profit=self.args.take_profit,
                    stop_loss=self.args.stop_loss,
                    amount=self.args.amount,
                    submit_order=self.args.submit
                )
                if order_result is None and self.args.submit:
                    print("Failed to submit short order")
                    return 1

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
