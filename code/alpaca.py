# import requests
# import json
# import math
# import time
import sys
import os
import math
import time
import subprocess
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date, time as dt_time
import pytz
import pandas as pd
import json
import glob

import alpaca_trade_api as tradeapi   # pip3 install alpaca-trade-api -U

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.api.get_cash import get_cash
# from atoms.api.get_active_orders import get_active_orders
from atoms.api.get_positions import get_positions
from atoms.api.get_latest_quote import get_latest_quote
from atoms.api.get_latest_quote_avg import get_latest_quote_avg
from atoms.api.init_alpaca_client import init_alpaca_client
from atoms.api.config import TradingConfig
from atoms.api.pnl import AlpacaDailyPnL
from atoms.display.print_cash import print_cash
from atoms.display.print_orders import print_active_orders
from atoms.display.print_positions import print_positions
from atoms.display.print_quote import print_quote
from atoms.display.generate_chart_from_df import generate_chart_from_dataframe
from atoms.utils.macd_alert_scorer import score_alerts_with_macd, MACDAlertScorer
from atoms.utils.calculate_macd import calculate_macd
from atoms.utils.delay import delay
from atoms.api.parse_args import parse_args
from atoms.telegram.telegram_post import TelegramPoster

# Load configuration from config file (with fallback to environment variables)
try:
    from alpaca_config import get_current_config, get_api_credentials
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Load environment variables from .env file (fallback)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Using system environment variables only.")


class AlpacaPrivate:
    """
    Alpaca trading API wrapper for automated trading operations.

    This class provides methods for interacting with the Alpaca trading API,
    including order management, position tracking, and bracket order execution.
    """

    STOP_LOSS_PERCENT = 0.05  # Default stop loss percentage (5.0%)

    def __init__(self, userArgs: Optional[List[str]] = None) -> None:
        """
        Initialize the Alpaca trading client.

        Args:
            userArgs: Optional command line arguments for configuration
        """
        self.history = {}

        # Parse arguments
        self.args = parse_args(userArgs)
        
        # Set account configuration from command line arguments
        self.account_name = self.args.account_name
        self.account = self.args.account

        # Set portfolio risk from config file or environment variable as fallback
        if 'CONFIG_AVAILABLE' in globals() and CONFIG_AVAILABLE:
            try:
                config = get_current_config()
                self.PORTFOLIO_RISK = config.portfolio_risk
            except Exception as e:
                print(f"Warning: Could not load portfolio risk from config ({e}), using environment variable")
                self.PORTFOLIO_RISK = float(os.getenv('PORTFOLIO_RISK', '0.10'))
        else:
            self.PORTFOLIO_RISK = float(os.getenv('PORTFOLIO_RISK', '0.10'))

        # Initialize Alpaca API client using account configuration
        self.api = init_alpaca_client("alpaca", self.account_name, self.account)

        # Setup logging for buy-market-trailing-sell-take-profit-percent operations
        self._setup_logging()

        self.active_orders = []
        
        # First trade tracking and position monitoring
        self._first_trade_completed = False
        self._monitoring_process = None
        self._monitoring_started_today = False
        self._today_date = datetime.now(pytz.timezone('America/New_York')).date()

    def _setup_logging(self) -> None:
        """
        Setup logging configuration for buy-market-trailing-sell-take-profit-percent operations.
        Creates timestamped log files in logs/alpaca directory.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/alpaca/trading_operations_{timestamp}.log"
        
        # Create logger for this instance
        self.logger = logging.getLogger(f'alpaca_trading_{timestamp}')
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if logger already exists
        if not self.logger.handlers:
            # Create file handler
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
        
        self.logger.info("=" * 80)
        self.logger.info("NEW TRADING SESSION STARTED")
        self.logger.info(f"Account: {getattr(self, 'account_name', 'Unknown')}")
        self.logger.info(f"Account Type: {getattr(self, 'account', 'Unknown')}")
        self.logger.info(f"Portfolio Risk: {getattr(self, 'PORTFOLIO_RISK', 'Unknown')}")
        self.logger.info("=" * 80)

    def _log_and_print(self, message: str, level: str = 'INFO') -> None:
        """
        Log message to file and print to console.
        
        Args:
            message: Message to log and print
            level: Log level (INFO, WARNING, ERROR)
        """
        print(message)
        if hasattr(self, 'logger'):
            if level == 'WARNING':
                self.logger.warning(message)
            elif level == 'ERROR':
                self.logger.error(message)
            else:
                self.logger.info(message)

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
        cash = get_cash(self.api, self.account_name, self.account)
        positions = get_positions(self.api, self.account_name, self.account)

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

    def _buy_market(self, symbol: str, amount: Optional[float] = None, submit_order: bool = False) -> Optional[Any]:
        """
        Execute a simple market buy order without bracket order protection.

        This method retrieves the latest quote, calculates position size based on
        available cash and existing positions, and submits a market order for
        immediate execution at current market price.

        Args:
            symbol: The stock symbol to buy
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)

        Returns:
            The order response from Alpaca API or None if dry run/error
        """
        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

        # Calculate quantity based on amount or use portfolio risk logic
        if amount is not None:
            # Use specified dollar amount to calculate shares
            quantity = round(amount / market_price)
        else:
            # Use existing portfolio risk calculation
            quantity = self._calculateQuantity(market_price, "_buy_market")

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol='{symbol}',\n"
                f"    qty={quantity},\n"
                f"    side='buy',\n"
                f"    type='market',\n"
                f"    time_in_force='day'\n"
                f")")

        if not submit_order:
            print("[DRY RUN] Market buy order not submitted (use --submit to execute)")
            return None

        # Submit the actual order if requested
        if submit_order:
            try:
                order_response = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print(f"✓ Market buy order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: {order_response.qty}")
                print(f"  Market Price: ~${market_price:.2f}")
                # Trigger position monitoring on successful trade
                self._onTradeExecuted(order_response, f"Market buy: {symbol}")
                return order_response
            except Exception as e:
                print(f"✗ Market buy order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}")
                return None

    def _poll_order_status(self, order_id: str, timeout_seconds: int = 60) -> Optional[Any]:
        """
        Poll order status until it reaches a terminal state or timeout.

        Args:
            order_id: The order ID to monitor
            timeout_seconds: Maximum time to wait (default: 60 seconds)

        Returns:
            The final order object or None if timeout/error
        """
        import time
        start_time = time.time()
        poll_interval = 2  # Poll every 2 seconds

        print(f"Polling order status for {order_id}...")

        while (time.time() - start_time) < timeout_seconds:
            try:
                order = self.api.get_order(order_id)
                current_status = order.status
                print(f"  Order status: {current_status}")

                # Check for terminal states
                if current_status in ['filled', 'canceled', 'rejected', 'expired']:
                    print(f"✓ Order reached terminal state: {current_status}")
                    return order

                # Continue polling for non-terminal states
                if current_status in ['new', 'accepted', 'partially_filled', 'pending_new', 'calculated', 'stopped', 'suspended']:
                    time.sleep(poll_interval)
                    continue
                else:
                    print(f"⚠️  Unknown order status: {current_status}")
                    time.sleep(poll_interval)
                    continue

            except Exception as e:
                print(f"✗ Error polling order status: {str(e)}")
                return None

        print(f"⚠️  Order polling timeout after {timeout_seconds} seconds")
        return None

    def _buy_market_trailing_sell(self, symbol: str, amount: Optional[float] = None, trailing_percent: Optional[float] = None, submit_order: bool = False) -> Optional[Dict]:
        """
        Execute market buy order followed by automatic trailing sell when filled.

        This method:
        1. Executes a market buy order
        2. Polls the order status until filled or canceled
        3. If filled, automatically places a trailing sell order for the filled quantity

        Args:
            symbol: The stock symbol to trade
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            trailing_percent: Trailing percentage for sell order (optional, uses default if not provided)
            submit_order: Whether to actually submit orders (default: False for dry run)

        Returns:
            Dictionary with buy and sell order responses, or None if error/dry run
        """
        print(f"Executing market buy with trailing sell for {symbol}...")

        # Step 1: Execute market buy order
        buy_order = self._buy_market(symbol=symbol, amount=amount, submit_order=submit_order)

        if not submit_order:
            print("[DRY RUN] Would poll order status and place trailing sell after fill")
            return None

        if buy_order is None:
            print("✗ Market buy order failed, aborting trailing sell setup")
            return None

        # Step 2: Poll order status until filled or terminal state
        final_order = self._poll_order_status(buy_order.id)

        if final_order is None:
            print("✗ Order polling failed, cannot proceed with trailing sell")
            return None

        if final_order.status != 'filled':
            print(f"✗ Order not filled (status: {final_order.status}), cannot place trailing sell")
            return None

        # Step 3: Extract filled quantity and place trailing sell
        filled_qty = int(final_order.filled_qty) if hasattr(final_order, 'filled_qty') else int(final_order.qty)
        print(f"✓ Buy order filled: {filled_qty} shares")

        # Execute trailing sell with the filled quantity
        sell_order = self._sell_trailing(
            symbol=symbol, 
            quantity=filled_qty, 
            trailing_percent=trailing_percent,
            submit_order=True  # Always submit if we got this far
        )

        if sell_order is None:
            print("✗ Trailing sell order failed")
            return {
                'buy_order': final_order,
                'sell_order': None,
                'error': 'Trailing sell failed'
            }

        print("✓ Market buy with trailing sell completed successfully")
        return {
            'buy_order': final_order,
            'sell_order': sell_order
        }

    def _buy_market_trailing_sell_take_profit_percent(self, symbol: str, take_profit_percent: float, amount: Optional[float] = None, trailing_percent: Optional[float] = None, submit_order: bool = False) -> Optional[Dict]:
        """
        Execute market buy order followed by automatic trailing sell and take profit percent order when filled.

        This method:
        1. Executes a market buy order
        2. Polls the order status until filled or canceled
        3. If filled, automatically places a trailing sell order for the filled quantity
        4. Also places a take profit percent order using the filled quantity and average fill price

        Args:
            symbol: The stock symbol to trade
            take_profit_percent: Percentage above average fill price for take profit order
            amount: Dollar amount to invest (optional, uses portfolio risk if not provided)
            trailing_percent: Trailing percentage for sell order (optional, uses default if not provided)
            submit_order: Whether to actually submit orders (default: False for dry run)

        Returns:
            Dictionary with buy, trailing sell, and take profit order responses, or None if error/dry run
        """
        self._log_and_print(f"🚀 STEP 1: Executing market buy with trailing sell and take profit percent for {symbol}...")

        # Step 1: Execute market buy order
        buy_order = self._buy_market(symbol=symbol, amount=amount, submit_order=submit_order)

        if not submit_order:
            self._log_and_print("[DRY RUN] Would poll order status and place trailing sell + take profit percent after fill")
            # In dry run mode, simulate the take profit calculation with estimated quantity
            # Get estimated quantity from the buy order simulation
            market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)
            if amount is not None:
                estimated_qty = round(amount / market_price)
            else:
                estimated_qty = self._calculateQuantity(market_price, "_buy_market_trailing_sell_take_profit_percent")
            
            self._log_and_print(f"\n[DRY RUN] Simulating complete workflow:")
            self._log_and_print(f"  1. Buy Order: Would fill {estimated_qty} shares @ ${market_price:.2f}")
            self._log_and_print(f"  2. Trailing Sell: Would place trailing sell order (default {trailing_percent or 7.5}%)")
            self._log_and_print(f"  3. Trailing Sell Polling: Would monitor for acceptance/rejection")
            self._log_and_print(f"  4. Take Profit: Would place take profit order at {take_profit_percent}% above fill price")
            self._log_and_print(f"  5. Take Profit Polling: Would monitor for acceptance/rejection")
            
            # Show what the take profit order would look like
            self._log_and_print(f"\n[DRY RUN] Take profit order simulation:")
            self._take_profit_percent(
                symbol=symbol,
                quantity=estimated_qty,
                take_profit_percent=take_profit_percent,
                submit_order=False,
                current_price=market_price
            )
            return None

        if buy_order is None:
            self._log_and_print("✗ Market buy order failed, aborting trailing sell and take profit setup", 'ERROR')
            return None

        # Step 2: Poll order status until filled or terminal state
        final_order = self._poll_order_status(buy_order.id)

        if final_order is None:
            self._log_and_print("✗ STEP 2: Order polling failed, cannot proceed with trailing sell and take profit", 'ERROR')
            return None

        if final_order.status != 'filled':
            self._log_and_print(f"✗ STEP 2: Order not filled (status: {final_order.status}), cannot place trailing sell and take profit", 'ERROR')
            return None

        # Step 3: Extract filled quantity and average fill price
        filled_qty = int(final_order.filled_qty) if hasattr(final_order, 'filled_qty') else int(final_order.qty)
        filled_avg_price = float(final_order.filled_avg_price) if hasattr(final_order, 'filled_avg_price') and final_order.filled_avg_price else None
        
        self._log_and_print(f"✅ STEP 2: Buy order filled: {filled_qty} shares")
        if filled_avg_price:
            self._log_and_print(f"  Average fill price: ${filled_avg_price:.2f}")
        else:
            self._log_and_print("  Warning: No average fill price available, using current market price", 'WARNING')

        # Step 4: Execute trailing sell with the filled quantity
        self._log_and_print(f"🚀 STEP 3: Executing trailing sell order for {filled_qty} shares...")
        sell_order = self._sell_trailing(
            symbol=symbol, 
            quantity=filled_qty, 
            trailing_percent=trailing_percent,
            submit_order=True  # Always submit if we got this far
        )

        if sell_order is None:
            self._log_and_print("✗ STEP 3: Trailing sell order failed", 'ERROR')
            return {
                'buy_order': final_order,
                'sell_order': None,
                'take_profit_order': None,
                'error': 'Trailing sell failed'
            }

        # Step 5: Poll trailing sell order status
        self._log_and_print(f"\n🔍 STEP 3: Polling Trailing Sell Order Status ---")
        final_sell_order = self._poll_order_status(sell_order.id, timeout_seconds=30)
        
        if final_sell_order is None:
            self._log_and_print("✗ STEP 3: Trailing sell order polling failed or timeout", 'ERROR')
            sell_status = "polling_failed"
        elif final_sell_order.status in ['canceled', 'rejected', 'expired']:
            self._log_and_print(f"✗ STEP 3: Trailing sell order failed with status: {final_sell_order.status}", 'ERROR')
            sell_status = "failed"
        else:
            self._log_and_print(f"✅ STEP 3: Trailing sell order successfully placed with status: {final_sell_order.status}")
            sell_status = "success"

        # Step 6: Execute take profit percent order
        self._log_and_print(f"🚀 STEP 4: Executing take profit order for {filled_qty} shares at {take_profit_percent}% above fill price...")
        take_profit_order = self._take_profit_percent(
            symbol=symbol,
            quantity=filled_qty,
            take_profit_percent=take_profit_percent,
            submit_order=True,  # Always submit if we got this far
            current_price=filled_avg_price  # Use average fill price if available
        )

        if take_profit_order is None:
            self._log_and_print("✗ STEP 4: Take profit percent order failed", 'ERROR')
            return {
                'buy_order': final_order,
                'sell_order': final_sell_order or sell_order,
                'sell_status': sell_status,
                'take_profit_order': None,
                'take_profit_status': "submission_failed",
                'error': 'Take profit percent failed'
            }

        # Step 7: Poll take profit order status
        self._log_and_print(f"\n🔍 STEP 4: Polling Take Profit Order Status ---")
        final_take_profit_order = self._poll_order_status(take_profit_order.id, timeout_seconds=30)
        
        if final_take_profit_order is None:
            self._log_and_print("✗ STEP 4: Take profit order polling failed or timeout", 'ERROR')
            take_profit_status = "polling_failed"
        elif final_take_profit_order.status in ['canceled', 'rejected', 'expired']:
            self._log_and_print(f"✗ STEP 4: Take profit order failed with status: {final_take_profit_order.status}", 'ERROR')
            take_profit_status = "failed"
        else:
            self._log_and_print(f"✅ STEP 4: Take profit order successfully placed with status: {final_take_profit_order.status}")
            take_profit_status = "success"

        # Step 8: Summary and results
        self._log_and_print(f"\n📊 === ORDER EXECUTION SUMMARY ===")
        self._log_and_print(f"✅ Buy Order: FILLED ({filled_qty} shares @ ${filled_avg_price:.2f})")
        self._log_and_print(f"{'✅' if sell_status == 'success' else '❌'} Trailing Sell: {sell_status.upper()}")
        self._log_and_print(f"{'✅' if take_profit_status == 'success' else '❌'} Take Profit: {take_profit_status.upper()}")
        
        overall_success = sell_status == "success" and take_profit_status == "success"
        if overall_success:
            self._log_and_print("✅ ALL ORDERS COMPLETED SUCCESSFULLY!")
        else:
            self._log_and_print("⚠️  Some orders encountered issues - monitor positions carefully", 'WARNING')

        return {
            'buy_order': final_order,
            'sell_order': final_sell_order or sell_order,
            'sell_status': sell_status,
            'take_profit_order': final_take_profit_order or take_profit_order,
            'take_profit_status': take_profit_status,
            'overall_success': overall_success
        }

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
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

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

    def _sell_trailing(self, symbol: str, quantity: int, trailing_percent: Optional[float] = None, submit_order: bool = False) -> Optional[Any]:
        """
        Execute a trailing sell order.

        This method creates a trailing stop sell order that will follow the stock price
        upward at a specified percentage distance, and execute a sell when the price
        starts moving down by the trailing percentage.

        Args:
            symbol: The stock symbol to sell
            quantity: Number of shares to sell (required)
            trailing_percent: Trailing percentage (optional, uses default from config if not provided)
            submit_order: Whether to actually submit the order (default: False for dry run)

        Returns:
            The order response from Alpaca API or None if dry run/error
        """
        # Use default trailing percent from config if not provided
        if trailing_percent is None:
            trailing_percent = TradingConfig.DEFAULT_TRAILING_PERCENT

        # Get current market data for the symbol (using average of bid/ask)
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol='{symbol}',\n"
                f"    qty={quantity},\n"
                f"    side='sell',\n"
                f"    type='trailing_stop',\n"
                f"    time_in_force='gtc',\n"
                f"    trail_percent='{trailing_percent}'\n"
                f")")

        if not submit_order:
            print("[DRY RUN] Trailing sell order not submitted (use --submit to execute)")
            return None

        # Submit the actual order if requested
        if submit_order:
            try:
                order_response = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='trailing_stop',
                    time_in_force='gtc',
                    trail_percent=str(trailing_percent)
                )
                print(f"✓ Trailing sell order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: {order_response.qty}")
                print(f"  Trail Percent: {trailing_percent}%")
                print(f"  Current Market Price: ~${market_price:.2f}")
                return order_response
            except Exception as e:
                print(f"✗ Trailing sell order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}, Trail Percent: {trailing_percent}%")
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
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

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
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

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
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

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
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

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
        market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)

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

    def _liquidate_position(self, symbol: str, submit_order: bool = False) -> Optional[Any]:
        """
        Liquidate a specific position and cancel related orders.

        This method:
        1. Cancels all open orders for the symbol
        2. Closes the position using Alpaca's close_position API

        Args:
            symbol: The stock symbol to liquidate
            submit_order: Whether to actually execute the liquidation (default: False for dry run)

        Returns:
            The order response from the liquidation or None if dry run/error
        """
        print(f"Liquidating position for {symbol}...")

        # First, check if position exists
        try:
            positions = self.api.list_positions()
            position = None
            for pos in positions:
                if pos.symbol == symbol:
                    position = pos
                    break

            if position is None:
                print(f"✗ No position found for symbol {symbol}")
                return None

            print(f"  Current position: {position.qty} shares @ ${position.avg_entry_price}")
            print(f"  Market value: ${position.market_value}")
            print(f"  Unrealized P&L: ${position.unrealized_pl}")

        except Exception as e:
            print(f"✗ Error checking position for {symbol}: {str(e)}")
            return None

        # Cancel all open orders for this symbol
        try:
            orders = self.api.list_orders(status="open")
            symbol_orders = [order for order in orders if order.symbol == symbol]

            if symbol_orders:
                print(f"  Cancelling {len(symbol_orders)} open orders for {symbol}...")
                for order in symbol_orders:
                    if submit_order:
                        self.api.cancel_order(order.id)
                        print(f"    ✓ Cancelled order {order.id} ({order.side} {order.qty} @ {order.order_type})")
                    else:
                        print(f"    [DRY RUN] Would cancel order {order.id} ({order.side} {order.qty} @ {order.order_type})")
            else:
                print(f"  No open orders found for {symbol}")

        except Exception as e:
            print(f"  ⚠️  Error cancelling orders for {symbol}: {str(e)}")

        # Close the position
        if not submit_order:
            print(f"[DRY RUN] Would liquidate position for {symbol} (use --submit to execute)")
            return None

        try:
            # Use the older alpaca-trade-api close_position method
            liquidation_order = self.api.close_position(symbol)
            print(f"✓ Position liquidated successfully for {symbol}")
            print(f"  Order ID: {liquidation_order.id}")
            print(f"  Status: {liquidation_order.status}")
            print(f"  Quantity: {liquidation_order.qty}")
            print(f"  Side: {liquidation_order.side}")
            return liquidation_order

        except Exception as e:
            print(f"✗ Failed to liquidate position for {symbol}: {str(e)}")
            return None

    def _liquidate_all(self, cancel_orders: bool = False, submit_order: bool = False) -> Optional[List[Any]]:
        """
        Liquidate all open positions and optionally cancel all orders.

        This method:
        1. Optionally cancels all open orders
        2. Closes all positions using Alpaca's close_all_positions API

        Args:
            cancel_orders: Whether to cancel all open orders first
            submit_order: Whether to actually execute the liquidation (default: False for dry run)

        Returns:
            List of order responses from liquidations or None if dry run/error
        """
        print("Liquidating all positions...")

        # Check current positions
        try:
            positions = self.api.list_positions()
            if not positions:
                print("  No open positions to liquidate")
                return []

            print(f"  Found {len(positions)} positions to liquidate:")
            total_value = 0
            for pos in positions:
                print(f"    {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price} (${pos.market_value})")
                total_value += float(pos.market_value)
            print(f"  Total portfolio value: ${total_value:.2f}")

        except Exception as e:
            print(f"✗ Error checking positions: {str(e)}")
            return None

        # Cancel all orders if requested
        if cancel_orders:
            try:
                orders = self.api.list_orders(status="open")
                if orders:
                    print(f"  Cancelling {len(orders)} open orders...")
                    for order in orders:
                        if submit_order:
                            self.api.cancel_order(order.id)
                            print(f"    ✓ Cancelled order {order.id} ({order.symbol} {order.side} {order.qty})")
                        else:
                            print(f"    [DRY RUN] Would cancel order {order.id} ({order.symbol} {order.side} {order.qty})")
                else:
                    print("  No open orders to cancel")

            except Exception as e:
                print(f"  ⚠️  Error cancelling orders: {str(e)}")

        # Close all positions
        if not submit_order:
            print("[DRY RUN] Would liquidate all positions (use --submit to execute)")
            return None

        try:
            # Use the older alpaca-trade-api close_all_positions method
            liquidation_orders = self.api.close_all_positions()
            print(f"✓ All positions liquidated successfully")
            print(f"  Generated {len(liquidation_orders)} liquidation orders")

            for order in liquidation_orders:
                print(f"    {order.symbol}: {order.side} {order.qty} shares (Order ID: {order.id})")

            return liquidation_orders

        except Exception as e:
            print(f"✗ Failed to liquidate all positions: {str(e)}")
            return None

    def _take_profit_percent(self, symbol: str, quantity: int, take_profit_percent: float, submit_order: bool = False, current_price: Optional[float] = None) -> Optional[Any]:
        """
        Execute a take profit order at a percentage above the current market price.

        This method creates a limit sell order at a calculated price based on the 
        current market price plus the specified percentage.

        Args:
            symbol: The stock symbol to sell
            quantity: Number of shares to sell (required)
            take_profit_percent: Percentage above current market price for take profit
            submit_order: Whether to actually submit the order (default: False for dry run)
            current_price: Optional current price to use for calculation (if None, fetches from market)

        Returns:
            The order response from Alpaca API or None if dry run/error
        """
        # Use provided current price or get current market data for the symbol
        if current_price is not None:
            market_price = current_price
            print(f"  Using provided current price: ${market_price:.2f}")
        else:
            market_price = get_latest_quote_avg(self.api, symbol, self.account_name, self.account)
            print(f"  Fetched current market price: ${market_price:.2f}")

        # Calculate take profit price based on percentage
        take_profit_price = round(market_price * (1 + take_profit_percent / 100), 2)

        # Display the order details that would be submitted
        print(f"submit_order(\n"
                f"    symbol='{symbol}',\n"
                f"    qty={quantity},\n"
                f"    side='sell',\n"
                f"    type='limit',\n"
                f"    time_in_force='gtc',\n"
                f"    limit_price={take_profit_price}\n"
                f")")
        print(f"  Take Profit Percent: {take_profit_percent}%")
        print(f"  Take Profit Price: ${take_profit_price:.2f}")

        if not submit_order:
            print("[DRY RUN] Take profit order not submitted (use --submit to execute)")
            return None

        # Submit the actual order if requested
        if submit_order:
            try:
                order_response = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='limit',
                    time_in_force='gtc',
                    limit_price=str(take_profit_price)
                )
                print(f"✓ Take profit order submitted successfully: {order_response.id}")
                print(f"  Status: {order_response.status}")
                print(f"  Symbol: {order_response.symbol}")
                print(f"  Quantity: {order_response.qty}")
                print(f"  Limit Price: ${take_profit_price:.2f}")
                print(f"  Current Market Price: ${market_price:.2f}")
                return order_response
            except Exception as e:
                print(f"✗ Take profit order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}, Take Profit Price: {take_profit_price}")
                return None

    def _cancel_all_orders(self, submit_order: bool = False) -> Optional[List[Any]]:
        """
        Cancel all open orders.

        This method lists all open orders and cancels them individually,
        since the legacy alpaca-trade-api library doesn't have a direct
        cancel all orders method.

        Args:
            submit_order: Whether to actually cancel the orders (default: False for dry run)

        Returns:
            List of cancelled order IDs or None if dry run/error
        """
        print("Cancelling all open orders...")

        # Get all open orders
        try:
            orders = self.api.list_orders(status="open")
            if not orders:
                print("  No open orders to cancel")
                return []

            print(f"  Found {len(orders)} open orders to cancel:")
            for order in orders:
                print(f"    {order.id}: {order.symbol} {order.side} {order.qty} @ {order.order_type} ({order.status})")

        except Exception as e:
            print(f"✗ Error retrieving open orders: {str(e)}")
            return None

        if not submit_order:
            print("[DRY RUN] Orders not cancelled (use --submit to execute)")
            return None

        # Cancel all orders
        cancelled_orders = []
        failed_cancellations = []

        for order in orders:
            try:
                self.api.cancel_order(order.id)
                cancelled_orders.append(order.id)
                print(f"  ✓ Cancelled order {order.id} ({order.symbol} {order.side} {order.qty})")
            except Exception as e:
                failed_cancellations.append((order.id, str(e)))
                print(f"  ✗ Failed to cancel order {order.id}: {str(e)}")

        # Summary
        print(f"\n✓ Successfully cancelled {len(cancelled_orders)} orders")
        if failed_cancellations:
            print(f"✗ Failed to cancel {len(failed_cancellations)} orders:")
            for order_id, error in failed_cancellations:
                print(f"    {order_id}: {error}")

        return cancelled_orders

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
                # Trigger position monitoring on successful trade
                self._onTradeExecuted(order_response, f"Bracket order: {symbol}")
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
                print(f"  Limit Price: ${limit_price:.2f}")
                print(f"  Order Class: {order_response.order_class}")
                # Trigger position monitoring on successful trade
                self._onTradeExecuted(order_response, f"Future bracket order: {symbol}")
                return order_response
            except Exception as e:
                print(f"✗ Future bracket order submission failed: {str(e)}")
                print(f"  Symbol: {symbol}, Quantity: {quantity}, Limit Price: {limit_price}")
                return None

    def _load_superduper_alerts_for_symbol(self, symbol: str, target_date: date) -> List[Dict[str, Any]]:
        """
        Load sent superduper alerts for a specific symbol and date.
        
        Args:
            symbol: Stock symbol to load alerts for
            target_date: Date to load alerts for
            
        Returns:
            List of alert dictionaries with timestamp_dt, alert_type, and alert_level
        """
        alerts = []
        
        try:
            # Format date as YYYY-MM-DD for directory structure
            date_str = target_date.strftime('%Y-%m-%d')
            alerts_base_dir = os.path.join('historical_data', date_str, 'superduper_alerts_sent')
            
            if not os.path.exists(alerts_base_dir):
                print(f"No superduper alerts directory found for {target_date}")
                return alerts
            
            # Check both bullish and bearish alert directories
            for alert_type in ['bullish', 'bearish']:
                alert_type_dir = os.path.join(alerts_base_dir, alert_type)
                
                if not os.path.exists(alert_type_dir):
                    continue
                
                # Check both yellow and green alert levels
                for alert_level in ['yellow', 'green']:
                    alert_level_dir = os.path.join(alert_type_dir, alert_level)
                    
                    if not os.path.exists(alert_level_dir):
                        continue
                    
                    # Look for superduper alert files matching the symbol
                    alert_pattern = f"superduper_alert_{symbol}_*.json"
                    alert_files = glob.glob(os.path.join(alert_level_dir, alert_pattern))
                    
                    for alert_file in alert_files:
                        try:
                            with open(alert_file, 'r') as f:
                                alert_data = json.load(f)
                            
                            # Add alert type and level
                            alert_data['alert_type'] = alert_type
                            alert_data['alert_level'] = alert_level
                            
                            # Parse timestamp to datetime object
                            if 'timestamp' in alert_data:
                                timestamp_str = alert_data['timestamp']
                                try:
                                    # Handle timezone format: convert -0400 to -04:00 for Python compatibility
                                    if timestamp_str.endswith(('-0400', '-0500')):
                                        timestamp_str = timestamp_str[:-2] + ':' + timestamp_str[-2:]
                                    
                                    # Parse the timestamp - super alerts are in ET timezone with offset
                                    alert_dt = datetime.fromisoformat(timestamp_str)
                                    alert_data['timestamp_dt'] = alert_dt
                                    
                                except ValueError:
                                    # Fallback: try parsing without timezone, then localize to ET
                                    try:
                                        alert_dt = datetime.fromisoformat(timestamp_str.split('+')[0].split('-0400')[0].split('-0500')[0])
                                        # If timezone-naive, assume it's in ET timezone
                                        if alert_dt.tzinfo is None:
                                            et_tz = pytz.timezone('America/New_York')
                                            alert_dt = et_tz.localize(alert_dt)
                                        alert_data['timestamp_dt'] = alert_dt
                                    except ValueError:
                                        print(f"Warning: Could not parse timestamp in {alert_file}")
                                        continue
                            
                            alerts.append(alert_data)
                            
                        except Exception as e:
                            print(f"Warning: Error loading superduper alert file {alert_file}: {e}")
                            continue
            
            # Sort alerts by timestamp
            alerts.sort(key=lambda x: x.get('timestamp_dt', datetime.min.replace(tzinfo=pytz.UTC)))
            
            print(f"Loaded {len(alerts)} superduper alerts for {symbol} on {target_date}")
            
        except Exception as e:
            print(f"Error loading superduper alerts for {symbol} on {target_date}: {e}")
        
        return alerts

    def _generate_plot(self, symbol: str, plot_date: Optional[str] = None) -> bool:
        """
        Generate candlestick chart with MACD for a symbol.

        Args:
            symbol: Stock symbol to plot
            plot_date: Date in YYYY-MM-DD format (default: today)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Set up date - use provided date or default to today
            if plot_date:
                target_date = datetime.strptime(plot_date, '%Y-%m-%d').date()
            else:
                target_date = date.today()

            print(f"Generating chart for {symbol} on {target_date}")

            # Set up time range for market data (Eastern Time)
            et_tz = pytz.timezone('America/New_York')
            start_time = datetime.combine(target_date, dt_time(4, 0), tzinfo=et_tz)
            end_time = datetime.combine(target_date, dt_time(20, 0), tzinfo=et_tz)

            print(f"Fetching market data from {start_time} to {end_time}")

            # Fetch market data using Alpaca API
            try:
                # Format as RFC3339 with proper timezone format
                start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S') + start_time.strftime('%z')[:3] + ':' + start_time.strftime('%z')[3:]
                end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S') + end_time.strftime('%z')[:3] + ':' + end_time.strftime('%z')[3:]
                
                bars = self.api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start_str,
                    end=end_str,
                    limit=10000,
                    feed='iex'  # Use IEX feed for reliability
                )
            except Exception as e:
                print(f"✗ Error fetching market data: {e}")
                return False

            if not bars:
                print(f"✗ No market data available for {symbol} on {target_date}")
                return False

            # Convert bars to DataFrame
            market_data = []
            for bar in bars:
                bar_data = {
                    'timestamp': bar.t.isoformat(),
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v),
                    'symbol': symbol
                }
                market_data.append(bar_data)

            # Create DataFrame
            df = pd.DataFrame(market_data)
            print(f"Retrieved {len(df)} minutes of data")

            if len(df) < 26:  # Need at least 26 periods for MACD
                print(f"✗ Insufficient data for MACD calculation: {len(df)} < 26 periods")
                return False

            # Set up output directory
            date_str = target_date.strftime('%Y%m%d')

            # Load superduper alerts for this symbol and date
            alerts = self._load_superduper_alerts_for_symbol(symbol, target_date)
            
            # Score alerts using MACD analysis
            if alerts:
                print(f"Scoring {len(alerts)} alerts using MACD analysis...")
                alerts = score_alerts_with_macd(df, alerts)

            # Generate chart using the chart generation atom
            success = generate_chart_from_dataframe(
                df=df,
                symbol=symbol,
                output_dir='plots',  # Base directory - function will create date subdirectory
                alerts=alerts,  # Include loaded alerts
                verbose=True
            )

            if success:
                chart_path = f'plots/{date_str}/{symbol}_chart.png'
                print(f"✓ Chart generated successfully: {chart_path}")
                print(f"  Chart includes:")
                print(f"    • Price candlesticks with ORB levels, EMA(9), EMA(20), VWAP")
                print(f"    • MACD indicators (12,26,9) with MACD line, Signal line, Histogram")
                print(f"    • Volume data")
                if alerts:
                    # Count alerts by MACD score color
                    colors = [alert.get('macd_score', {}).get('color', 'unknown') for alert in alerts]
                    green_count = colors.count('green')
                    yellow_count = colors.count('yellow')
                    red_count = colors.count('red')
                    
                    print(f"    • {len(alerts)} superduper alert overlays with MACD scoring:")
                    if green_count:
                        print(f"      🟢 {green_count} GREEN (excellent MACD conditions)")
                    if yellow_count:
                        print(f"      🟡 {yellow_count} YELLOW (moderate MACD conditions)")
                    if red_count:
                        print(f"      🔴 {red_count} RED (poor MACD conditions)")
                return True
            else:
                print("✗ Chart generation failed")
                return False

        except Exception as e:
            print(f"✗ Error generating plot: {e}")
            return False

    def _isFirstTradeOfDay(self) -> bool:
        """
        Check if position monitoring has already been started today.
        
        Returns:
            True if this is the first trade and monitoring hasn't started today
        """
        current_date = datetime.now(pytz.timezone('America/New_York')).date()
        
        # Reset tracking if it's a new day
        if current_date != self._today_date:
            self._today_date = current_date
            self._first_trade_completed = False
            self._monitoring_started_today = False
            
        return not self._monitoring_started_today

    def _startPositionMonitoring(self) -> bool:
        """
        Start position monitoring as a separate subprocess.
        
        Returns:
            True if monitoring was started successfully, False otherwise
        """
        try:
            # Check if monitoring is already running
            if self._monitoring_process and self._monitoring_process.poll() is None:
                print("✓ Position monitoring already running")
                return True
                
            print("🚀 Starting automatic position monitoring...")
            
            # Build command to start position monitoring
            python_path = sys.executable
            script_path = os.path.abspath(__file__)
            
            monitor_cmd = [
                python_path,
                script_path,
                "--monitor-positions",
                "--account-name", self.account_name,
                "--account", self.account
            ]
            
            print(f"Command: {' '.join(monitor_cmd)}")
            
            # Start monitoring as subprocess
            self._monitoring_process = subprocess.Popen(
                monitor_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent process
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if process started successfully
            if self._monitoring_process.poll() is None:
                self._monitoring_started_today = True
                print(f"✅ Position monitoring started successfully (PID: {self._monitoring_process.pid})")
                print("📊 Monitoring will liquidate positions when MACD score is RED")
                print("📱 Telegram notifications will be sent to Bruce for liquidations")
                return True
            else:
                # Process failed to start or exited immediately
                stdout, stderr = self._monitoring_process.communicate()
                print(f"✗ Position monitoring failed to start")
                print(f"STDOUT: {stdout.decode()[:200]}...")
                print(f"STDERR: {stderr.decode()[:200]}...")
                self._monitoring_process = None
                return False
                
        except Exception as e:
            print(f"✗ Error starting position monitoring: {str(e)}")
            self._monitoring_process = None
            return False

    def _onTradeExecuted(self, trade_result: Any, operation: str) -> None:
        """
        Called after a successful trade execution to trigger monitoring if needed.
        
        Args:
            trade_result: Result from trade execution
            operation: Description of the operation performed
        """
        try:
            if trade_result and self._isFirstTradeOfDay():
                print(f"🎯 First trade of the day executed: {operation}")
                print("🔄 Automatically starting position monitoring...")
                
                if self._startPositionMonitoring():
                    print("✅ Position monitoring is now active")
                else:
                    print("⚠️ Failed to start position monitoring - please start manually")
                    
        except Exception as e:
            print(f"⚠️ Error in post-trade monitoring setup: {str(e)}")

    def _cleanup_monitoring_process(self) -> None:
        """
        Clean up the monitoring subprocess if it's running.
        Should be called during graceful shutdown.
        """
        try:
            if self._monitoring_process and self._monitoring_process.poll() is None:
                print("🛑 Cleaning up position monitoring process...")
                self._monitoring_process.terminate()
                
                # Give it a few seconds to terminate gracefully
                try:
                    self._monitoring_process.wait(timeout=5)
                    print("✅ Position monitoring process terminated gracefully")
                except subprocess.TimeoutExpired:
                    print("⚠️ Monitoring process didn't terminate gracefully, killing...")
                    self._monitoring_process.kill()
                    self._monitoring_process.wait()
                    print("✅ Position monitoring process killed")
                    
                self._monitoring_process = None
        except Exception as e:
            print(f"⚠️ Error cleaning up monitoring process: {str(e)}")

    def _sendTelegramLiquidationNotification(self, symbol: str, reason: str, macd_details: Dict[str, Any]) -> None:
        """
        Send Telegram notification when a position is liquidated.
        
        Args:
            symbol: Stock symbol that was liquidated
            reason: Reason for liquidation
            macd_details: MACD analysis details for context
        """
        try:
            # Use account name directly for Telegram username
            account_holder = self.account_name
            
            # Format the notification message
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            message = f"""🚨 POSITION LIQUIDATED 🚨

Account: {self.account_name} ({account_holder})
Symbol: {symbol}
Time: {timestamp}
Environment: {self.account}

Reason: {reason}

MACD Details:
• MACD Line: {macd_details.get('macd_line', 'N/A'):.3f}
• Signal Line: {macd_details.get('signal_line', 'N/A'):.3f}
• Histogram: {macd_details.get('histogram', 'N/A'):.3f}
• Score: {macd_details.get('score', 'N/A')}/4 ({macd_details.get('color', 'N/A').upper()})

This position was automatically liquidated due to poor MACD conditions."""
            
            # Send notification to the specific account holder
            telegram_poster = TelegramPoster()
            result = telegram_poster.send_message_to_user(
                message=message,
                username=account_holder,
                urgent=True  # Mark as urgent for liquidation alerts
            )
            
            if result['success']:
                print(f"    ✓ Telegram notification sent to {account_holder}")
            else:
                print(f"    ⚠️  Failed to send Telegram notification to {account_holder}: {result['errors']}")
                
        except Exception as e:
            print(f"    ⚠️  Error sending Telegram notification: {str(e)}")

    def _sendTelegramLiquidationFailureNotification(self, symbol: str, reason: str, macd_details: Dict[str, Any]) -> None:
        """
        Send Telegram notification when a position liquidation fails.
        
        Args:
            symbol: Stock symbol that failed to liquidate
            reason: Reason for attempted liquidation
            macd_details: MACD analysis details for context
        """
        try:
            # Use account name directly for Telegram username
            account_holder = self.account_name
            
            # Format the failure notification message
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            message = f"""⚠️ LIQUIDATION FAILED ⚠️

Account: {self.account_name} ({account_holder})
Symbol: {symbol}
Time: {timestamp}
Environment: {self.account}

ATTEMPTED LIQUIDATION - FAILED TO EXECUTE

Reason for liquidation attempt: {reason}

MACD Details (indicating poor conditions):
• MACD Line: {macd_details.get('macd_line', 'N/A'):.3f}
• Signal Line: {macd_details.get('signal_line', 'N/A'):.3f}
• Histogram: {macd_details.get('histogram', 'N/A'):.3f}
• Score: {macd_details.get('score', 'N/A')}/4 ({macd_details.get('color', 'N/A').upper()})

⚠️ MANUAL INTERVENTION REQUIRED ⚠️
The system attempted to liquidate this position due to poor MACD conditions but the liquidation failed. Please review the position manually and take appropriate action."""
            
            # Send notification to the specific account holder
            telegram_poster = TelegramPoster()
            result = telegram_poster.send_message_to_user(
                message=message,
                username=account_holder,
                urgent=True  # Mark as urgent for failed liquidation alerts
            )
            
            if result['success']:
                print(f"    ✓ Telegram failure notification sent to {account_holder}")
            else:
                print(f"    ⚠️  Failed to send Telegram failure notification to {account_holder}: {result['errors']}")
                
        except Exception as e:
            print(f"    ⚠️  Error sending Telegram failure notification: {str(e)}")

    def _sendHourlyPositionReport(self, positions, current_time) -> None:
        """
        Send hourly position report via Telegram.
        
        Args:
            positions: List of current positions
            current_time: Current datetime with ET timezone
        """
        try:
            account_holder = self.account_name
            
            # Create position summary
            position_summary = []
            total_market_value = 0
            
            for pos in positions:
                shares = int(pos.qty)
                market_value = float(pos.market_value)
                current_price = float(pos.current_price)
                unrealized_pnl = float(pos.unrealized_pnl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100
                
                total_market_value += market_value
                
                pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
                
                position_summary.append(
                    f"• {pos.symbol}: {shares:,} shares @ ${current_price:.2f}\n"
                    f"  Value: ${market_value:,.2f} {pnl_emoji} ${unrealized_pnl:+.2f} ({unrealized_pnl_pct:+.1f}%)"
                )
            
            # Format the report message
            time_str = current_time.strftime('%Y-%m-%d %H:%M:%S ET')
            current_minute = current_time.minute
            
            # Determine if this is top of hour or bottom of hour report
            if current_minute <= 2:
                report_type = "Top of Hour"
            elif 28 <= current_minute <= 32:
                report_type = "Bottom of Hour"
            elif current_minute >= 58:
                report_type = "Top of Hour (Next)"
            else:
                report_type = "Position"  # fallback
            
            message = f"""📊 {report_type} Position Report 📊

Account: {self.account_name}
Time: {time_str}
Environment: {self.account}

Current Holdings ({len(positions)} positions):

{chr(10).join(position_summary)}

Total Portfolio Value: ${total_market_value:,.2f}

Monitor Status: ✅ Active
Next Report: {"Top of hour" if current_minute >= 28 else "Bottom of hour"} ({("00" if current_minute >= 28 else "30")})"""
            
            # Send notification
            telegram_poster = TelegramPoster()
            result = telegram_poster.send_message_to_user(
                message=message,
                username=account_holder,
                urgent=False
            )
            
            if result['success']:
                print(f"    ✓ {report_type} position report sent to {account_holder}")
            else:
                print(f"    ⚠️  Failed to send position report to {account_holder}: {result['errors']}")
                
        except Exception as e:
            print(f"    ⚠️  Error sending hourly position report: {str(e)}")

    def _closeAllPositions(self) -> None:
        """
        Close all open positions and send Telegram notification.
        """
        try:
            positions = get_positions(self.api, self.account_name, self.account)
            if not positions:
                return
            
            account_holder = self.account_name
            closed_positions = []
            failed_positions = []
            
            print(f"Attempting to close {len(positions)} positions...")
            
            for pos in positions:
                symbol = pos.symbol
                shares = int(pos.qty)
                side = "sell" if shares > 0 else "buy"
                quantity = abs(shares)
                
                try:
                    # Close the position
                    result = self._liquidate_position(symbol=symbol, submit_order=True)
                    if result:
                        closed_positions.append(f"{symbol} ({shares:,} shares)")
                        print(f"    ✓ Closed position in {symbol}")
                    else:
                        failed_positions.append(f"{symbol} ({shares:,} shares)")
                        print(f"    ✗ Failed to close position in {symbol}")
                        
                except Exception as e:
                    failed_positions.append(f"{symbol} ({shares:,} shares) - Error: {str(e)}")
                    print(f"    ✗ Error closing {symbol}: {str(e)}")
            
            # Send Telegram notification about position closures
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            success_section = ""
            if closed_positions:
                success_section = f"""
✅ Successfully Closed ({len(closed_positions)}):
{chr(10).join(f'• {pos}' for pos in closed_positions)}"""
            
            failure_section = ""
            if failed_positions:
                failure_section = f"""
❌ Failed to Close ({len(failed_positions)}):
{chr(10).join(f'• {pos}' for pos in failed_positions)}"""
            
            message = f"""🚨 ALL POSITIONS CLOSED - Market Close Time 🚨

Account: {self.account_name}
Time: {timestamp}
Environment: {self.account}

Reason: Automatic closure at 15:40 ET{success_section}{failure_section}

Monitor Status: 🔄 Shutting down after position closure"""
            
            telegram_poster = TelegramPoster()
            result = telegram_poster.send_message_to_user(
                message=message,
                username=account_holder,
                urgent=True
            )
            
            if result['success']:
                print(f"    ✓ Position closure notification sent to {account_holder}")
            else:
                print(f"    ⚠️  Failed to send closure notification to {account_holder}: {result['errors']}")
                
        except Exception as e:
            print(f"    ⚠️  Error in _closeAllPositions: {str(e)}")

    def _sendTelegramShutdownNotification(self) -> None:
        """
        Send Telegram notification when monitor shuts down due to no positions.
        """
        try:
            account_holder = self.account_name
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            message = f"""🏁 POSITION MONITOR SHUTDOWN 🏁

Account: {self.account_name}
Time: {timestamp}
Environment: {self.account}

Reason: All positions closed
Status: ✅ Monitor stopped automatically

No further position monitoring will occur until restarted."""
            
            telegram_poster = TelegramPoster()
            result = telegram_poster.send_message_to_user(
                message=message,
                username=account_holder,
                urgent=False
            )
            
            if result['success']:
                print(f"    ✓ Monitor shutdown notification sent to {account_holder}")
            else:
                print(f"    ⚠️  Failed to send shutdown notification to {account_holder}: {result['errors']}")
                
        except Exception as e:
            print(f"    ⚠️  Error sending shutdown notification: {str(e)}")

    def _monitorPositions(self) -> None:
        """
        Monitor positions continuously with enhanced features:
        
        1. Polls positions every minute
        2. Sends hourly Telegram position reports (top and bottom of hour)
        3. Calculates MACD and scores it for each position
        4. Liquidates position if MACD score is red
        5. Closes all positions at 15:40 ET and sends notification
        6. Shuts down monitor when all positions are closed
        
        The polling continues until all positions are closed or stopped manually.
        """
        print("Starting enhanced position monitoring...")
        print("Features:")
        print("- Polling every 60 seconds")
        print("- Hourly Telegram position reports")
        print("- MACD-based position liquidation")
        print("- Auto-close all positions at 15:40 ET")
        print("- Auto-shutdown when all positions closed")
        print("=" * 60)
        
        macd_scorer = MACDAlertScorer()
        et_tz = pytz.timezone('America/New_York')
        last_hour_report = None
        
        try:
            while True:
                try:
                    current_et = datetime.now(et_tz)
                    
                    # Check if it's time to close all positions (15:40 ET)
                    if current_et.hour == 15 and current_et.minute >= 40:
                        positions = get_positions(self.api, self.account_name, self.account)
                        if positions:
                            print(f"[{current_et.strftime('%Y-%m-%d %H:%M:%S ET')}] 🚨 Closing all positions - Market close time (15:40 ET)")
                            self._closeAllPositions()
                            print("Monitor shutting down - All positions closed at market close time")
                            break
                    
                    # Get current positions
                    positions = get_positions(self.api, self.account_name, self.account)
                    
                    # Check if all positions are closed - shutdown monitor
                    if not positions:
                        print(f"[{current_et.strftime('%Y-%m-%d %H:%M:%S ET')}] No positions found")
                        
                        # Send shutdown notification if we had positions before
                        if hasattr(self, '_had_positions'):
                            self._sendTelegramShutdownNotification()
                            print("Monitor shutting down - All positions closed")
                            break
                    else:
                        # Mark that we have positions
                        self._had_positions = True
                        
                        # Create a list of unique stock symbols
                        unique_symbols = list(set(pos.symbol for pos in positions))
                        
                        print(f"[{current_et.strftime('%Y-%m-%d %H:%M:%S ET')}] Monitoring {len(unique_symbols)} symbols: {', '.join(unique_symbols)}")
                        
                        # Send position reports twice per hour (top and bottom of hour)
                        current_hour = current_et.hour
                        current_minute = current_et.minute
                        
                        # Check for reports at top of hour (minute 0 ±2) and bottom of hour (minute 30 ±2)
                        is_top_of_hour = current_minute <= 2
                        is_bottom_of_hour = 28 <= current_minute <= 32
                        is_next_hour_prep = current_minute >= 58
                        
                        if is_top_of_hour or is_bottom_of_hour or is_next_hour_prep:
                            # Create unique identifier for each report time
                            if is_top_of_hour:
                                # Top of hour: use hour * 100 (e.g., 14:00 = 1400)
                                report_id = current_hour * 100
                            elif is_bottom_of_hour:
                                # Bottom of hour: use hour * 100 + 30 (e.g., 14:30 = 1430)
                                report_id = current_hour * 100 + 30
                            else:  # is_next_hour_prep (minutes 58-59)
                                # Next hour prep: use (next_hour) * 100 (e.g., 14:58 -> 1500 for 15:00)
                                next_hour = (current_hour + 1) % 24
                                report_id = next_hour * 100
                            
                            # Only send if we haven't sent this specific report already
                            if last_hour_report != report_id:
                                self._sendHourlyPositionReport(positions, current_et)
                                last_hour_report = report_id
                        
                        # Process each unique symbol for MACD analysis
                        for symbol in unique_symbols:
                            try:
                                self._processSymbolForMonitoring(symbol, macd_scorer)
                            except Exception as e:
                                print(f"  ✗ Error processing {symbol}: {str(e)}")
                                continue
                    
                    print(f"  Next check in 60 seconds...")
                    print("-" * 40)
                    
                    # Wait for 60 seconds (1 minute polling period)
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    print("\nReceived Ctrl+C. Stopping position monitoring...")
                    break
                except Exception as e:
                    print(f"  ✗ Error during monitoring cycle: {str(e)}")
                    print("  Waiting 60 seconds before retry...")
                    time.sleep(60)
                    continue
                    
        except KeyboardInterrupt:
            print("\nPosition monitoring stopped by user.")
        except Exception as e:
            print(f"✗ Critical error in position monitoring: {str(e)}")
            
    def _processSymbolForMonitoring(self, symbol: str, macd_scorer: MACDAlertScorer) -> None:
        """
        Process a single symbol for MACD analysis and potential liquidation.
        
        Args:
            symbol: Stock symbol to analyze
            macd_scorer: MACD scorer instance
        """
        print(f"  Processing {symbol}...")
        
        # Collect sufficient data for MACD calculation (need at least 26 periods for default MACD)
        try:
            # Get market data for the last 50 minutes to ensure we have enough data
            et_tz = pytz.timezone('America/New_York')
            end_time = datetime.now(et_tz)
            start_time = end_time - pd.Timedelta(minutes=50)
            
            # Fetch market data using Alpaca API
            # Format as RFC3339 with proper timezone format
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S') + start_time.strftime('%z')[:3] + ':' + start_time.strftime('%z')[3:]
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S') + end_time.strftime('%z')[:3] + ':' + end_time.strftime('%z')[3:]
            
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Minute,
                start=start_str,
                end=end_str,
                limit=1000,
                feed='iex'
            )
            
            if not bars or len(bars) < 26:
                print(f"    ⚠️  Insufficient market data for {symbol} ({len(bars) if bars else 0} bars, need ≥26)")
                return
            
            # Convert bars to DataFrame
            market_data = []
            for bar in bars:
                bar_data = {
                    'timestamp': bar.t.isoformat(),
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v),
                    'symbol': symbol
                }
                market_data.append(bar_data)
            
            df = pd.DataFrame(market_data)
            
            # Calculate MACD
            macd_success, macd_values = calculate_macd(
                df,
                fast_length=12,
                slow_length=26,
                signal_length=9,
                source='close'
            )
            
            if not macd_success:
                print(f"    ✗ MACD calculation failed for {symbol}")
                return
            
            # Use the most recent timestamp for scoring
            current_time = datetime.now(et_tz)
            
            # Analyze MACD conditions
            macd_analysis = macd_scorer.calculate_macd_conditions(df, current_time)
            
            if not macd_analysis.get('is_valid', False):
                print(f"    ✗ MACD analysis failed for {symbol}: {macd_analysis.get('error', 'Unknown error')}")
                return
            
            # Score the MACD conditions
            score_result = macd_scorer.score_alert(macd_analysis)
            
            # Get current price and MACD values for display
            current_price = float(df.iloc[-1]['close'])
            macd_line = macd_analysis['macd_line']
            signal_line = macd_analysis['signal_line']
            histogram = macd_analysis['histogram']
            
            # Display analysis results
            color_emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
            emoji = color_emoji.get(score_result['color'], '❓')
            
            print(f"    {emoji} {symbol}: Price=${current_price:.2f}, MACD={macd_line:.3f}, "
                  f"Signal={signal_line:.3f}, Histogram={histogram:.3f}")
            print(f"    MACD Score: {score_result['color'].upper()} ({score_result['score']}/4) - "
                  f"{score_result['confidence']} confidence")
            
            # If MACD score is red, liquidate the position
            if score_result['color'] == 'red':
                print(f"    🚨 LIQUIDATING {symbol} - MACD score is RED")
                print(f"    Reason: {score_result['reasoning']}")
                
                liquidation_result = self._liquidate_position(
                    symbol=symbol,
                    submit_order=True  # Actually execute the liquidation
                )
                
                # Prepare MACD details for notification
                macd_details = {
                    'macd_line': macd_line,
                    'signal_line': signal_line,
                    'histogram': histogram,
                    'score': score_result['score'],
                    'color': score_result['color']
                }
                
                if liquidation_result:
                    print(f"    ✓ Successfully liquidated position in {symbol}")
                    
                    # Send Telegram notification for successful liquidation
                    self._sendTelegramLiquidationNotification(
                        symbol=symbol,
                        reason=score_result['reasoning'],
                        macd_details=macd_details
                    )
                else:
                    print(f"    ✗ Failed to liquidate position in {symbol}")
                    
                    # Send Telegram notification for failed liquidation
                    self._sendTelegramLiquidationFailureNotification(
                        symbol=symbol,
                        reason=score_result['reasoning'],
                        macd_details=macd_details
                    )
            else:
                print(f"    ✓ Keeping position in {symbol} - "
                      f"MACD score is {score_result['color'].upper()}")

        except Exception as e:
            print(f"    ✗ Error analyzing {symbol}: {str(e)}")

    def Exec(self) -> int:
        """
        Execute the main trading logic.

        Prints current positions, cash balance, and active orders based on arguments.
        If bracket order arguments are provided, executes a bracket order.

        Returns:
            Exit code (0 for success)
        """

        # Handle PNL report (standalone operation)
        if self.args.PNL:
            try:
                # Get account credentials using class variables
                if CONFIG_AVAILABLE:
                    api_key, secret_key, base_url = get_api_credentials("alpaca", self.account_name, self.account)
                    pnl_client = AlpacaDailyPnL(api_key, secret_key, base_url, self.account_name, self.account)
                    pnl_client.create_pnl()
                else:
                    print(f"Error: Configuration not available. "
                          f"Unable to access {self.account_name}:{self.account} account credentials.")
                    return 1
            except Exception as e:
                print(f"Error generating PNL report: {str(e)}")
                return 1
            return 0  # Exit early for PNL operation

        # Handle position monitoring (standalone operation)
        if self.args.monitor_positions:
            try:
                self._monitorPositions()
                return 0
            except Exception as e:
                print(f"Error during position monitoring: {str(e)}")
                return 1

        # Handle plot generation (standalone operation)
        if self.args.plot:
            try:
                success = self._generate_plot(self.args.symbol, self.args.date)
                if success:
                    print("✓ Chart generation completed successfully")
                    return 0
                else:
                    print("✗ Chart generation failed")
                    return 1
            except Exception as e:
                print(f"✗ Error during chart generation: {e}")
                return 1

        # Handle display-only arguments
        display_args = [self.args.positions, self.args.cash, self.args.active_order]
        if any(display_args):
            # Only show requested information
            if self.args.positions:
                print_positions(self.api, self.account_name, self.account)
            if self.args.cash:
                print_cash(self.api, self.account_name, self.account)
            if self.args.active_order:
                print_active_orders(self.api)
            return 0  # Exit early for display-only operations

        # Default behavior: show all information for other operations
        print_positions(self.api, self.account_name, self.account)
        print_cash(self.api, self.account_name, self.account)
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
            print_quote(self.api, self.args.symbol, self.account_name, self.account)

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

        # Handle buy-market order if requested
        if self.args.buy_market:
            order_result = self._buy_market(
                symbol=self.args.symbol,
                amount=self.args.amount,
                submit_order=self.args.submit
            )
            if order_result is None and self.args.submit:
                print("Failed to submit market buy order")
                return 1

        # Handle buy-market-trailing-sell order if requested
        if self.args.buy_market_trailing_sell:
            order_result = self._buy_market_trailing_sell(
                symbol=self.args.symbol,
                amount=self.args.amount,
                trailing_percent=self.args.trailing_percent,
                submit_order=self.args.submit
            )
            if order_result is None and self.args.submit:
                print("Failed to submit market buy with trailing sell")
                return 1

        # Handle buy-market-trailing-sell-take-profit-percent order if requested
        if self.args.buy_market_trailing_sell_take_profit_percent:
            order_result = self._buy_market_trailing_sell_take_profit_percent(
                symbol=self.args.symbol,
                take_profit_percent=self.args.take_profit_percent,
                amount=self.args.amount,
                trailing_percent=self.args.trailing_percent,
                submit_order=self.args.submit
            )
            if order_result is None and self.args.submit:
                print("Failed to submit market buy with trailing sell and take profit percent")
                return 1

        # Handle trailing sell order if requested
        if self.args.sell_trailing:
            order_result = self._sell_trailing(
                symbol=self.args.symbol,
                quantity=self.args.quantity,
                trailing_percent=self.args.trailing_percent,
                submit_order=self.args.submit
            )
            if order_result is None and self.args.submit:
                print("Failed to submit trailing sell order")
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

        # Handle liquidation operations
        if self.args.liquidate:
            liquidation_result = self._liquidate_position(
                symbol=self.args.symbol,
                submit_order=self.args.submit
            )
            if liquidation_result is None and self.args.submit:
                print("Failed to liquidate position")
                return 1

        if self.args.liquidate_all:
            liquidation_result = self._liquidate_all(
                cancel_orders=self.args.cancel_orders,
                submit_order=self.args.submit
            )
            if liquidation_result is None and self.args.submit:
                print("Failed to liquidate all positions")
                return 1

        # Handle cancel all orders operation
        if self.args.cancel_all_orders:
            cancel_result = self._cancel_all_orders(
                submit_order=True  # Always execute cancellation for --cancel-all-orders
            )
            if cancel_result is None:
                print("Failed to cancel all orders")
                return 1

        # Handle take-profit-percent order if requested (standalone only, not combined operations)
        if self.args.take_profit_percent and not self.args.buy_market_trailing_sell_take_profit_percent:
            order_result = self._take_profit_percent(
                symbol=self.args.symbol,
                quantity=self.args.quantity,
                take_profit_percent=self.args.take_profit_percent,
                submit_order=self.args.submit
            )
            if order_result is None and self.args.submit:
                print("Failed to submit take profit percent order")
                return 1

        # Cleanup before exit
        self._cleanup_monitoring_process()
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

    alpacaObj = AlpacaPrivate(userArgs)

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
