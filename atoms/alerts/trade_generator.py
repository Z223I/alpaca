"""
Trade Generator - Monitors superduper alerts and executes automated trades

This atom handles the monitoring of superduper alert files and executes automated
trades based on configuration settings. It mirrors the structure of superduper_alert_generator.py
but focuses on trade execution rather than alert generation.
"""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pytz


class TradeGenerator:
    """Monitors superduper alerts and executes automated trades."""

    def __init__(self, historical_data_dir: Path, test_mode: bool = False):
        """
        Initialize the trade generator.

        Args:
            historical_data_dir: Base directory for historical data
            test_mode: Whether running in test mode
        """
        self.historical_data_dir = historical_data_dir
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self._load_config()

        # Track trades executed (limit to 1 for initial testing)
        self.max_trades = 1
        self.trades_executed = 0

    def _load_config(self):
        """Load configuration from alpaca_config.py."""
        try:
            # Add code directory to path to import config
            code_dir = Path(__file__).parent.parent.parent / "code"
            sys.path.insert(0, str(code_dir))

            from alpaca_config import get_current_config
            self.config = get_current_config()
            self.logger.debug("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract stock symbol from superduper alert filename.

        Args:
            filename: Filename like 'superduper_alert_LOBO_20250802_135836.json'

        Returns:
            Stock symbol (e.g., 'LOBO') or None if extraction fails
        """
        try:
            # Pattern: superduper_alert_SYMBOL_YYYYMMDD_HHMMSS
            pattern = r'superduper_alert_([A-Z]+)_\d{8}_\d{6}'
            match = re.match(pattern, filename)

            if match:
                symbol = match.group(1)
                self.logger.debug(f"Extracted symbol '{symbol}' from filename '{filename}'")
                return symbol
            else:
                self.logger.warning(f"Could not extract symbol from filename: {filename}")
                return None

        except Exception as e:
            self.logger.error(f"Error extracting symbol from filename '{filename}': {e}")
            return None

    def monitor_superduper_alerts(self, date_str: str) -> List[str]:
        """
        Monitor the superduper alerts directory for new files.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            List of symbols extracted from new alert files
        """
        try:
            alerts_dir = (self.historical_data_dir / date_str /
                          "superduper_alerts_sent" / "bullish" / "green")

            if not alerts_dir.exists():
                self.logger.debug(f"Alerts directory does not exist: {alerts_dir}")
                return []

            symbols = []
            for file_path in alerts_dir.glob("superduper_alert_*.json"):
                if file_path.is_file():
                    symbol = self.extract_symbol_from_filename(file_path.name)
                    if symbol:
                        symbols.append(symbol)
                        self.logger.info(f"Found superduper alert for symbol: {symbol}")

            return symbols

        except Exception as e:
            self.logger.error(f"Error monitoring superduper alerts: {e}")
            return []

    def execute_trade(self, symbol: str, account_name: str, account_type: str,
                      auto_amount: int, trailing_percent: float) -> Dict[str, Any]:
        """
        Execute a trade using the alpaca.py script.

        Args:
            symbol: Stock symbol to trade
            account_name: Account name (Primary, Bruce, Dale)
            account_type: Account type (paper, live, cash)
            auto_amount: Dollar amount to trade
            trailing_percent: Trailing stop percentage

        Returns:
            Dictionary containing trade results
        """
        try:
            # Use direct Python path as specified in instructions
            python_path = "/home/wilsonb/dl/github.com/z223i/alpaca/code/alpaca.py"

            cmd = [
                "~/miniconda3/envs/alpaca/bin/python",
                python_path,
                "--sell-trailing",
                "--symbol", symbol,
                "--amount", str(auto_amount),
                "--trailing-percent", str(trailing_percent)
                # Note: Intentionally NOT adding --submit for initial testing (dry run)
            ]

            self.logger.info(f"Executing trade command for {symbol}: {' '.join(cmd)}")

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True,  # Use shell to handle tilde expansion
                timeout=60
            )

            # Parse the result
            trade_result = {
                "symbol": symbol,
                "account_name": account_name,
                "account_type": account_type,
                "auto_amount": auto_amount,
                "trailing_percent": trailing_percent,
                "command": " ".join(cmd),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": "no",  # Default to no since we're in dry run mode
                "timestamp": datetime.now(pytz.timezone('US/Eastern')).isoformat(),
                "test_mode": self.test_mode
            }

            # Determine success based on return code and output
            if result.returncode == 0:
                # Check if this was a dry run (no --submit flag)
                if "--submit" not in cmd:
                    trade_result["success"] = "no"  # Dry run
                    trade_result["message"] = "Dry run executed successfully"
                else:
                    trade_result["success"] = "yes"  # Actual trade
                    trade_result["message"] = "Trade executed successfully"
            else:
                trade_result["success"] = "no"
                trade_result["message"] = f"Trade failed with return code {result.returncode}"

            self.logger.info(f"Trade execution completed for {symbol}: success={trade_result['success']}")
            return trade_result

        except subprocess.TimeoutExpired:
            error_result = {
                "symbol": symbol,
                "account_name": account_name,
                "account_type": account_type,
                "success": "no",
                "message": "Trade command timed out",
                "timestamp": datetime.now(pytz.timezone('US/Eastern')).isoformat(),
                "test_mode": self.test_mode
            }
            self.logger.error(f"Trade command timed out for {symbol}")
            return error_result

        except Exception as e:
            error_result = {
                "symbol": symbol,
                "account_name": account_name,
                "account_type": account_type,
                "success": "no",
                "message": f"Trade execution error: {str(e)}",
                "timestamp": datetime.now(pytz.timezone('US/Eastern')).isoformat(),
                "test_mode": self.test_mode
            }
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return error_result

    def save_trade_result(self, trade_result: Dict[str, Any], date_str: str) -> Optional[str]:
        """
        Save trade result to JSON file.

        Args:
            trade_result: Trade result data
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            symbol = trade_result['symbol']
            account_name = trade_result['account_name']
            account_type = trade_result['account_type']

            # Create directory structure: historical_data/YYYY-MM-DD/<account_name>/<account_type>/
            trades_dir = (self.historical_data_dir / date_str / account_name / account_type)
            trades_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for filename
            et_tz = pytz.timezone('US/Eastern')
            timestamp = datetime.now(et_tz)

            # Generate filename
            filename = f"trade_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = trades_dir / filename

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(trade_result, f, indent=2)

            self.logger.debug(f"Trade result saved: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error saving trade result: {e}")
            return None

    def process_trades_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Process trades for a specific date.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            List of trade results
        """
        try:
            # Monitor for new superduper alerts
            symbols = self.monitor_superduper_alerts(date_str)

            if not symbols:
                self.logger.info(f"No superduper alerts found for {date_str}")
                return []

            trade_results = []
            trade_count = 0

            # Process each account configuration
            for account_name, account_config in self.config.providers["alpaca"].accounts.items():
                for account_type in ["paper", "live", "cash"]:
                    env_config = getattr(account_config, account_type)

                    # Check if auto_trade is enabled
                    if env_config.auto_trade == "yes":
                        trade_count += 1

                        # Limit to maximum trades for initial testing
                        if trade_count <= self.max_trades:
                            # Process each symbol
                            for symbol in symbols:
                                trade_result = self.execute_trade(
                                    symbol=symbol,
                                    account_name=account_name,
                                    account_type=account_type,
                                    auto_amount=env_config.auto_amount,
                                    trailing_percent=env_config.trailing_percent
                                )

                                # Save trade result
                                filename = self.save_trade_result(trade_result, date_str)
                                if filename:
                                    trade_result["saved_file"] = filename

                                trade_results.append(trade_result)
                                self.trades_executed += 1

                                # Display trade result
                                self._display_trade_result(trade_result, filename)
                        else:
                            msg = f"Trade limit reached ({self.max_trades}), skipping {account_name}/{account_type}"
                            self.logger.info(msg)
                    else:
                        self.logger.debug(f"Auto-trade disabled for {account_name}/{account_type}")

            return trade_results

        except Exception as e:
            self.logger.error(f"Error processing trades for {date_str}: {e}")
            return []

    def _display_trade_result(self, trade_result: Dict[str, Any], filename: Optional[str]) -> None:
        """Display trade result information."""
        try:
            symbol = trade_result['symbol']
            account_name = trade_result['account_name']
            account_type = trade_result['account_type']
            success = trade_result['success']
            auto_amount = trade_result.get('auto_amount', 'unknown')
            trailing_percent = trade_result.get('trailing_percent', 'unknown')

            message = (f"🎯 TRADE EXECUTED: {symbol} ({account_name}/{account_type})\n"
                       f"   Amount: ${auto_amount} | Trailing: {trailing_percent}% | Success: {success.upper()}\n"
                       f"   Message: {trade_result.get('message', 'No message')}")

            if filename:
                message += f"\n   Saved: {filename}"

            if self.test_mode:
                print(f"[TEST MODE] {message}")
            else:
                print(message)

            self.logger.info(f"Trade result displayed for {symbol} - {account_name}/{account_type}")

        except Exception as e:
            self.logger.error(f"Error displaying trade result: {e}")

    def run_today(self) -> List[Dict[str, Any]]:
        """
        Run trade generator for today's date.

        Returns:
            List of trade results
        """
        et_tz = pytz.timezone('US/Eastern')
        today = datetime.now(et_tz).strftime('%Y-%m-%d')

        self.logger.info(f"Running trade generator for {today}")
        return self.process_trades_for_date(today)

    def run_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Run trade generator for a specific date.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            List of trade results
        """
        self.logger.info(f"Running trade generator for {date_str}")
        return self.process_trades_for_date(date_str)
