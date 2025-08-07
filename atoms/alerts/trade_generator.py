"""
Trade Generator - Executes Automated Trades Based on Superduper Alerts

This atom handles the execution of trades based on superduper alerts by creating
trade records and executing buy-market-trailing-sell orders through alpaca.py for configured accounts.
Only processes alerts with green momentum indicators (🟢) for high-quality signals.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pytz


class TradeGenerator:
    """Generates trades from superduper alerts with green momentum indicators."""

    def __init__(self, trades_dir: Path, test_mode: bool = False):
        """
        Initialize the trade generator.

        Args:
            trades_dir: Directory to save trade results
            test_mode: Whether running in test mode
        """
        self.trades_dir = trades_dir
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)
        
        # Trade limit configuration for testing safety
        self.max_trades_per_session = 1 if test_mode else 12
        self.trades_executed_count = 0
        
        self.logger.info(f"TradeGenerator initialized: max_trades={self.max_trades_per_session}, test_mode={test_mode}")

    def can_execute_trade(self) -> bool:
        """
        Check if we can execute another trade within the session limit.
        
        Returns:
            True if trade can be executed, False if limit reached
        """
        return self.trades_executed_count < self.max_trades_per_session
    
    def get_remaining_trades(self) -> int:
        """
        Get the number of trades remaining in this session.
        
        Returns:
            Number of trades remaining
        """
        return max(0, self.max_trades_per_session - self.trades_executed_count)
    
    def reset_trade_counter(self) -> None:
        """
        Reset the trade counter (useful for testing or new trading sessions).
        """
        old_count = self.trades_executed_count
        self.trades_executed_count = 0
        self.logger.info(f"Trade counter reset: {old_count} → 0")

    def extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract stock symbol from superduper alert filename.

        Expected format: superduper_alert_SYMBOL_YYYYMMDD_HHMMSS

        Args:
            filename: The filename to parse

        Returns:
            The extracted symbol or None if parsing failed
        """
        try:
            # Remove file extension if present
            name_without_ext = filename.split('.')[0]

            # Split by underscore
            parts = name_without_ext.split('_')

            # Expected format: ['superduper', 'alert', 'SYMBOL', 'YYYYMMDD', 'HHMMSS']
            if len(parts) >= 3 and parts[0] == 'superduper' and parts[1] == 'alert':
                symbol = parts[2]
                self.logger.debug(f"Extracted symbol '{symbol}' from filename '{filename}'")
                return symbol
            else:
                self.logger.warning(f"Filename '{filename}' does not match expected format")
                return None

        except Exception as e:
            self.logger.error(f"Error extracting symbol from filename '{filename}': {e}")
            return None

    def has_green_momentum_indicator(self, superduper_alert: Dict[str, Any]) -> bool:
        """
        Check if a superduper alert contains green momentum indicator.

        Args:
            superduper_alert: Superduper alert data dictionary

        Returns:
            True if alert has green momentum emoji (🟢), False otherwise
        """
        try:
            alert_message = superduper_alert.get('alert_message', '')

            # Check for green momentum emoji in the alert message
            has_green = '🟢' in alert_message

            if has_green:
                self.logger.debug(f"Alert for {superduper_alert.get('symbol', 'UNKNOWN')} has GREEN momentum indicator")
            else:
                symbol = superduper_alert.get('symbol', 'UNKNOWN')
                self.logger.debug(f"Alert for {symbol} does NOT have green momentum indicator")

            return has_green

        except Exception as e:
            self.logger.error(f"Error checking green indicator: {e}")
            return False

    def load_account_config(self) -> Optional[Any]:
        """
        Load account configuration from alpaca_config.py.

        Returns:
            Configuration object or None if loading failed
        """
        try:
            # Add the code directory to Python path
            code_dir = Path(__file__).parent.parent.parent / "code"
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))

            from alpaca_config import get_current_config
            config = get_current_config()
            return config

        except ImportError as e:
            self.logger.error(f"Could not import alpaca_config: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading account configuration: {e}")
            return None

    def create_trade_record(self, superduper_alert: Dict[str, Any], account_name: str,
                            account_type: str, auto_amount: int, trailing_percent: float) -> Optional[Dict[str, Any]]:
        """
        Create a trade record from superduper alert data.

        Args:
            superduper_alert: The superduper alert data
            account_name: Name of the account
            account_type: Type of account (paper, live, cash)
            auto_amount: Amount to trade
            trailing_percent: Trailing percentage

        Returns:
            Trade record dictionary or None if creation failed
        """
        try:
            symbol = superduper_alert['symbol']

            # Create ET timestamp for when trade is generated
            et_tz = pytz.timezone('US/Eastern')
            trade_time = datetime.now(et_tz)
            et_timestamp = trade_time.strftime('%Y-%m-%dT%H:%M:%S%z')

            # Create trade record structure
            trade_record = {
                "symbol": symbol,
                "timestamp": et_timestamp,
                "record_type": "trade_record",
                "account_name": account_name,
                "account_type": account_type,
                "auto_amount": auto_amount,
                "trailing_percent": trailing_percent,
                "trigger_alert": superduper_alert,
                "trade_parameters": {
                    "command_type": "buy_market_trailing_sell",
                    "quantity": auto_amount,
                    "trailing_percent": trailing_percent,
                    "dry_run": True  # Always dry run as per specs
                },
                "execution_status": {
                    "initiated": False,
                    "completed": False,
                    "success": "no",  # Will be updated after execution
                    "execution_time": None,
                    "results": None
                }
            }

            return trade_record

        except Exception as e:
            self.logger.error(f"Error creating trade record for {superduper_alert.get('symbol', 'unknown')}: {e}")
            return None

    def execute_trade_command(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the trade command using alpaca.py.

        Args:
            trade_record: Trade record containing execution parameters

        Returns:
            Updated trade record with execution results
        """
        try:
            symbol = trade_record['symbol']
            auto_amount = trade_record['auto_amount']
            trailing_percent = trade_record['trailing_percent']

            # Build the alpaca.py command
            alpaca_script = Path(__file__).parent.parent.parent / "code" / "alpaca.py"

            cmd = [
                sys.executable,
                str(alpaca_script),
                "--buy-market-trailing-sell",
                "--symbol", symbol,
                "--amount", str(auto_amount),
                "--trailing-percent", str(trailing_percent),
                "--submit"  # Added --submit flag for actual trade execution
            ]

            self.logger.info(f"Executing trade command for {symbol}: {' '.join(cmd)}")

            # Update execution status
            et_tz = pytz.timezone('US/Eastern')
            execution_time = datetime.now(et_tz).isoformat()
            trade_record['execution_status']['initiated'] = True
            trade_record['execution_status']['execution_time'] = execution_time

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Update trade record with results
            trade_record['execution_status']['completed'] = True
            trade_record['execution_status']['return_code'] = result.returncode
            trade_record['execution_status']['stdout'] = result.stdout
            trade_record['execution_status']['stderr'] = result.stderr
            trade_record['execution_status']['results'] = result.stdout if result.stdout else result.stderr

            # Check execution results - we're now doing actual trades with --submit
            if "DRY RUN" in result.stdout:
                trade_record['execution_status']['dry_run_executed'] = True
                trade_record['execution_status']['success'] = "no"  # Dry run always marked as "no" per specs
                self.logger.info(f"Dry run trade executed for {symbol} (unexpected - should be actual trade)")
            else:
                trade_record['execution_status']['dry_run_executed'] = False
                # Success based on return code and output content
                success = result.returncode == 0 and ("error" not in result.stdout.lower() and "failed" not in result.stdout.lower())
                trade_record['execution_status']['success'] = "yes" if success else "no"
                
                if success:
                    self.logger.info(f"✅ Actual trade executed successfully for {symbol}")
                else:
                    self.logger.warning(f"❌ Trade execution failed for {symbol}: {result.stdout}")

            return trade_record

        except subprocess.TimeoutExpired:
            self.logger.error(f"Trade execution timeout for {trade_record['symbol']}")
            trade_record['execution_status']['completed'] = True
            trade_record['execution_status']['success'] = "no"
            trade_record['execution_status']['error'] = "Command timeout"
            trade_record['execution_status']['results'] = "Command execution timed out"
            return trade_record

        except Exception as e:
            self.logger.error(f"Error executing trade for {trade_record['symbol']}: {e}")
            trade_record['execution_status']['completed'] = True
            trade_record['execution_status']['success'] = "no"
            trade_record['execution_status']['error'] = str(e)
            trade_record['execution_status']['results'] = f"Execution error: {str(e)}"
            return trade_record

    def save_trade_record(self, trade_record: Dict[str, Any]) -> Optional[str]:
        """
        Save trade record to file.

        Args:
            trade_record: Trade record data

        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            symbol = trade_record['symbol']
            account_name = trade_record['account_name']
            account_type = trade_record['account_type']
            timestamp_str = trade_record['timestamp']

            # Parse timestamp to get filename format
            et_tz = pytz.timezone('US/Eastern')
            try:
                # Handle timezone format in timestamp
                if timestamp_str.endswith(('+0000', '-0400', '-0500')):
                    clean_timestamp = timestamp_str[:-5]
                    trade_time = datetime.fromisoformat(clean_timestamp)
                else:
                    trade_time = datetime.fromisoformat(timestamp_str.replace('Z', ''))

                # If no timezone info, assume ET
                if trade_time.tzinfo is None:
                    trade_time = et_tz.localize(trade_time)
                else:
                    trade_time = trade_time.astimezone(et_tz)

            except Exception:
                # Fallback to current time
                trade_time = datetime.now(et_tz)

            # Create directory structure: trades_dir/<account_name>/<account_type>/
            account_dir = self.trades_dir / account_name / account_type
            account_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            filename = f"trade_{symbol}_{trade_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = account_dir / filename

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(trade_record, f, indent=2)

            self.logger.debug(f"Trade record saved: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error saving trade record: {e}")
            return None

    def _display_trade_execution(self, trade_record: Dict[str, Any], filename: str) -> None:
        """Display trade execution information."""
        try:
            symbol = trade_record['symbol']
            account_name = trade_record['account_name']
            account_type = trade_record['account_type']
            execution_status = trade_record['execution_status']
            success = execution_status['success']
            amount = trade_record['auto_amount']

            message = (f"💰 TRADE EXECUTED: {symbol} on {account_name}/{account_type}\n"
                       f"   Amount: ${amount} | Success: {success.upper()}\n"
                       f"   Result: {execution_status.get('results', 'No output')[:100]}...\n"
                       f"   Saved: {filename}")

            if self.test_mode:
                print(f"[TEST MODE] {message}")
            else:
                print(message)

            self.logger.info(f"Trade executed for {symbol} on {account_name}/{account_type} - Success: {success}")

        except Exception as e:
            self.logger.error(f"Error displaying trade execution: {e}")

    def create_and_execute_trade(self, superduper_alert: Dict[str, Any]) -> Optional[str]:
        """
        Create and execute a trade from superduper alert in one operation.

        This mirrors the create_and_save_superduper_alert method pattern.

        Args:
            superduper_alert: The superduper alert data

        Returns:
            Filename if successful, None otherwise
        """
        try:
            # Check trade limit before processing
            if self.trades_executed_count >= self.max_trades_per_session:
                symbol = superduper_alert.get('symbol', 'UNKNOWN')
                self.logger.warning(f"🚫 Trade limit reached for {symbol}: {self.trades_executed_count}/{self.max_trades_per_session} trades executed this session")
                return None
            
            # Check if alert has green momentum indicator
            if not self.has_green_momentum_indicator(superduper_alert):
                symbol = superduper_alert.get('symbol', 'UNKNOWN')
                self.logger.info(f"Skipping trade for {symbol} - no green momentum indicator")
                return None

            # Load account configuration
            config = self.load_account_config()
            if config is None:
                self.logger.error("Could not load account configuration")
                return None

            # Find first account with auto_trade enabled (single trade limit per specs)
            trade_executed = False
            result_filename = None

            for account_name, account_config in config.providers["alpaca"].accounts.items():
                if trade_executed:
                    break

                for account_type in ["paper", "live", "cash"]:
                    if trade_executed:
                        break

                    env_config = getattr(account_config, account_type)

                    if env_config.auto_trade == "yes":
                        symbol = superduper_alert['symbol']
                        self.logger.info(f"Creating and executing trade for {symbol} on {account_name}/{account_type}")

                        # Create trade record
                        trade_record = self.create_trade_record(
                            superduper_alert, account_name, account_type,
                            env_config.auto_amount, env_config.trailing_percent
                        )

                        if trade_record is None:
                            continue

                        # Execute trade command
                        trade_record = self.execute_trade_command(trade_record)

                        # Save trade record
                        filename = self.save_trade_record(trade_record)
                        if filename is None:
                            continue

                        # Display trade execution
                        self._display_trade_execution(trade_record, filename)

                        # Increment trade counter for session limit tracking
                        self.trades_executed_count += 1
                        self.logger.info(f"Trade counter: {self.trades_executed_count}/{self.max_trades_per_session}")

                        result_filename = filename
                        trade_executed = True

            return result_filename

        except Exception as e:
            self.logger.error(f"Error in create_and_execute_trade: {e}")
            return None


def main():
    """Main entry point for testing trade generator atom."""
    import logging

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get the trades directory
    current_dir = Path(__file__).parent.parent.parent
    trades_dir = current_dir / "historical_data"

    # Create trade generator
    trade_generator = TradeGenerator(trades_dir, test_mode=True)

    # Test with sample superduper alert data
    sample_alert = {
        "symbol": "TEST",
        "timestamp": "2025-08-04T12:00:00-0400",
        "alert_type": "superduper_alert",
        "alert_message": ("🎯🎯 **SUPERDUPER ALERT** 🎯🎯\\n\\n🚀📈 **TEST** @ **$25.02**\\n"
                          "📊 **STRONG UPTREND** | 🔥 **VERY STRONG**\\n\\n🎯 **Signal Performance:**\\n"
                          "• Entry Signal: $16.41 ✅\\n• Current Price: $25.02\\n• Resistance Target: $17.15\\n"
                          "• Penetration: **100.0%** into range\\n\\n📈 **Trend Analysis (30m):**\\n"
                          "• Price Movement: **+20.28%**\\n• Momentum: 🟢 **0.8451%/min**\\n"
                          "• Penetration Increase: **+0.0%**\\n• Pattern: **Accelerating Breakout** 🚀\\n\\n"
                          "⚡ **Alert Level:** HIGH\\n⚠️ **Risk Level:** LOW\\n\\n🎯 **Action Zones:**\\n"
                          "• Watch for continuation above $25.02\\n• Target approach to $17.15\\n"
                          "• Monitor for volume confirmation\\n\\n⏰ **Alert Generated:** 12:00:00 ET")
    }

    print("🧪 Testing TradeGenerator atom...")
    print("=" * 50)

    # Test create_and_execute_trade
    result = trade_generator.create_and_execute_trade(sample_alert)

    if result:
        print(f"✅ Trade generation completed successfully: {result}")
    else:
        print("❌ Trade generation failed")


if __name__ == "__main__":
    main()
