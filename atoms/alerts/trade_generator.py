"""
Trade Generator - Executes Automated Trades Based on Superduper Alerts

This atom handles the execution of trades based on superduper alerts by creating
trade records and executing buy-market-trailing-sell-take-profit-percent orders through alpaca.py for configured accounts.
Only processes alerts with green momentum indicators (🟢) for high-quality signals.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
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
        
        # Trade limit configuration per account (loaded from config)
        self.trades_executed_by_account = {}  # Track trades per account: {"account_name/account_type": count}
        
        # Session-wide trade limits (for backward compatibility with orb_alerts_trade_stocks.py)
        self.max_trades_per_session = 5 if not test_mode else 1  # Default session limit
        
        self.logger.info(f"TradeGenerator initialized: test_mode={test_mode}, max_trades_per_session={self.max_trades_per_session}")

    def can_execute_trade(self, account_name: str, account_type: str, env_config) -> bool:
        """
        Check if we can execute another trade for a specific account within the session limit.
        
        Args:
            account_name: Name of the account
            account_type: Type of account (paper, live, cash)
            env_config: Environment configuration with max_trades_per_day
        
        Returns:
            True if trade can be executed, False if limit reached
        """
        account_key = f"{account_name}/{account_type}"
        current_count = self.trades_executed_by_account.get(account_key, 0)
        max_trades = 1 if self.test_mode else env_config.max_trades_per_day
        return current_count < max_trades
    
    def get_remaining_trades(self, account_name: str = None, account_type: str = None, env_config=None) -> int:
        """
        Get the number of trades remaining in this session.
        
        Args:
            account_name: Name of the account (optional for backward compatibility)
            account_type: Type of account (paper, live, cash) (optional for backward compatibility)
            env_config: Environment configuration with max_trades_per_day (optional for backward compatibility)
        
        Returns:
            Number of trades remaining (session-wide if no params, account-specific if params provided)
        """
        if account_name and account_type and env_config:
            # Per-account remaining trades (original behavior)
            account_key = f"{account_name}/{account_type}"
            current_count = self.trades_executed_by_account.get(account_key, 0)
            max_trades = 1 if self.test_mode else env_config.max_trades_per_day
            return max(0, max_trades - current_count)
        else:
            # Session-wide remaining trades (for backward compatibility)
            total_executed = sum(self.trades_executed_by_account.values())
            return max(0, self.max_trades_per_session - total_executed)
    
    def reset_trade_counter(self, account_name: str = None, account_type: str = None) -> None:
        """
        Reset the trade counter (useful for testing or new trading sessions).
        
        Args:
            account_name: If specified, reset only this account. If None, reset all accounts.
            account_type: Account type to reset (required if account_name specified)
        """
        if account_name and account_type:
            account_key = f"{account_name}/{account_type}"
            old_count = self.trades_executed_by_account.get(account_key, 0)
            self.trades_executed_by_account[account_key] = 0
            self.logger.info(f"Trade counter reset for {account_key}: {old_count} → 0")
        else:
            old_counts = dict(self.trades_executed_by_account)
            self.trades_executed_by_account.clear()
            self.logger.info(f"All trade counters reset: {old_counts} → {{}}")

    def is_market_hours(self) -> bool:
        """
        Check if current ET time is within market trading hours.
        
        Market hours: Monday-Friday 9:30 AM - 4:00 PM ET
        
        Returns:
            True if within trading hours, False otherwise
        """
        try:
            et_tz = pytz.timezone('US/Eastern')
            current_et = datetime.now(et_tz)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if current_et.weekday() > 4:  # Saturday=5, Sunday=6
                self.logger.info(f"Market closed: Weekend (day {current_et.weekday()})")
                return False
            
            # Get current time as hour and minute
            current_time = current_et.time()
            
            # Market open: 9:30 AM ET
            market_open = current_et.replace(hour=9, minute=30, second=0, microsecond=0).time()
            
            # Market close: 4:00 PM ET
            market_close = current_et.replace(hour=16, minute=0, second=0, microsecond=0).time()
            
            # Check if current time is within market hours
            is_open = market_open <= current_time <= market_close
            
            if is_open:
                self.logger.debug(f"Market is OPEN: {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                self.logger.info(f"Market is CLOSED: {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')} (Hours: 9:30-16:00 ET)")
                
            return is_open
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            # Fail safe: don't trade if we can't determine market hours
            return False

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
                            account_type: str, auto_amount: int, trailing_percent: float, take_profit_percent: float) -> Optional[Dict[str, Any]]:
        """
        Create a trade record from superduper alert data.

        Args:
            superduper_alert: The superduper alert data
            account_name: Name of the account
            account_type: Type of account (paper, live, cash)
            auto_amount: Amount to trade
            trailing_percent: Trailing percentage
            take_profit_percent: Take profit percentage

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
                "take_profit_percent": take_profit_percent,
                "trigger_alert": superduper_alert,
                "trade_parameters": {
                    "command_type": "buy_market_trailing_sell_take_profit_percent",
                    "quantity": auto_amount,
                    "trailing_percent": trailing_percent,
                    "take_profit_percent": take_profit_percent,
                    "dry_run": True  # Always dry run as per specs
                },
                "execution_status": {
                    "initiated": False,
                    "completed": False,
                    "success": "no",  # Will be updated after execution
                    "reason": None,  # Will be populated with failure/success reason
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
        Execute the trade command using alpaca.py with retry logic for API timeouts.

        Args:
            trade_record: Trade record containing execution parameters

        Returns:
            Updated trade record with execution results
        """
        # Use the retry logic for trade execution
        return self._execute_trade_command_with_retries(trade_record, max_retries=3, retry_delay=2.0)

    def _is_retryable_error(self, stdout: str, stderr: str) -> bool:
        """
        Check if the error is retryable (API timeout/overload conditions).
        
        Args:
            stdout: Standard output from alpaca command
            stderr: Standard error from alpaca command
            
        Returns:
            True if the error is retryable, False otherwise
        """
        try:
            # Combine stdout and stderr for analysis
            full_output = f"{stdout}\n{stderr}".lower()
            
            # Retryable error patterns - API timeouts and overload conditions
            retryable_patterns = [
                r"timeout",
                r"server.*error",
                r"server error occurred", 
                r"connection.*error",
                r"rate.*limit",
                r"503",  # Service unavailable
                r"502",  # Bad gateway
                r"504",  # Gateway timeout
            ]
            
            # Check if any retryable pattern matches
            for pattern in retryable_patterns:
                if re.search(pattern, full_output):
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if error is retryable: {e}")
            return False

    def _execute_trade_command_with_retries(self, trade_record: Dict[str, Any], max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, Any]:
        """
        Execute trade command with retry logic for API timeouts/overloads.
        
        Args:
            trade_record: Trade record containing execution parameters
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 2.0)
            
        Returns:
            Updated trade record with execution results
        """
        symbol = trade_record['symbol']
        retry_attempts = []
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Trade execution attempt {attempt + 1}/{max_retries} for {symbol}")
                
                # Execute the trade command
                result_record = self._execute_single_trade_attempt(trade_record, attempt + 1)
                
                # Check if the execution was successful
                if result_record['execution_status']['success'] == "yes":
                    self.logger.info(f"✅ Trade execution succeeded on attempt {attempt + 1} for {symbol}")
                    # Add retry info to the record
                    if attempt > 0:
                        result_record['execution_status']['retry_attempts'] = retry_attempts
                        result_record['execution_status']['successful_attempt'] = attempt + 1
                    return result_record
                
                # Check if the error is retryable
                stdout = result_record['execution_status'].get('stdout', '')
                stderr = result_record['execution_status'].get('stderr', '')
                
                if not self._is_retryable_error(stdout, stderr):
                    self.logger.info(f"❌ Trade execution failed with non-retryable error for {symbol}: {result_record['execution_status']['reason']}")
                    # Add retry info even for non-retryable failures
                    if attempt > 0:
                        result_record['execution_status']['retry_attempts'] = retry_attempts
                    return result_record
                
                # Record this attempt
                retry_attempts.append({
                    'attempt': attempt + 1,
                    'reason': result_record['execution_status']['reason'],
                    'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat()
                })
                
                # If this is not the last attempt, wait before retrying
                if attempt < max_retries - 1:
                    self.logger.warning(f"⏱️ Retryable error for {symbol} on attempt {attempt + 1}: {result_record['execution_status']['reason']}")
                    self.logger.info(f"⏳ Waiting {retry_delay} seconds before retry {attempt + 2}...")
                    time.sleep(retry_delay)
                else:
                    # Final attempt failed
                    self.logger.error(f"❌ Trade execution failed after {max_retries} attempts for {symbol}: {result_record['execution_status']['reason']}")
                    result_record['execution_status']['retry_attempts'] = retry_attempts
                    result_record['execution_status']['max_retries_reached'] = True
                    return result_record
                    
            except Exception as e:
                error_msg = f"Exception during trade execution attempt {attempt + 1}: {str(e)}"
                self.logger.error(f"❌ {error_msg} for {symbol}")
                
                retry_attempts.append({
                    'attempt': attempt + 1,
                    'reason': error_msg,
                    'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat()
                })
                
                # If this is the last attempt or a non-retryable exception
                if attempt == max_retries - 1:
                    trade_record['execution_status']['completed'] = True
                    trade_record['execution_status']['success'] = "no"
                    trade_record['execution_status']['reason'] = f"Failed after {max_retries} attempts: {error_msg}"
                    trade_record['execution_status']['retry_attempts'] = retry_attempts
                    trade_record['execution_status']['max_retries_reached'] = True
                    trade_record['execution_status']['error'] = str(e)
                    return trade_record
                else:
                    self.logger.info(f"⏳ Waiting {retry_delay} seconds before retry {attempt + 2} after exception...")
                    time.sleep(retry_delay)
        
        # This should never be reached, but just in case
        return trade_record

    def _execute_single_trade_attempt(self, trade_record: Dict[str, Any], attempt_number: int) -> Dict[str, Any]:
        """
        Execute a single trade attempt (extracted from original execute_trade_command).
        
        Args:
            trade_record: Trade record containing execution parameters
            attempt_number: Current attempt number
            
        Returns:
            Updated trade record with execution results
        """
        try:
            symbol = trade_record['symbol']
            auto_amount = trade_record['auto_amount']
            trailing_percent = trade_record['trailing_percent']
            take_profit_percent = trade_record['take_profit_percent']

            # Build the alpaca.py command
            alpaca_script = Path(__file__).parent.parent.parent / "code" / "alpaca.py"

            cmd = [
                sys.executable,
                str(alpaca_script),
                "--buy-market-trailing-sell-take-profit-percent",
                "--symbol", symbol,
                "--amount", str(auto_amount),
                "--trailing-percent", str(trailing_percent),
                "--take-profit-percent", str(take_profit_percent),
                "--submit"  # Added --submit flag for actual trade execution
            ]

            if attempt_number == 1:
                self.logger.info(f"Executing trade command for {symbol}: {' '.join(cmd)}")
            else:
                self.logger.info(f"Retry {attempt_number}: Executing trade command for {symbol}")

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
                trade_record['execution_status']['reason'] = "Dry run execution (not actual trade)"
                self.logger.info(f"Dry run trade executed for {symbol} (unexpected - should be actual trade)")
            else:
                trade_record['execution_status']['dry_run_executed'] = False
                # Success based on return code and output content
                success = result.returncode == 0 and ("error" not in result.stdout.lower() and "failed" not in result.stdout.lower())
                trade_record['execution_status']['success'] = "yes" if success else "no"
                
                if success:
                    trade_record['execution_status']['reason'] = "Trade executed successfully"
                    if attempt_number > 1:
                        self.logger.info(f"✅ Trade executed successfully for {symbol} on retry attempt {attempt_number}")
                    else:
                        self.logger.info(f"✅ Actual trade executed successfully for {symbol}")
                else:
                    # Extract failure reason from alpaca output
                    failure_reason = self._extract_failure_reason(result.stdout, result.stderr)
                    trade_record['execution_status']['reason'] = failure_reason
                    self.logger.warning(f"❌ Trade execution failed for {symbol} on attempt {attempt_number}: {failure_reason}")

            return trade_record

        except subprocess.TimeoutExpired:
            self.logger.error(f"Trade execution timeout for {trade_record['symbol']} on attempt {attempt_number}")
            trade_record['execution_status']['completed'] = True
            trade_record['execution_status']['success'] = "no"
            trade_record['execution_status']['reason'] = "Command execution timed out (60 seconds)"
            trade_record['execution_status']['error'] = "Command timeout"
            trade_record['execution_status']['results'] = "Command execution timed out"
            return trade_record

        except Exception as e:
            self.logger.error(f"Error executing trade for {trade_record['symbol']} on attempt {attempt_number}: {e}")
            trade_record['execution_status']['completed'] = True
            trade_record['execution_status']['success'] = "no"
            trade_record['execution_status']['reason'] = f"Execution error: {str(e)}"
            trade_record['execution_status']['error'] = str(e)
            trade_record['execution_status']['results'] = f"Execution error: {str(e)}"
            return trade_record

    def _extract_failure_reason(self, stdout: str, stderr: str) -> str:
        """
        Extract meaningful failure reason from alpaca command output.
        
        Args:
            stdout: Standard output from alpaca command
            stderr: Standard error from alpaca command
            
        Returns:
            Human-readable failure reason
        """
        try:
            # Combine stdout and stderr for analysis
            full_output = f"{stdout}\n{stderr}".lower()
            
            # Common failure patterns and their human-readable explanations
            failure_patterns = [
                # Market/Quote related errors
                (r"no quote found for", "Market data unavailable (symbol may not exist or market closed)"),
                (r"market is closed", "Market is closed - orders cannot be executed"),
                (r"market hours", "Outside market trading hours"),
                
                # Account/Authentication errors
                (r"unauthorized", "Authentication failed - check API credentials"),
                (r"insufficient.*funds", "Insufficient funds in account"),
                (r"buying power", "Insufficient funds in account"),
                (r"account.*restricted", "Account trading restrictions in place"),
                
                # Order validation errors
                (r"invalid.*symbol", "Invalid or unrecognized stock symbol"),
                (r"minimum.*quantity", "Order quantity below minimum requirements"),
                (r"maximum.*quantity", "Order quantity exceeds maximum allowed"),
                (r"invalid.*price", "Invalid order price specified"),
                
                # API/Network errors
                (r"connection.*error", "Network connection error to Alpaca API"),
                (r"timeout", "Request timeout - API may be overloaded"),
                (r"rate.*limit", "API rate limit exceeded"),
                (r"server.*error", "Alpaca server error (HTTP 5xx)"),
                (r"server error occurred", "Alpaca server error (HTTP 5xx)"),
                
                # Order specific errors
                (r"order.*rejected", "Order rejected by broker"),
                (r"order.*canceled", "Order was canceled"),
                (r"failed.*to.*submit", "Failed to submit order to broker"),
                (r"position.*not.*found", "Position does not exist for trailing sell"),
                
                # General trading errors
                (r"trading.*halted", "Trading halted for this symbol"),
                (r"not.*tradable", "Symbol is not tradable"),
                (r"fractional.*shares", "Fractional shares not supported for this symbol"),
            ]
            
            # Search for known patterns
            for pattern, reason in failure_patterns:
                if re.search(pattern, full_output):
                    return reason
            
            # Extract specific error messages from stderr
            if stderr:
                # Look for APIError messages
                api_error_match = re.search(r"APIError:\s*(.+?)(?:\n|$)", stderr)
                if api_error_match:
                    return f"API Error: {api_error_match.group(1).strip()}"
                
                # Look for Exception messages
                exception_match = re.search(r"Exception:\s*(.+?)(?:\n|$)", stderr)
                if exception_match:
                    return f"Exception: {exception_match.group(1).strip()}"
                
                # Look for generic error patterns
                error_match = re.search(r"error:\s*(.+?)(?:\n|$)", stderr, re.IGNORECASE)
                if error_match:
                    return f"Error: {error_match.group(1).strip()}"
            
            # Check for specific failed operations in stdout
            if "failed" in stdout.lower():
                failed_match = re.search(r"✗\s*(.+?)(?:\n|$)", stdout)
                if failed_match:
                    return f"Operation failed: {failed_match.group(1).strip()}"
            
            # If no specific pattern matched, return generic failure with snippet
            output_snippet = (stdout[:100] if stdout else stderr[:100] if stderr else "No output").strip()
            return f"Trade execution failed: {output_snippet}{'...' if len(output_snippet) == 100 else ''}"
            
        except Exception as e:
            return f"Failed to parse error reason: {str(e)}"

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

            reason = execution_status.get('reason', 'No reason provided')
            message = (f"💰 TRADE EXECUTED: {symbol} on {account_name}/{account_type}\n"
                       f"   Amount: ${amount} | Success: {success.upper()}\n"
                       f"   Reason: {reason}\n"
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
            # Check session-wide trade limit before processing
            total_executed = sum(self.trades_executed_by_account.values())
            if total_executed >= self.max_trades_per_session:
                symbol = superduper_alert.get('symbol', 'UNKNOWN')
                self.logger.warning(f"🚫 Trade limit reached for {symbol}: {total_executed}/{self.max_trades_per_session} trades executed this session")
                return None
            
            # Check if alert has green momentum indicator
            if not self.has_green_momentum_indicator(superduper_alert):
                symbol = superduper_alert.get('symbol', 'UNKNOWN')
                self.logger.info(f"Skipping trade for {symbol} - no green momentum indicator")
                return None

            # Check market hours before executing trade
            if not self.is_market_hours():
                symbol = superduper_alert.get('symbol', 'UNKNOWN')
                self.logger.info(f"🚫 Trade rejected for {symbol} - market is closed (outside trading hours)")
                return None

            # Load account configuration
            config = self.load_account_config()
            if config is None:
                self.logger.error("Could not load account configuration")
                return None

            # Execute trades on all accounts with auto_trade enabled
            executed_trades = []
            successful_trades = []

            for account_name, account_config in config.providers["alpaca"].accounts.items():
                for account_type in ["paper", "live", "cash"]:
                    env_config = getattr(account_config, account_type)

                    if env_config.auto_trade == "yes":
                        symbol = superduper_alert['symbol']
                        
                        # Check if this account can still execute trades
                        if not self.can_execute_trade(account_name, account_type, env_config):
                            remaining = self.get_remaining_trades(account_name, account_type, env_config)
                            max_trades = 1 if self.test_mode else env_config.max_trades_per_day
                            current_count = self.trades_executed_by_account.get(f"{account_name}/{account_type}", 0)
                            self.logger.warning(f"🚫 Trade limit reached for {account_name}/{account_type}: {current_count}/{max_trades} trades executed (remaining: {remaining})")
                            continue
                        
                        self.logger.info(f"Creating and executing trade for {symbol} on {account_name}/{account_type}")

                        # Create trade record
                        trade_record = self.create_trade_record(
                            superduper_alert, account_name, account_type,
                            env_config.auto_amount, env_config.trailing_percent, env_config.take_profit_percent
                        )

                        if trade_record is None:
                            self.logger.warning(f"Failed to create trade record for {account_name}/{account_type}")
                            continue

                        # Execute trade command
                        trade_record = self.execute_trade_command(trade_record)

                        # Save trade record
                        filename = self.save_trade_record(trade_record)
                        if filename is None:
                            self.logger.warning(f"Failed to save trade record for {account_name}/{account_type}")
                            continue

                        # Display trade execution
                        self._display_trade_execution(trade_record, filename)

                        # Increment trade counter for this specific account
                        account_key = f"{account_name}/{account_type}"
                        self.trades_executed_by_account[account_key] = self.trades_executed_by_account.get(account_key, 0) + 1
                        remaining = self.get_remaining_trades(account_name, account_type, env_config)
                        max_trades = 1 if self.test_mode else env_config.max_trades_per_day
                        current_count = self.trades_executed_by_account[account_key]
                        self.logger.info(f"Trade counter for {account_key}: {current_count}/{max_trades} (remaining: {remaining})")

                        # Track all executed trades
                        executed_trades.append(filename)
                        
                        # Track successful trades separately
                        if trade_record['execution_status']['success'] == "yes":
                            successful_trades.append(filename)
                            self.logger.info(f"✅ Successful trade executed on {account_name}/{account_type}: {filename}")
                        else:
                            self.logger.warning(f"❌ Trade failed on {account_name}/{account_type}: {trade_record['execution_status']['reason']}")

            # Log summary of all trade attempts
            if executed_trades:
                self.logger.info(f"Trade execution summary: {len(executed_trades)} total attempts, {len(successful_trades)} successful")
                # Return all executed trades for Telegram notification
                return executed_trades
            else:
                self.logger.info("No trades were executed (no accounts configured for auto_trade or all limits reached)")
                return []

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
                          "• Watch for continuation above $25.02\\n• Watch for major resistance\\n"
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
