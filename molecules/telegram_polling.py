#!/usr/bin/env python3
"""
Telegram Polling Service for Automated User Management

This service continuously polls the Telegram API for new messages and automatically
manages user subscriptions based on bot commands like /start, /stop, etc.
"""

import io
import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.telegram.config import TelegramConfig
from atoms.telegram.send_image import TelegramImageSender
from atoms.telegram.telegram_post import send_alert, send_message
from atoms.telegram.user_manager import UserManager


class TelegramPollingService:
    """Service for polling Telegram messages and managing user subscriptions."""

    def __init__(self):
        self.user_manager = UserManager()
        self.config = TelegramConfig()
        self.image_sender = TelegramImageSender()
        self.running = False
        self.last_update_id = 0
        self.poll_interval = 5  # seconds
        self.base_url = f"https://api.telegram.org/bot{self.config.BOT_TOKEN}"

        # Command handlers (only for slash commands)
        self.command_handlers = {
            '/start': self._handle_start,
            '/stop': self._handle_stop,
            '/subscribe': self._handle_subscribe,
            '/unsubscribe': self._handle_unsubscribe,
            '/status': self._handle_status,
            '/help': self._handle_help,
            '/newbie': self._handle_newbie
        }

        # Authorized users for Alpaca commands
        self.authorized_users = ['bruce']  # Case insensitive matching

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        sys.stdout.flush()  # Force output to be written immediately

    def start_polling(self):
        """Start the polling service."""
        try:
            # Validate configuration
            self.config.validate_config()
            self._log("🔧 Configuration validated successfully")

            # Get bot info
            bot_info = self._get_bot_info()
            if bot_info:
                self._log(f"🤖 Bot connected: @{bot_info.get('username')} ({bot_info.get('first_name')})")
            else:
                self._log("❌ Failed to connect to bot", "ERROR")
                return False

            self.running = True
            self._log("🚀 Starting Telegram polling service...")
            self._log(f"📊 Poll interval: {self.poll_interval} seconds")
            self._log("💡 Send CTRL+C to stop")

            # Main polling loop
            while self.running:
                try:
                    self._poll_updates()
                    time.sleep(self.poll_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self._log(f"❌ Polling error: {e}", "ERROR")
                    time.sleep(self.poll_interval * 2)  # Wait longer on error

            self._log("🛑 Polling service stopped")
            return True

        except Exception as e:
            self._log(f"❌ Failed to start polling service: {e}", "ERROR")
            return False

    def _get_bot_info(self) -> Optional[Dict]:
        """Get bot information from Telegram API."""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    return result.get('result')

            return None

        except Exception as e:
            self._log(f"Error getting bot info: {e}", "ERROR")
            return None

    def _poll_updates(self):
        """Poll for new updates from Telegram."""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'limit': 100,
                'timeout': 30
            }

            response = requests.get(url, params=params, timeout=35)

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    updates = result.get('result', [])

                    for update in updates:
                        self._process_update(update)
                        self.last_update_id = max(self.last_update_id, update.get('update_id', 0))
                else:
                    self._log(f"API error: {result.get('description')}", "ERROR")
            else:
                self._log(f"HTTP error {response.status_code}: {response.text}", "ERROR")

        except requests.exceptions.Timeout:
            # Timeout is normal, just continue
            pass
        except Exception as e:
            self._log(f"Error polling updates: {e}", "ERROR")

    def _process_update(self, update: Dict):
        """Process a single update from Telegram."""
        try:
            message = update.get('message', {})
            if not message:
                return

            chat = message.get('chat', {})
            from_user = message.get('from', {})
            text = message.get('text', '').strip()

            chat_id = str(chat.get('id', ''))
            username = from_user.get('username', '')
            first_name = from_user.get('first_name', '')
            last_name = from_user.get('last_name', '')

            if not chat_id:
                return

            # Log the message
            display_name = f"{first_name} {last_name}".strip() or username or chat_id
            self._log(f"📨 Message from {display_name} ({chat_id}): {text}")

            # Process commands and trigger words
            if text.startswith('/'):
                command = text.split()[0].lower()
                if command in self.command_handlers:
                    self.command_handlers[command](chat_id, username, first_name, last_name, text)
                else:
                    self._handle_unknown_command(chat_id, text)
            elif text.lower().startswith('57chevy'):
                # Handle alpaca trigger word (authorized users only)
                self._handle_alpaca_command(chat_id, username, first_name, last_name, text)
            elif text.lower().startswith('plot'):
                # Handle plot trigger word (any user)
                self._handle_plot_command(chat_id, username, first_name, last_name, text)
            elif text.lower() == 'bam':
                # Handle bam trigger word (authorized users only)
                self._handle_bam_command(chat_id, username, first_name, last_name, text)
            elif text.lower() == 'signal':
                # Handle signal trigger word (any user)
                self._handle_signal_command(chat_id, username, first_name, last_name, text)
            elif text.lower() == 'volume surge':
                # Handle volume surge trigger word (any user)
                self._handle_volume_surge_command(chat_id, username, first_name, last_name, text)
            elif text.lower() == 'top gainers':
                # Handle top gainers trigger word (any user)
                self._handle_top_gainers_command(chat_id, username, first_name, last_name, text)
            elif text.lower() == 'premarket top gainers':
                # Handle premarket top gainers trigger word (any user)
                self._handle_premarket_top_gainers_command(chat_id, username, first_name, last_name, text)
            elif text.lower() == 'market open top gainers':
                # Handle market open top gainers trigger word (any user)
                self._handle_market_open_top_gainers_command(chat_id, username, first_name, last_name, text)
            elif text.lower().startswith('configure alpaca'):
                # Handle configure alpaca trigger word (Bruce only)
                self._handle_configure_alpaca_command(chat_id, username, first_name, last_name, text)
            else:
                # Handle non-command messages
                self._handle_regular_message(chat_id, username, first_name, text)

        except Exception as e:
            self._log(f"Error processing update: {e}", "ERROR")

    def _handle_start(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle /start command."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"

            # Try to add/enable user (handles both new users and re-enabling existing users)
            success = self.user_manager.add_user(
                chat_id=chat_id,
                username=display_name,
                enabled=True,
                notes=f"Auto-added via /start on {datetime.now().strftime('%Y-%m-%d')}"
            )

            if success:
                # Check if they were already active
                active_users = self.user_manager.get_active_users()
                is_existing = any(u['chat_id'] == chat_id for u in active_users)

                if is_existing:
                    response = f"✅ Welcome back, {display_name}! You're now subscribed to trading alerts."
                else:
                    response = f"🎉 Welcome, {display_name}! You've been subscribed to trading alerts.\n\n" + \
                              "You'll receive:\n" + \
                              "🚀 ORB breakout signals\n" + \
                              "📊 Trade notifications\n" + \
                              "📈 Portfolio updates\n\n" + \
                              "Send /stop to unsubscribe anytime."

                self._log(f"✅ User subscribed: {display_name} ({chat_id})")
            else:
                response = "❌ Sorry, there was an error subscribing you to alerts. Please try again."
                self._log(f"❌ Failed to subscribe user: {display_name} ({chat_id})", "ERROR")

            self._send_response(chat_id, response)

        except Exception as e:
            self._log(f"Error handling /start: {e}", "ERROR")
            self._send_response(chat_id, "❌ Sorry, there was an error processing your request.")

    def _handle_stop(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle /stop command."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"

            # Disable user
            success = self.user_manager.disable_user(chat_id)

            if success:
                response = f"🔕 {display_name}, you've been unsubscribed from trading alerts.\n\n" + \
                          "Send /start anytime to resubscribe."
                self._log(f"🔕 Disabled user: {display_name} ({chat_id})")
            else:
                response = "❌ You weren't subscribed to alerts, or there was an error."

            self._send_response(chat_id, response)

        except Exception as e:
            self._log(f"Error handling /stop: {e}", "ERROR")
            self._send_response(chat_id, "❌ Sorry, there was an error processing your request.")

    def _handle_subscribe(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle /subscribe command (same as /start)."""
        self._handle_start(chat_id, username, first_name, last_name, text)

    def _handle_unsubscribe(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle /unsubscribe command (same as /stop)."""
        self._handle_stop(chat_id, username, first_name, last_name, text)

    def _handle_status(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle /status command."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"

            # Check user status
            active_users = self.user_manager.get_active_users()
            is_active = any(u['chat_id'] == chat_id for u in active_users)

            if is_active:
                response = f"✅ {display_name}, you're currently subscribed to trading alerts.\n\n" + \
                          "Send /stop to unsubscribe."
            else:
                response = f"🔕 {display_name}, you're not currently subscribed to trading alerts.\n\n" + \
                          "Send /start to subscribe."

            self._send_response(chat_id, response)

        except Exception as e:
            self._log(f"Error handling /status: {e}", "ERROR")
            self._send_response(chat_id, "❌ Sorry, there was an error checking your status.")

    def _handle_help(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle /help command."""
        # Base help text for all users
        help_text = """🤖 Trading Alerts Bot Help

Available Commands:
/start - Subscribe to trading alerts
/stop - Unsubscribe from alerts
/subscribe - Same as /start
/unsubscribe - Same as /stop
/status - Check your subscription status
/help - Show this help message
/newbie - New trader welcome guide and tips

📊 Chart Generation:
plot -plot -symbol [TICKER] [-date YYYY-MM-DD] - Generate and send chart for any stock symbol
Examples:
  • plot -plot -symbol AAPL
  • plot -plot -symbol TSLA -date 2025-08-15
  • plot -plot -symbol MSFT

Note: Use regular hyphens (-) in the command

📈 Signal Analysis:
signal - Run oracle signal analysis to identify stocks near support levels

📈 Volume Surge Scanner:
volume surge - Run comprehensive volume surge analysis on NASDAQ/AMEX stocks
  • Scans up to 7,000 symbols
  • Filters stocks $0.75-$40.00 with 250K+ volume
  • Identifies 5%+ price changes with 5x volume surge over 50 days
  • Returns detailed CSV results (may take up to 10 minutes)

🚀 Top Gainers Scanner:
top gainers - Find the top 40 gaining stocks on NASDAQ/AMEX exchanges
  • Scans up to 7,000 symbols
  • Filters stocks $0.75-$40.00 with 250K+ volume
  • Returns top 40 gainers with detailed metrics
  • Returns formatted CSV results (may take up to 10 minutes)

🌅 Premarket Top Gainers Scanner:
premarket top gainers - Find top 40 premarket gaining stocks on NASDAQ/AMEX
  • Scans up to 7,000 symbols for premarket activity
  • Filters stocks $0.75-$40.00 with 250K+ volume
  • Analyzes 5-minute candles since last market close
  • Returns detailed premarket CSV results (may take up to 10 minutes)

📈 Market Open Top Gainers Scanner:
market open top gainers - Find top 40 market open gaining stocks on NASDAQ/AMEX
  • Scans up to 7,000 symbols for market activity since open
  • Filters stocks $0.75-$40.00 with 250K+ volume
  • Analyzes 1-minute candles from market open to current/close
  • Returns detailed market CSV results (may take up to 15 minutes)"""

        # Add admin commands section only for Bruce
        if self._is_authorized_user(username, first_name):
            help_text += """

🚨 Admin Commands (Bruce only):
bam - Execute Bulk Account Manager (liquidate all auto-trade positions/orders)

🔧 Configuration Management:
configure alpaca - Modify Alpaca trading configuration parameters
Usage: configure alpaca --account-name [NAME] --account [paper/live/cash] [options]

Options:
  --auto-trade [yes/no]
  --auto-amount [NUMBER]
  --trailing-percent [NUMBER]
  --take-profit-percent [NUMBER]
  --max-trades-per-day [NUMBER]
  --dry-run (preview changes)

Examples:
  configure alpaca --account-name Bruce --account paper --auto-amount 5000
  configure alpaca --account-name "Dale Wilson" --account live --auto-trade yes --dry-run"""

        # Add footer for all users
        help_text += """

You'll receive:
🚀 ORB breakout signals
📊 Trade notifications
📈 Portfolio updates
⚠️ System alerts

Questions? Contact the bot administrator."""

        self._send_response(chat_id, help_text)

    def _handle_newbie(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle /newbie command."""
        newbie_text = """**Welcome!**

Daytrading is frequently very fast. In and out of a stock in ten minutes is common.
I once made almost 9% in less than 90 seconds.

**Hot Keys**

Hot keys are a way to buy and sell quickly. These are supported by Lightning Trader,
Charles Schwab (Tos), and Webull. See https://www.youtube.com/watch?v=9hgpZMvCY08
for setup instructions.

**Scanner Commands**

'volume surge' and 'top gainers' function only after market open. If before market open,
the prior trading day results will be given.

**Best wishes!**"""

        self._send_response(chat_id, newbie_text)

    def _handle_unknown_command(self, chat_id: str, text: str):
        """Handle unknown commands."""
        response = f"❓ Unknown command: {text}\n\nSend /help to see available commands."
        self._send_response(chat_id, response)

    def _handle_regular_message(self, chat_id: str, username: str, first_name: str, text: str):
        """Handle non-command messages."""
        # Check for subscription keywords
        text_lower = text.lower()

        if any(word in text_lower for word in ['subscribe', 'join', 'alerts', 'start']):
            self._send_response(chat_id, "💡 To subscribe to trading alerts, send: /start")
        elif any(word in text_lower for word in ['unsubscribe', 'stop', 'leave']):
            self._send_response(chat_id, "💡 To unsubscribe from alerts, send: /stop")
        else:
            self._send_response(chat_id, "👋 Hello! Send /help to see available commands.")

    def _handle_plot_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle plot trigger for generating and sending charts."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"📊 Plot command from {display_name}: {text}")

            # Parse plot arguments from message
            args = self._parse_plot_args(text)
            if not args:
                error_msg = ("❌ Invalid plot command. Use: plot -plot -symbol [TICKER] "
                             "[-date YYYY-MM-DD]\nExamples:\n  • plot -plot -symbol AAPL\n"
                             "  • plot -plot -symbol AAPL -date 2025-08-15")
                self._send_response(chat_id, error_msg)
                return

            # Execute alpaca command to generate plot
            self._send_response(chat_id, f"📊 Generating chart for {args.get('symbol', 'symbol')}...")

            alpaca_output = self._execute_alpaca_command(args['alpaca_args'])

            # Extract plot path from output
            plot_path = self._extract_plot_path(alpaca_output)

            if plot_path:
                # Send the image using TelegramImageSender
                caption = (f"📊 Chart for {args.get('symbol', 'symbol')} - "
                           f"Generated for {display_name}")
                result = self.image_sender._send_image_to_chat(
                    chat_id=chat_id,
                    image_path=plot_path,
                    caption=caption,
                    urgent=False
                )

                if result:
                    self._log(f"✅ Chart sent successfully to {display_name}")
                else:
                    self._send_response(
                        chat_id,
                        f"❌ Failed to send chart image. Plot generated at: {plot_path}"
                    )
            else:
                # No plot path found, send the text output
                self._send_response(
                    chat_id,
                    f"❌ No chart generated. Output:\n```\n{alpaca_output}\n```"
                )

        except Exception as e:
            self._log(f"Error handling plot command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error generating plot: {str(e)}")

    def _handle_alpaca_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 57chevy trigger for Alpaca trading commands."""

        # Check user authorization
        if not self._is_authorized_user(username, first_name):
            return  # Silently ignore unauthorized users

        try:
            # Parse alpaca arguments from message
            args = self._parse_alpaca_args(text)

            # Execute alpaca command
            result = self._execute_alpaca_command(args)

            # Send result back to authorized user
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling alpaca command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error processing command: {str(e)}")

    def _handle_bam_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 'bam' command for bulk account management (Bruce only)."""

        # Check user authorization - only Bruce can use this command
        if not self._is_authorized_user(username, first_name):
            return  # Silently ignore unauthorized users

        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"🚨 BAM command from {display_name}: {text}")

            # Send confirmation message
            message = ("🚨 Executing BAM (Bulk Account Manager)...\n"
                       "This will liquidate all positions and cancel all orders "
                       "for auto-trade accounts.")
            self._send_response(chat_id, message)

            # Execute bam.py script
            result = self._execute_bam_command()

            # Send result back to Bruce
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling bam command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error executing BAM command: {str(e)}")

    def _handle_signal_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 'signal' command to run oracle_signal.py."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"📊 Signal command from {display_name}: {text}")

            # Send processing message
            self._send_response(chat_id, "📊 Running oracle signal analysis...")

            # Execute oracle_signal.py script
            result = self._execute_oracle_signal_command()

            # Send result back to user
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling signal command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error executing signal command: {str(e)}")

    def _handle_volume_surge_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 'volume surge' command to run alpaca_screener.py."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"📈 Volume surge command from {display_name}: {text}")

            # Send processing message
            message = "📈 Running volume surge scanner... This may take up to 10 minutes."
            self._send_response(chat_id, message)

            # Execute alpaca_screener.py script
            result = self._execute_volume_surge_command()

            # Send result back to user
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling volume surge command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error executing volume surge command: {str(e)}")

    def _handle_top_gainers_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 'top gainers' command to run alpaca_screener.py."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"🚀 Top gainers command from {display_name}: {text}")

            # Send processing message
            message = "🚀 Running top gainers scanner... This may take up to 10 minutes."
            self._send_response(chat_id, message)

            # Execute alpaca_screener.py script
            result = self._execute_top_gainers_command()

            # Send result back to user
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling top gainers command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error executing top gainers command: {str(e)}")

    def _is_authorized_user(self, username: str, first_name: str) -> bool:
        """Check if user is authorized for Alpaca commands."""
        return ((username and username.lower() in self.authorized_users) or
                (first_name and first_name.lower() in self.authorized_users))

    def _parse_alpaca_args(self, text: str) -> List[str]:
        """Parse Alpaca arguments from Telegram message."""
        # Remove trigger word and extract arguments
        parts = text.strip().split()[1:]  # Skip '57chevy'

        # Convert single hyphens to double hyphens for Alpaca CLI compatibility
        # Telegram changes -- to — (em dash) or - (single hyphen)
        converted_parts = []
        for part in parts:
            if part.startswith('-') and not part.startswith('--'):
                # Check if this is a negative number (starts with - followed by digits)
                if len(part) > 1 and (part[1:].replace('.', '').isdigit()):
                    # This is a negative number, keep as single hyphen
                    converted_parts.append(part)
                else:
                    # Convert single hyphen to double hyphen and normalize case for CLI arguments
                    # Special case: preserve PNL in uppercase as it's defined as --PNL in alpaca.py
                    if part[1:].upper() == 'PNL':  # Remove the leading hyphen before checking
                        converted_parts.append('--PNL')
                    else:
                        converted_parts.append(('-' + part).lower())
            elif part.startswith('--'):
                # Normalize existing double hyphen arguments to lowercase
                # Special case: preserve --PNL in uppercase
                if part.upper() == '--PNL':
                    converted_parts.append('--PNL')
                else:
                    converted_parts.append(part.lower())
            else:
                # Non-hyphen arguments (like symbol names, values) keep original case
                converted_parts.append(part)

        return converted_parts

    def _parse_plot_args(self, text: str) -> Optional[Dict]:
        """Parse plot arguments from Telegram message."""
        try:
            # Expected format: "plot -plot -symbol [TICKER] [-date YYYY-MM-DD]" where TICKER is any stock symbol
            # Note: Telegram may convert hyphens to em dashes (—) or en dashes (–)

            # Normalize different dash characters to regular hyphens
            normalized_text = text.replace('—', '-').replace('–', '-').replace('‐', '-')
            parts = normalized_text.strip().split()

            if len(parts) < 4:
                return None

            # Check for required format
            if parts[0].lower() != 'plot' or parts[1] != '-plot' or parts[2] != '-symbol':
                return None

            symbol = parts[3].upper()

            # Build alpaca arguments: ['--plot', '--symbol', 'SYMBOL']
            alpaca_args = ['--plot', '--symbol', symbol]

            # Check for optional date parameter
            date_value = None
            if len(parts) >= 6 and parts[4] == '-date':
                date_value = parts[5]
                # Validate date format (basic check for YYYY-MM-DD)
                if len(date_value) == 10 and date_value.count('-') == 2:
                    try:
                        # Verify it's a valid date
                        from datetime import datetime
                        datetime.strptime(date_value, '%Y-%m-%d')
                        alpaca_args.extend(['--date', date_value])
                    except ValueError:
                        self._log(f"Invalid date format: {date_value}", "ERROR")
                        return None
                else:
                    self._log(f"Invalid date format: {date_value}", "ERROR")
                    return None

            result = {
                'symbol': symbol,
                'alpaca_args': alpaca_args
            }

            if date_value:
                result['date'] = date_value

            return result

        except Exception as e:
            self._log(f"Error parsing plot args: {e}", "ERROR")
            return None

    def _parse_configure_alpaca_args(self, text: str) -> Optional[Dict]:
        """Parse configure alpaca arguments from Telegram message."""
        try:
            # Expected format: "configure alpaca --account-name [NAME] --account [ACCOUNT] [options]"
            # Normalize different dash characters to regular hyphens
            normalized_text = text.replace('—', '-').replace('–', '-').replace('‐', '-')

            # Use shlex to properly handle quoted strings
            import shlex
            try:
                parts = shlex.split(normalized_text)
            except ValueError:
                # Fall back to simple split if shlex fails
                parts = normalized_text.strip().split()

            if len(parts) < 3:
                return None

            # Check for required format start
            if parts[0].lower() != 'configure' or parts[1].lower() != 'alpaca':
                return None

            # Remove "configure alpaca" from the start
            raw_args = parts[2:]

            # Convert single hyphens to double hyphens for configure_alpaca.py compatibility
            # This follows the same pattern as _parse_alpaca_args for consistency
            converted_args = []
            for part in raw_args:
                if part.startswith('-') and not part.startswith('--'):
                    # Check if this is a negative number (starts with - followed by digits)
                    if len(part) > 1 and (part[1:].replace('.', '').isdigit()):
                        # This is a negative number, keep as single hyphen
                        converted_args.append(part)
                    else:
                        # Convert single hyphen to double hyphen
                        converted_args.append('--' + part[1:])
                elif part.startswith('--'):
                    # Already double hyphen, keep as is
                    converted_args.append(part)
                else:
                    # Non-hyphen arguments (values), keep original
                    converted_args.append(part)

            # Parse the converted arguments
            configure_args = []
            parsed_args = {}
            i = 0

            while i < len(converted_args):
                arg = converted_args[i]

                if arg == '--account-name' and i + 1 < len(converted_args):
                    account_name = converted_args[i + 1]
                    parsed_args['account_name'] = account_name
                    configure_args.extend([arg, account_name])
                    i += 2
                elif arg == '--account' and i + 1 < len(converted_args):
                    account = converted_args[i + 1]
                    if account.lower() in ['paper', 'live', 'cash']:
                        parsed_args['account'] = account.lower()
                        configure_args.extend([arg, account.lower()])
                    else:
                        self._log(f"Invalid account type: {account}", "ERROR")
                        return None
                    i += 2
                elif arg == '--auto-trade' and i + 1 < len(converted_args):
                    auto_trade = converted_args[i + 1]
                    if auto_trade.lower() in ['yes', 'no']:
                        parsed_args['auto_trade'] = auto_trade.lower()
                        configure_args.extend([arg, auto_trade.lower()])
                    else:
                        self._log(f"Invalid auto-trade value: {auto_trade}", "ERROR")
                        return None
                    i += 2
                elif arg == '--auto-amount' and i + 1 < len(converted_args):
                    try:
                        auto_amount = int(converted_args[i + 1])
                        parsed_args['auto_amount'] = auto_amount
                        configure_args.extend([arg, str(auto_amount)])
                        i += 2
                    except ValueError:
                        self._log(f"Invalid auto-amount value: {converted_args[i + 1]}", "ERROR")
                        return None
                elif arg == '--trailing-percent' and i + 1 < len(converted_args):
                    try:
                        trailing_percent = float(converted_args[i + 1])
                        parsed_args['trailing_percent'] = trailing_percent
                        configure_args.extend([arg, str(trailing_percent)])
                        i += 2
                    except ValueError:
                        self._log(f"Invalid trailing-percent value: {converted_args[i + 1]}", "ERROR")
                        return None
                elif arg == '--take-profit-percent' and i + 1 < len(converted_args):
                    try:
                        take_profit_percent = float(converted_args[i + 1])
                        parsed_args['take_profit_percent'] = take_profit_percent
                        configure_args.extend([arg, str(take_profit_percent)])
                        i += 2
                    except ValueError:
                        self._log(f"Invalid take-profit-percent value: {converted_args[i + 1]}", "ERROR")
                        return None
                elif arg == '--max-trades-per-day' and i + 1 < len(converted_args):
                    try:
                        max_trades = int(converted_args[i + 1])
                        parsed_args['max_trades_per_day'] = max_trades
                        configure_args.extend([arg, str(max_trades)])
                        i += 2
                    except ValueError:
                        self._log(f"Invalid max-trades-per-day value: {converted_args[i + 1]}", "ERROR")
                        return None
                elif arg == '--dry-run':
                    parsed_args['dry_run'] = True
                    configure_args.append(arg)
                    i += 1
                else:
                    self._log(f"Unknown argument: {arg}", "ERROR")
                    return None

            # Ensure at least one parameter besides account selection is provided
            param_keys = ['auto_trade', 'auto_amount', 'trailing_percent', 'take_profit_percent', 'max_trades_per_day']
            if not any(key in parsed_args for key in param_keys):
                self._log("No configuration parameters provided", "ERROR")
                return None

            parsed_args['configure_args'] = configure_args
            return parsed_args

        except Exception as e:
            self._log(f"Error parsing configure alpaca args: {e}", "ERROR")
            return None

    def _execute_configure_alpaca_command(self, args: List[str]) -> str:
        """Execute configure_alpaca.py command and return output."""
        try:
            import subprocess

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            configure_script = os.path.join(project_root, 'code', 'configure_alpaca.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command
            cmd = [python_path, configure_script] + args

            # Execute command
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                if output:
                    return f"✅ Configuration updated successfully:\n```\n{output}\n```"
                else:
                    return "✅ Configuration updated successfully"
            else:
                error_msg = error_output if error_output else output
                return f"❌ Configuration update failed (exit code: {result.returncode}):\n```\n{error_msg}\n```"

        except subprocess.TimeoutExpired:
            return "❌ Configuration command timed out after 30 seconds"
        except Exception as e:
            return f"❌ Error executing configuration command: {str(e)}"

    def _extract_plot_path(self, alpaca_output: str) -> Optional[str]:
        """Extract plot file path from alpaca.py output."""
        try:
            # Look for pattern: "Chart generated successfully: plots/20250818/SYMBOL_chart.png"
            pattern = r'Chart generated successfully:\s*([^\s]+\.png)'
            match = re.search(pattern, alpaca_output, re.IGNORECASE)

            if match:
                plot_path = match.group(1).strip()
                self._log(f"📊 Extracted plot path: {plot_path}")

                # Verify file exists
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                full_path = os.path.join(project_root, plot_path)

                if os.path.isfile(full_path):
                    return full_path
                else:
                    self._log(f"❌ Plot file not found: {full_path}", "ERROR")
                    return None
            else:
                self._log("❌ No chart generation message found in output", "ERROR")
                return None

        except Exception as e:
            self._log(f"Error extracting plot path: {e}", "ERROR")
            return None

    def _execute_alpaca_command(self, args: List[str]) -> str:
        """Execute Alpaca command and return output."""
        try:
            import subprocess

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alpaca_script = os.path.join(project_root, 'code', 'alpaca.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command
            cmd = [python_path, alpaca_script] + args

            # Execute command
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                if output:
                    return f"✅ Command executed successfully:\n```\n{output}\n```"
                else:
                    return "✅ Command executed successfully (no output)"
            else:
                error_msg = error_output if error_output else output
                return f"❌ Command failed (exit code: {result.returncode}):\n```\n{error_msg}\n```"

        except subprocess.TimeoutExpired:
            return "❌ Command timed out after 60 seconds"
        except Exception as e:
            return f"❌ Error executing command: {str(e)}"

    def _execute_bam_command(self) -> str:
        """Execute BAM (Bulk Account Manager) command and return output."""
        try:
            import subprocess

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bam_script = os.path.join(project_root, 'code', 'bam.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command (no additional arguments needed for bam.py)
            cmd = [python_path, bam_script]

            # Execute command
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120  # Longer timeout for BAM operations
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                if output:
                    return f"✅ BAM executed successfully:\n```\n{output}\n```"
                else:
                    return "✅ BAM executed successfully (no output)"
            else:
                error_msg = error_output if error_output else output
                return f"❌ BAM failed (exit code: {result.returncode}):\n```\n{error_msg}\n```"

        except subprocess.TimeoutExpired:
            return "❌ BAM command timed out after 120 seconds"
        except Exception as e:
            return f"❌ Error executing BAM command: {str(e)}"

    def _execute_oracle_signal_command(self) -> str:
        """Execute oracle_signal.py command and return output."""
        try:
            import subprocess

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            oracle_script = os.path.join(project_root, 'code', 'oracle_signal.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command (no additional arguments needed for oracle_signal.py)
            cmd = [python_path, oracle_script]

            # Execute command
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60  # Standard timeout for oracle signal operations
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                if output:
                    return f"📊 Oracle Signal Results:\n```\n{output}\n```"
                else:
                    return "📊 Oracle Signal executed successfully (no results found)"
            else:
                error_msg = error_output if error_output else output
                return f"❌ Oracle Signal failed (exit code: {result.returncode}):\n```\n{error_msg}\n```"

        except subprocess.TimeoutExpired:
            return "❌ Oracle Signal command timed out after 60 seconds"
        except Exception as e:
            return f"❌ Error executing Oracle Signal command: {str(e)}"

    def _execute_volume_surge_command(self) -> str:
        """Execute alpaca_screener.py command and return output with CSV file contents."""
        try:
            import subprocess
            import glob

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            screener_script = os.path.join(project_root, 'code', 'alpaca_screener.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command with the specified arguments
            cmd = [
                python_path, screener_script,
                '--exchanges', 'NASDAQ', 'AMEX',
                '--max-symbols', '7000',
                '--min-price', '0.75',
                '--max-price', '40.00',
                '--min-volume', '250000',
                '--min-percent-change', '5.0',
                '--surge-days', '50',
                '--volume-surge', '5.0',
                '--export-csv', 'relative_volume_nasdaq_amex.csv',
                '--verbose'
            ]

            # Execute command with 10 minute timeout
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                # Extract filename from last line of output
                lines = output.split('\n')
                csv_file_path = None

                # Look for a line containing the CSV file path
                for line in reversed(lines):
                    if ('relative_volume_' in line or 'relativevolume' in line) and '.csv' in line:
                        # Extract the file path from the line
                        # Handle format like "Results exported to ./historical_data/..."
                        if 'Results exported to' in line:
                            csv_file_path = line.split('Results exported to')[1].strip()
                        else:
                            csv_file_path = line.strip()
                        break

                if csv_file_path:
                    # Try to read the CSV file
                    full_csv_path = os.path.join(project_root, csv_file_path)
                    try:
                        with open(full_csv_path, 'r') as f:
                            csv_content = f.read().strip()

                        # Format CSV content to 3 decimal places
                        formatted_csv = self._format_csv_decimals(csv_content, decimal_places=3)

                        # Return the CSV content
                        return (f"📈 Volume Surge Scanner Results:\n\n"
                                f"📄 **File:** `{csv_file_path}`\n\n"
                                f"```csv\n{formatted_csv}\n```")

                    except FileNotFoundError:
                        return (f"📈 Scanner completed successfully but CSV file not found "
                                f"at: {csv_file_path}\n\n```\n{output}\n```")
                    except Exception as e:
                        return (f"📈 Scanner completed but error reading CSV file: "
                                f"{str(e)}\n\n```\n{output}\n```")
                else:
                    return f"📈 Volume Surge Scanner Results:\n```\n{output}\n```"
            else:
                error_msg = error_output if error_output else output
                return (f"❌ Volume Surge Scanner failed (exit code: "
                        f"{result.returncode}):\n```\n{error_msg}\n```")

        except subprocess.TimeoutExpired:
            return "❌ Volume Surge Scanner timed out after 10 minutes"
        except Exception as e:
            return f"❌ Error executing Volume Surge Scanner: {str(e)}"

    def _execute_top_gainers_command(self) -> str:
        """Execute alpaca_screener.py command for top gainers and return output with CSV file contents."""
        try:
            import subprocess
            import glob

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            screener_script = os.path.join(project_root, 'code', 'alpaca_screener.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command with the specified arguments for top gainers
            cmd = [
                python_path, screener_script,
                '--exchanges', 'NASDAQ', 'AMEX',
                '--max-symbols', '7000',
                '--min-price', '0.75',
                '--max-price', '40.00',
                '--min-volume', '250000',
                '--top-gainers', '40',
                '--export-csv', 'top_gainers_nasdaq_amex.csv',
                '--verbose'
            ]

            # Execute command with 10 minute timeout
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                # Extract filename from last line of output
                lines = output.split('\n')
                csv_file_path = None

                # Look for a line containing the CSV file path
                for line in reversed(lines):
                    if ('top_gainers_' in line or 'topgainers' in line) and '.csv' in line:
                        # Extract the file path from the line
                        # Handle format like "Results exported to ./historical_data/..."
                        if 'Results exported to' in line:
                            csv_file_path = line.split('Results exported to')[1].strip()
                        else:
                            csv_file_path = line.strip()
                        break

                if csv_file_path:
                    # Try to read the CSV file
                    full_csv_path = os.path.join(project_root, csv_file_path)
                    try:
                        with open(full_csv_path, 'r') as f:
                            csv_content = f.read().strip()

                        # Format CSV content to 3 decimal places
                        formatted_csv = self._format_csv_decimals(csv_content, decimal_places=3)

                        # Return the CSV content
                        return (f"🚀 Top Gainers Scanner Results:\n\n"
                                f"📄 **File:** `{csv_file_path}`\n\n"
                                f"```csv\n{formatted_csv}\n```")

                    except FileNotFoundError:
                        return (f"🚀 Scanner completed successfully but CSV file not found "
                                f"at: {csv_file_path}\n\n```\n{output}\n```")
                    except Exception as e:
                        return (f"🚀 Scanner completed but error reading CSV file: "
                                f"{str(e)}\n\n```\n{output}\n```")
                else:
                    return f"🚀 Top Gainers Scanner Results:\n```\n{output}\n```"
            else:
                error_msg = error_output if error_output else output
                return (f"❌ Top Gainers Scanner failed (exit code: "
                        f"{result.returncode}):\n```\n{error_msg}\n```")

        except subprocess.TimeoutExpired:
            return "❌ Top Gainers Scanner timed out after 10 minutes"
        except Exception as e:
            return f"❌ Error executing Top Gainers Scanner: {str(e)}"

    def _handle_premarket_top_gainers_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 'premarket top gainers' command to run premarket_top_gainers.py."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"🌅 Premarket top gainers command from {display_name}: {text}")

            # Send processing message
            message = "🌅 Running premarket top gainers scanner... This may take up to 10 minutes."
            self._send_response(chat_id, message)

            # Execute premarket_top_gainers.py script
            result = self._execute_premarket_top_gainers_command()

            # Send result back to user
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling premarket top gainers command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error executing premarket top gainers command: {str(e)}")

    def _handle_market_open_top_gainers_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 'market open top gainers' command to run market_open_top_gainers.py."""
        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"📈 Market open top gainers command from {display_name}: {text}")

            # Send processing message
            message = "📈 Running market open top gainers scanner... This may take up to 15 minutes."
            self._send_response(chat_id, message)

            # Execute market_open_top_gainers.py script
            result = self._execute_market_open_top_gainers_command()

            # Send result back to user
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling market open top gainers command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error executing market open top gainers command: {str(e)}")

    def _handle_configure_alpaca_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
        """Handle 'configure alpaca' command for configuration management (Bruce only)."""

        # Check user authorization - only Bruce can use this command
        if not self._is_authorized_user(username, first_name):
            return  # Silently ignore unauthorized users

        try:
            display_name = f"{first_name} {last_name}".strip() or username or f"User_{chat_id[-4:]}"
            self._log(f"🔧 Configure Alpaca command from {display_name}: {text}")

            # Parse configure alpaca arguments from message
            args = self._parse_configure_alpaca_args(text)
            if not args:
                error_msg = ("❌ Invalid configure alpaca command. Usage:\n"
                             "configure alpaca --account-name [NAME] --account [paper/live/cash] [options]\n\n"
                             "Options:\n"
                             "  --auto-trade [yes/no]\n"
                             "  --auto-amount [NUMBER]\n"
                             "  --trailing-percent [NUMBER]\n"
                             "  --take-profit-percent [NUMBER]\n"
                             "  --max-trades-per-day [NUMBER]\n"
                             "  --dry-run (preview changes)\n\n"
                             "Examples:\n"
                             "  configure alpaca --account-name Bruce --account paper --auto-amount 5000\n"
                             "  configure alpaca --account-name \"Dale Wilson\" --account live --auto-trade yes --dry-run")
                self._send_response(chat_id, error_msg)
                return

            # Send processing message
            account_name = args.get('account_name', 'Bruce')
            account = args.get('account', 'paper')
            dry_run = args.get('dry_run', False)
            action = "Preview" if dry_run else "Updating"

            self._send_response(chat_id, f"🔧 {action} configuration for {account_name}/{account}...")

            # Execute configure_alpaca.py command
            result = self._execute_configure_alpaca_command(args['configure_args'])

            # Send result back to Bruce
            self._send_response(chat_id, result)

        except Exception as e:
            self._log(f"Error handling configure alpaca command: {e}", "ERROR")
            self._send_response(chat_id, f"❌ Error executing configure alpaca command: {str(e)}")

    def _execute_premarket_top_gainers_command(self) -> str:
        """Execute premarket_top_gainers.py command and return output with CSV file contents."""
        try:
            import subprocess
            import glob

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            premarket_script = os.path.join(project_root, 'code', 'premarket_top_gainers.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command with the specified arguments for premarket top gainers
            cmd = [
                python_path, premarket_script,
                '--exchanges', 'NASDAQ', 'AMEX',
                '--max-symbols', '7000',
                '--min-price', '0.75',
                '--max-price', '40.00',
                '--min-volume', '250000',
                '--top-gainers', '40',
                '--export-csv', 'top_gainers_nasdaq_amex.csv',
                '--verbose'
            ]

            # Execute command with 10 minute timeout
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                # Extract filename from last line of output
                lines = output.split('\n')
                csv_file_path = None

                # Look for a line containing the CSV file path
                for line in reversed(lines):
                    if ('top_gainers_' in line or 'topgainers' in line) and '.csv' in line:
                        # Extract the file path from the line
                        # Handle format like \"Results exported to ./historical_data/...\"
                        if 'Results exported to' in line:
                            csv_file_path = line.split('Results exported to')[1].strip()
                        else:
                            csv_file_path = line.strip()
                        break

                if csv_file_path:
                    # Try to read the CSV file
                    full_csv_path = os.path.join(project_root, csv_file_path)
                    try:
                        with open(full_csv_path, 'r') as f:
                            csv_content = f.read().strip()

                        # Format CSV content to 3 decimal places
                        formatted_csv = self._format_csv_decimals(csv_content, decimal_places=3)

                        # Return the CSV content
                        return (f"🌅 Premarket Top Gainers Scanner Results:\n\n"
                                f"📄 **File:** `{csv_file_path}`\n\n"
                                f"```csv\n{formatted_csv}\n```")

                    except FileNotFoundError:
                        return (f"🌅 Scanner completed successfully but CSV file not found "
                                f"at: {csv_file_path}\n\n```\n{output}\n```")
                    except Exception as e:
                        return (f"🌅 Scanner completed but error reading CSV file: "
                                f"{str(e)}\n\n```\n{output}\n```")
                else:
                    return f"🌅 Premarket Top Gainers Scanner Results:\n```\n{output}\n```"
            else:
                error_msg = error_output if error_output else output
                return (f"❌ Premarket Top Gainers Scanner failed (exit code: "
                        f"{result.returncode}):\n```\n{error_msg}\n```")

        except subprocess.TimeoutExpired:
            return "❌ Premarket Top Gainers Scanner timed out after 10 minutes"
        except Exception as e:
            return f"❌ Error executing Premarket Top Gainers Scanner: {str(e)}"

    def _execute_market_open_top_gainers_command(self) -> str:
        """Execute market_open_top_gainers.py command and return output with CSV file contents."""
        try:
            import subprocess
            import glob

            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            market_open_script = os.path.join(project_root, 'code', 'market_open_top_gainers.py')

            # Use conda environment python to execute the script
            python_path = os.path.expanduser('~/miniconda3/envs/alpaca/bin/python')

            # Build command with the specified arguments for market open top gainers
            cmd = [
                python_path, market_open_script,
                '--exchanges', 'NASDAQ', 'AMEX',
                '--max-symbols', '7000',
                '--min-price', '0.75',
                '--max-price', '40.00',
                '--min-volume', '250000',
                '--top-gainers', '40',
                '--export-csv', 'gainers_nasdaq_amex.csv',
                '--verbose'
            ]

            # Execute command with 15 minute timeout
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout
            )

            # Get output
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if result.returncode == 0:
                # Extract filename from last line of output
                lines = output.split('\n')
                csv_file_path = None

                # Look for a line containing the CSV file path
                for line in reversed(lines):
                    if 'gainers_' in line and '.csv' in line:
                        # Extract the file path from the line
                        # Handle format like "Results exported to ./historical_data/..."
                        if 'Results exported to' in line:
                            csv_file_path = line.split('Results exported to')[1].strip()
                        else:
                            csv_file_path = line.strip()
                        break

                if csv_file_path:
                    # Try to read the CSV file
                    full_csv_path = os.path.join(project_root, csv_file_path)
                    try:
                        with open(full_csv_path, 'r') as f:
                            csv_content = f.read().strip()

                        # Format CSV content to 3 decimal places
                        formatted_csv = self._format_csv_decimals(csv_content, decimal_places=3)

                        # Return the CSV content
                        return (f"📈 Market Open Top Gainers Scanner Results:\n\n"
                                f"📄 **File:** `{csv_file_path}`\n\n"
                                f"```csv\n{formatted_csv}\n```")

                    except FileNotFoundError:
                        return (f"📈 Scanner completed successfully but CSV file not found "
                                f"at: {csv_file_path}\n\n```\n{output}\n```")
                    except Exception as e:
                        return (f"📈 Scanner completed but error reading CSV file: "
                                f"{str(e)}\n\n```\n{output}\n```")
                else:
                    return f"📈 Market Open Top Gainers Scanner Results:\n```\n{output}\n```"
            else:
                error_msg = error_output if error_output else output
                return (f"❌ Market Open Top Gainers Scanner failed (exit code: "
                        f"{result.returncode}):\n```\n{error_msg}\n```")

        except subprocess.TimeoutExpired:
            return "❌ Market Open Top Gainers Scanner timed out after 15 minutes"
        except Exception as e:
            return f"❌ Error executing Market Open Top Gainers Scanner: {str(e)}"

    def _format_csv_decimals(self, csv_content: str, decimal_places: int = 3) -> str:
        """Format CSV content to show specified number of decimal places for numeric values."""
        try:
            import csv
            import io
            
            # Parse CSV content
            csv_reader = csv.reader(io.StringIO(csv_content))
            rows = list(csv_reader)
            
            if not rows:
                return csv_content
            
            # Process each row
            formatted_rows = []
            for i, row in enumerate(rows):
                if i == 0:  # Header row
                    formatted_rows.append(row)
                else:
                    formatted_row = []
                    for cell in row:
                        # Try to format as float with specified decimal places
                        try:
                            # Check if it's a percentage
                            if '%' in cell:
                                # It's a percentage - extract number and reformat
                                num_part = cell.replace('%', '').strip()
                                if num_part.replace('.', '').replace('-', '').replace('+', '').isdigit():
                                    num_value = float(num_part)
                                    formatted_cell = f"{num_value:.{decimal_places}f}%"
                                else:
                                    formatted_cell = cell
                            else:
                                # Try parsing as number
                                num_value = float(cell)
                                # Check if it's a whole number (integer)
                                if num_value == int(num_value):
                                    formatted_cell = str(int(num_value))
                                else:
                                    formatted_cell = f"{num_value:.{decimal_places}f}"
                        except (ValueError, TypeError):
                            # Not a number, keep original
                            formatted_cell = cell
                        
                        formatted_row.append(formatted_cell)
                    formatted_rows.append(formatted_row)
            
            # Convert back to CSV string
            output = io.StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerows(formatted_rows)
            return output.getvalue().strip()
            
        except Exception as e:
            # If formatting fails, return original content
            return csv_content

    def _send_response(self, chat_id: str, message: str):
        """Send a response message to a specific chat."""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    return True
                else:
                    self._log(f"Failed to send response: {result.get('description')}", "ERROR")
            else:
                self._log(f"HTTP error sending response: {response.status_code}", "ERROR")

        except Exception as e:
            self._log(f"Error sending response: {e}", "ERROR")

        return False


def main():
    """Main entry point."""
    print("🚀 TELEGRAM POLLING SERVICE")
    print("=" * 50)

    try:
        service = TelegramPollingService()
        service.start_polling()
    except KeyboardInterrupt:
        print("\n👋 Service stopped by user")
    except Exception as e:
        print(f"❌ Service error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
