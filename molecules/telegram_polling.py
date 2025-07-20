#!/usr/bin/env python3
"""
Telegram Polling Service for Automated User Management

This service continuously polls the Telegram API for new messages and automatically
manages user subscriptions based on bot commands like /start, /stop, etc.
"""

import sys
import os
import time
import json
import signal
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.telegram.telegram_post import send_message, send_alert
from atoms.telegram.user_manager import UserManager
from atoms.telegram.config import TelegramConfig
import requests

class TelegramPollingService:
    """Service for polling Telegram messages and managing user subscriptions."""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.config = TelegramConfig()
        self.running = False
        self.last_update_id = 0
        self.poll_interval = 5  # seconds
        self.base_url = f"https://api.telegram.org/bot{self.config.BOT_TOKEN}"
        
        # Command handlers
        self.command_handlers = {
            '/start': self._handle_start,
            '/stop': self._handle_stop,
            '/subscribe': self._handle_subscribe,
            '/unsubscribe': self._handle_unsubscribe,
            '/status': self._handle_status,
            '/help': self._handle_help
        }
        
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
            
            # Process commands
            if text.startswith('/'):
                command = text.split()[0].lower()
                if command in self.command_handlers:
                    self.command_handlers[command](chat_id, username, first_name, last_name, text)
                else:
                    self._handle_unknown_command(chat_id, text)
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
                response = f"❌ Sorry, there was an error subscribing you to alerts. Please try again."
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
        help_text = """🤖 Trading Alerts Bot Help

Available Commands:
/start - Subscribe to trading alerts
/stop - Unsubscribe from alerts  
/subscribe - Same as /start
/unsubscribe - Same as /stop
/status - Check your subscription status
/help - Show this help message

You'll receive:
🚀 ORB breakout signals
📊 Trade notifications  
📈 Portfolio updates
⚠️ System alerts

Questions? Contact the bot administrator."""

        self._send_response(chat_id, help_text)
    
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
    
    def _send_response(self, chat_id: str, message: str):
        """Send a response message to a specific chat."""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
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