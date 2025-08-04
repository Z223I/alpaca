import requests
import time
from typing import Dict, List
from .config import TelegramConfig
from .user_manager import UserManager

class TelegramPoster:
    """Handles sending messages to Telegram users."""

    def __init__(self):
        self.config = TelegramConfig()
        self.user_manager = UserManager()
        self.base_url = f"https://api.telegram.org/bot{self.config.BOT_TOKEN}"

    def send_message(self, message: str, urgent: bool = False) -> Dict:
        """
        Send message to all enabled users in CSV file.

        Args:
            message (str): Message content to send
            urgent (bool): If True, ignore user notification preferences

        Returns:
            dict: {
                'success': bool,
                'sent_count': int,
                'failed_count': int,
                'errors': list
            }
        """
        # Validate configuration
        try:
            self.config.validate_config()
        except ValueError as e:
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 0,
                'errors': [f"Configuration error: {str(e)}"]
            }

        # Check if Telegram is enabled
        if not self.config.ENABLED:
            return {
                'success': True,
                'sent_count': 0,
                'failed_count': 0,
                'errors': ['Telegram notifications disabled']
            }

        # Get active users
        users = self.user_manager.get_active_users()
        if not users:
            return {
                'success': True,
                'sent_count': 0,
                'failed_count': 0,
                'errors': ['No active users found in CSV file']
            }

        # Send messages to all users
        sent_count = 0
        failed_count = 0
        errors = []

        for user in users:
            chat_id = user.get('chat_id')
            if not chat_id:
                failed_count += 1
                errors.append(f"Invalid chat_id for user: {user.get('username', 'unknown')}")
                continue

            success = self._send_to_chat(chat_id, message, urgent)
            if success:
                sent_count += 1
            else:
                failed_count += 1
                errors.append(f"Failed to send to {user.get('username', chat_id)}")

            # Rate limiting - small delay between messages
            time.sleep(0.1)

        return {
            'success': sent_count > 0,
            'sent_count': sent_count,
            'failed_count': failed_count,
            'errors': errors
        }

    def send_message_to_user(self, message: str, username: str, urgent: bool = False) -> Dict:
        """
        Send message to a specific user by username.

        Args:
            message (str): Message content to send
            username (str): Username to send to
            urgent (bool): If True, ignore user notification preferences

        Returns:
            dict: {
                'success': bool,
                'sent_count': int,
                'failed_count': int,
                'errors': list,
                'target_user': str
            }
        """
        # Validate configuration
        try:
            self.config.validate_config()
        except ValueError as e:
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 0,
                'errors': [f"Configuration error: {str(e)}"],
                'target_user': username
            }

        # Check if Telegram is enabled
        if not self.config.ENABLED:
            return {
                'success': True,
                'sent_count': 0,
                'failed_count': 0,
                'errors': ['Telegram notifications disabled'],
                'target_user': username
            }

        # Find the specific user
        user = self.user_manager.get_user_by_username(username)
        if not user:
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 1,
                'errors': [f'User "{username}" not found or not enabled'],
                'target_user': username
            }

        # Send message to the specific user
        chat_id = user.get('chat_id')
        if not chat_id:
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 1,
                'errors': [f'Invalid chat_id for user: {username}'],
                'target_user': username
            }

        success = self._send_to_chat(chat_id, message, urgent)
        if success:
            return {
                'success': True,
                'sent_count': 1,
                'failed_count': 0,
                'errors': [],
                'target_user': username
            }
        else:
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 1,
                'errors': [f'Failed to send to {username}'],
                'target_user': username
            }

    def _send_to_chat(self, chat_id: str, message: str, urgent: bool = False) -> bool:
        """Send message to a specific chat with retry logic."""
        url = f"{self.base_url}/sendMessage"

        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': self.config.PARSE_MODE,
            'disable_notification': self.config.DISABLE_NOTIFICATION and not urgent
        }

        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, 
                    json=payload, 
                    timeout=self.config.TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get('ok'):
                        return True
                    else:
                        print(f"Telegram API error: {result.get('description', 'Unknown error')}")
                        return False

                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get('Retry-After', 1))
                    time.sleep(retry_after)
                    continue

                else:
                    print(f"HTTP error {response.status_code}: {response.text}")
                    return False

            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

        return False

    def send_alert(self, message: str, level: str = 'info', chat_id: str = None) -> bool:
        """Send custom alert with formatting based on level."""
        # Format message based on level
        level_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }

        icon = level_icons.get(level.lower(), 'ℹ️')
        formatted_message = f"{icon} {message}"

        if chat_id:
            # Send to specific chat
            return self._send_to_chat(chat_id, formatted_message, urgent=(level == 'error'))
        else:
            # Send to all users
            result = self.send_message(formatted_message, urgent=(level == 'error'))
            return result['success']

# Convenience functions for easy import
def send_message(message: str, urgent: bool = False) -> Dict:
    """Send message to all enabled users."""
    poster = TelegramPoster()
    return poster.send_message(message, urgent)

def send_alert(message: str, level: str = 'info', chat_id: str = None) -> bool:
    """Send formatted alert message."""
    poster = TelegramPoster()
    return poster.send_alert(message, level, chat_id)

def add_user(chat_id: str, username: str = "", enabled: bool = True, notes: str = "") -> bool:
    """Add new user to CSV file."""
    user_manager = UserManager()
    return user_manager.add_user(chat_id, username, enabled, notes)

def get_active_users() -> List[Dict[str, str]]:
    """Return list of enabled users."""
    user_manager = UserManager()
    return user_manager.get_active_users()