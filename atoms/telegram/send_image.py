import requests
import time
import os
from typing import Dict, List
from .config import TelegramConfig
from .user_manager import UserManager


class TelegramImageSender:
    """Handles sending images to Telegram users."""

    def __init__(self):
        self.config = TelegramConfig()
        self.user_manager = UserManager()
        self.base_url = f"https://api.telegram.org/bot{self.config.BOT_TOKEN}"

    def send_image(self, image_path: str, caption: str = "",
                   urgent: bool = False) -> Dict:
        """
        Send image to all enabled users in CSV file.

        Args:
            image_path (str): Path to image file to send
            caption (str): Optional caption text for the image
            urgent (bool): If True, ignore user notification preferences

        Returns:
            dict: {
                'success': bool,
                'sent_count': int,
                'failed_count': int,
                'errors': list
            }
        """
        # Validate image file exists
        if not os.path.isfile(image_path):
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 0,
                'errors': [f"Image file not found: {image_path}"]
            }

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

        # Send images to all users
        sent_count = 0
        failed_count = 0
        errors = []

        for user in users:
            chat_id = user.get('chat_id')
            if not chat_id:
                failed_count += 1
                username = user.get('username', 'unknown')
                errors.append(f"Invalid chat_id for user: {username}")
                continue

            success = self._send_image_to_chat(
                chat_id, image_path, caption, urgent
            )
            if success:
                sent_count += 1
            else:
                failed_count += 1
                target = user.get('username', chat_id)
                errors.append(f"Failed to send to {target}")

            # Rate limiting - small delay between messages
            time.sleep(0.1)

        return {
            'success': sent_count > 0,
            'sent_count': sent_count,
            'failed_count': failed_count,
            'errors': errors
        }

    def send_image_to_user(self, image_path: str, username: str,
                           caption: str = "", urgent: bool = False) -> Dict:
        """
        Send image to a specific user by username.

        Args:
            image_path (str): Path to image file to send
            username (str): Username to send to
            caption (str): Optional caption text for the image
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
        # Validate image file exists
        if not os.path.isfile(image_path):
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 1,
                'errors': [f"Image file not found: {image_path}"],
                'target_user': username
            }

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

        # Send image to the specific user
        chat_id = user.get('chat_id')
        if not chat_id:
            return {
                'success': False,
                'sent_count': 0,
                'failed_count': 1,
                'errors': [f'Invalid chat_id for user: {username}'],
                'target_user': username
            }

        success = self._send_image_to_chat(
            chat_id, image_path, caption, urgent
        )
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

    def _send_image_to_chat(self, chat_id: str, image_path: str,
                            caption: str = "", urgent: bool = False) -> bool:
        """Send image to a specific chat with retry logic."""
        url = f"{self.base_url}/sendPhoto"

        # Prepare form data
        data = {
            'chat_id': chat_id,
            'disable_notification': (
                self.config.DISABLE_NOTIFICATION and not urgent
            )
        }

        # Add caption if provided
        if caption:
            data['caption'] = caption
            data['parse_mode'] = self.config.PARSE_MODE

        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(image_path, 'rb') as image_file:
                    files = {'photo': image_file}
                    response = requests.post(
                        url,
                        data=data,
                        files=files,
                        timeout=self.config.TIMEOUT
                    )

                if response.status_code == 200:
                    result = response.json()
                    if result.get('ok'):
                        return True
                    else:
                        desc = result.get('description', 'Unknown error')
                        print(f"Telegram API error: {desc}")
                        return False

                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(
                        response.headers.get('Retry-After', 1)
                    )
                    time.sleep(retry_after)
                    continue

                else:
                    print(
                        f"HTTP error {response.status_code}: {response.text}"
                    )
                    return False

            except FileNotFoundError:
                print(f"Image file not found: {image_path}")
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

            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

        return False

    def send_image_alert(self, image_path: str, caption: str = "",
                         level: str = 'info', chat_id: str = None) -> bool:
        """Send custom image alert with formatting based on level."""
        # Format caption based on level
        level_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }

        icon = level_icons.get(level.lower(), 'ℹ️')
        formatted_caption = f"{icon} {caption}" if caption else icon

        if chat_id:
            # Send to specific chat
            return self._send_image_to_chat(
                chat_id, image_path, formatted_caption,
                urgent=(level == 'error')
            )
        else:
            # Send to all users
            result = self.send_image(
                image_path, formatted_caption, urgent=(level == 'error')
            )
            return result['success']


# Convenience functions for easy import (excluding add_user functionality)
def send_image(image_path: str, caption: str = "",
               urgent: bool = False) -> Dict:
    """Send image to all enabled users."""
    sender = TelegramImageSender()
    return sender.send_image(image_path, caption, urgent)


def send_image_alert(image_path: str, caption: str = "",
                     level: str = 'info', chat_id: str = None) -> bool:
    """Send formatted image alert."""
    sender = TelegramImageSender()
    return sender.send_image_alert(image_path, caption, level, chat_id)


def get_active_users() -> List[Dict[str, str]]:
    """Return list of enabled users."""
    user_manager = UserManager()
    return user_manager.get_active_users()
