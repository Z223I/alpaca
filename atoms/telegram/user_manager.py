import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from .config import TelegramConfig

class UserManager:
    """Manages Telegram users from CSV file."""

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or TelegramConfig.CSV_FILE_PATH
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create CSV file with default structure if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            self._create_default_csv()

    def _create_default_csv(self):
        """Create default CSV file with instructions."""
        default_content = [
            ['chat_id', 'username', 'enabled', 'created_date', 'notes'],
            ['# Add your Telegram chat IDs here', '', '', '', ''],
            ['# To get your chat ID, message @userinfobot on Telegram', '', '', '', ''],
            ['# Group chat IDs are negative numbers (start with -)', '', '', '', ''],
            ['# Example: -1001234567890,my_group,true,2024-01-15,Trading alerts', '', '', '', '']
        ]

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(default_content)

    def get_active_users(self) -> List[Dict[str, str]]:
        """Return list of enabled users from CSV."""
        users = []

        if not os.path.exists(self.csv_path):
            print(f"CSV file does not exist: {self.csv_path}")
            return users

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                row_count = 0
                for row in reader:
                    row_count += 1
                    chat_id = row.get('chat_id', '').strip()

                    # Skip comment lines, empty rows, and header duplicates
                    if (not chat_id or 
                        chat_id.startswith('#') or 
                        chat_id == 'chat_id'):
                        continue

                    # Only include enabled users
                    enabled = row.get('enabled', '').strip().lower()
                    if enabled == 'true':
                        user = {
                            'chat_id': chat_id,
                            'username': row.get('username', '').strip(),
                            'notes': row.get('notes', '').strip()
                        }
                        users.append(user)

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            import traceback
            traceback.print_exc()

        return users

    def get_user_by_username(self, username: str) -> Optional[Dict[str, str]]:
        """
        Find a user by username (case-insensitive).
        
        Args:
            username: Username to search for
            
        Returns:
            User dict if found and enabled, None otherwise
        """
        if not username:
            return None
            
        users = self.get_active_users()
        for user in users:
            user_name = user.get('username', '').strip()
            if user_name.lower() == username.lower():
                return user
        return None

    def add_user(self, chat_id: str, username: str = "", enabled: bool = True, notes: str = "") -> bool:
        """Add new user to CSV file."""
        try:
            # Check if user already exists
            existing_users = self._read_all_users()
            for user in existing_users:
                if user.get('chat_id') == chat_id:
                    print(f"User {chat_id} already exists - enabling instead")
                    # User exists but might be disabled - enable them
                    return self.enable_user(chat_id)

            # Add new user
            new_user = {
                'chat_id': chat_id,
                'username': username,
                'enabled': str(enabled).lower(),
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'notes': notes
            }

            # Append to file
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['chat_id', 'username', 'enabled', 'created_date', 'notes'])
                writer.writerow(new_user)

            return True

        except Exception as e:
            print(f"Error adding user: {e}")
            import traceback
            traceback.print_exc()
            return False

    def disable_user(self, chat_id: str) -> bool:
        """Disable user without removing from CSV."""
        return self._update_user_status(chat_id, enabled=False)

    def enable_user(self, chat_id: str) -> bool:
        """Enable user in CSV."""
        return self._update_user_status(chat_id, enabled=True)

    def _update_user_status(self, chat_id: str, enabled: bool) -> bool:
        """Update user enabled status."""
        try:
            users = self._read_all_users()
            updated = False

            for user in users:
                if user.get('chat_id') == chat_id:
                    user['enabled'] = str(enabled).lower()
                    updated = True
                    break

            if updated:
                self._write_all_users(users)
                return True
            else:
                print(f"User {chat_id} not found")
                return False

        except Exception as e:
            print(f"Error updating user status: {e}")
            return False

    def _read_all_users(self) -> List[Dict[str, str]]:
        """Read all users from CSV, including disabled ones."""
        users = []

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    users.append(dict(row))
        except Exception as e:
            print(f"Error reading all users: {e}")

        return users

    def _write_all_users(self, users: List[Dict[str, str]]):
        """Write all users back to CSV."""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
            if users:
                fieldnames = ['chat_id', 'username', 'enabled', 'created_date', 'notes']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(users)