#!/usr/bin/env python3
"""
Send update announcement to active Telegram users using plain text mode.
"""

import sys
import os
from pathlib import Path
import requests
import time

# Add project root to Python path
sys.path.insert(0, '.')

try:
    from atoms.telegram.config import TelegramConfig
    from atoms.telegram.user_manager import UserManager
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("❌ Telegram integration not available - missing dependencies")


def send_plain_text_message(message: str) -> dict:
    """Send message using plain text mode instead of HTML."""
    
    if not TELEGRAM_AVAILABLE:
        return {
            'success': False,
            'sent_count': 0,
            'failed_count': 0,
            'errors': ['Telegram integration not available']
        }
    
    # Initialize config and user manager
    config = TelegramConfig()
    user_manager = UserManager()
    
    try:
        config.validate_config()
    except ValueError as e:
        return {
            'success': False,
            'sent_count': 0,
            'failed_count': 0,
            'errors': [f'Configuration error: {str(e)}']
        }
    
    if not config.ENABLED:
        return {
            'success': True,
            'sent_count': 0,
            'failed_count': 0,
            'errors': ['Telegram notifications disabled']
        }
    
    # Get active users
    users = user_manager.get_active_users()
    if not users:
        return {
            'success': True,
            'sent_count': 0,
            'failed_count': 0,
            'errors': ['No active users found']
        }
    
    # Send messages using plain text mode
    base_url = f"https://api.telegram.org/bot{config.BOT_TOKEN}"
    sent_count = 0
    failed_count = 0
    errors = []
    
    for user in users:
        chat_id = user.get('chat_id')
        if not chat_id:
            failed_count += 1
            errors.append(f"Invalid chat_id for user: {user.get('username', 'unknown')}")
            continue
        
        # Send with plain text mode (no HTML parsing)
        url = f"{base_url}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'disable_notification': False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    sent_count += 1
                    print(f"✅ Sent to {user.get('username', chat_id)}")
                else:
                    failed_count += 1
                    error_msg = result.get('description', 'Unknown error')
                    errors.append(f"Failed to send to {user.get('username', chat_id)}: {error_msg}")
                    print(f"❌ API error for {user.get('username', chat_id)}: {error_msg}")
            else:
                failed_count += 1
                errors.append(f"HTTP {response.status_code} for {user.get('username', chat_id)}")
                print(f"❌ HTTP {response.status_code} for {user.get('username', chat_id)}: {response.text}")
        
        except requests.exceptions.RequestException as e:
            failed_count += 1
            errors.append(f"Request error for {user.get('username', chat_id)}: {str(e)}")
            print(f"❌ Request error for {user.get('username', chat_id)}: {str(e)}")
        
        # Rate limiting
        time.sleep(0.1)
    
    return {
        'success': sent_count > 0,
        'sent_count': sent_count,
        'failed_count': failed_count,
        'errors': errors
    }


def main():
    """Main function to send the update announcement."""
    
    print("📨 Sending Update Announcement to Telegram Users (Plain Text)")
    print("=" * 70)
    
    # Read the update announcement
    update_file = Path("update_plain.txt")
    if not update_file.exists():
        print("❌ update_plain.txt file not found")
        return False
    
    try:
        with open(update_file, 'r', encoding='utf-8') as f:
            update_content = f.read()
        
        print(f"✅ Loaded update announcement ({len(update_content)} characters)")
        
        # Preview first 500 characters
        preview = update_content[:500]
        if len(update_content) > 500:
            preview += "\n\n[... truncated for preview ...]"
        
        print("\n👁️  Preview:")
        print("-" * 50)
        print(preview)
        print("-" * 50)
        
        # Ask for confirmation if running interactively
        if sys.stdin.isatty():
            response = input("\nSend this announcement to active Telegram users? (y/N): ").strip().lower()
            if response != 'y':
                print("📝 Announcement not sent")
                return False
        
        print("\n📤 Sending announcement to active users...")
        
        # Send the message using plain text mode
        result = send_plain_text_message(update_content)
        
        # Report results
        if result['success']:
            print(f"\n✅ Announcement sent successfully!")
            print(f"   📊 Sent to: {result['sent_count']} users")
            if result['failed_count'] > 0:
                print(f"   ⚠️  Failed: {result['failed_count']} users")
                for error in result['errors']:
                    print(f"      • {error}")
        else:
            print(f"\n❌ Failed to send announcement")
            print(f"   📊 Sent: {result['sent_count']}, Failed: {result['failed_count']}")
            for error in result['errors']:
                print(f"   • {error}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Update announcement sent to active Telegram users!")
        print("📱 Users will now see the enhanced superduper alert features")
    else:
        print("\n❌ Failed to send update announcement")
        print("🔧 Check Telegram configuration and try again")
    
    print("\n📋 Summary of announced features:")
    print("  📊 MACD Technical Analysis with 🔴 filtering")
    print("  ⚡ Short-term momentum with traffic lights")
    print("  🕐 Time of day market timing")
    print("  📨 Enhanced message format with visual indicators")