#!/usr/bin/env python3
"""
Telegram Debug Tool

This tool helps debug Telegram bot configuration and get proper chat IDs.
"""

import sys
import os
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.telegram.config import TelegramConfig

def get_bot_info():
    """Get bot information from Telegram API."""
    try:
        TelegramConfig.validate_config()
        
        url = f"https://api.telegram.org/bot{TelegramConfig.BOT_TOKEN}/getMe"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                bot_info = result.get('result', {})
                print("🤖 Bot Information:")
                print(f"   ID: {bot_info.get('id')}")
                print(f"   Username: @{bot_info.get('username')}")
                print(f"   First Name: {bot_info.get('first_name')}")
                print(f"   Can Join Groups: {bot_info.get('can_join_groups')}")
                print(f"   Can Read All Group Messages: {bot_info.get('can_read_all_group_messages')}")
                return True
            else:
                print(f"❌ Bot API Error: {result.get('description')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting bot info: {e}")
    
    return False

def get_updates():
    """Get recent updates to find chat IDs."""
    try:
        url = f"https://api.telegram.org/bot{TelegramConfig.BOT_TOKEN}/getUpdates"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                updates = result.get('result', [])
                
                if not updates:
                    print("📭 No recent messages found")
                    print("💡 Send a message to your bot first, then run this again")
                    return
                
                print(f"📨 Found {len(updates)} recent updates:")
                print("-" * 50)
                
                chat_ids = set()
                for update in updates[-10:]:  # Show last 10 updates
                    message = update.get('message', {})
                    chat = message.get('chat', {})
                    from_user = message.get('from', {})
                    
                    chat_id = chat.get('id')
                    chat_type = chat.get('type')
                    username = from_user.get('username', 'N/A')
                    first_name = from_user.get('first_name', 'N/A')
                    text = message.get('text', 'N/A')
                    
                    print(f"💬 Chat ID: {chat_id}")
                    print(f"   Type: {chat_type}")
                    print(f"   From: {first_name} (@{username})")
                    print(f"   Message: {text}")
                    print()
                    
                    if chat_id:
                        chat_ids.add(str(chat_id))
                
                if chat_ids:
                    print("📋 Unique Chat IDs found:")
                    for chat_id in sorted(chat_ids):
                        print(f"   {chat_id}")
                    
                    print("\n💡 Copy one of these chat IDs to your .telegram_users.csv file")
                
            else:
                print(f"❌ API Error: {result.get('description')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting updates: {e}")

def test_send_to_chat(chat_id: str):
    """Test sending a message to a specific chat ID."""
    try:
        url = f"https://api.telegram.org/bot{TelegramConfig.BOT_TOKEN}/sendMessage"
        
        payload = {
            'chat_id': chat_id,
            'text': f'🧪 Test message from debug tool at {os.popen("date").read().strip()}',
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print(f"✅ Message sent successfully to chat {chat_id}")
                return True
            else:
                print(f"❌ Send failed: {result.get('description')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error sending test message: {e}")
    
    return False

def main():
    """Main debug function."""
    print("🔍 TELEGRAM DEBUG TOOL")
    print("=" * 40)
    
    # Check configuration
    print("\n1. 🔧 Checking Configuration...")
    try:
        TelegramConfig.validate_config()
        print("✅ Configuration valid")
        print(f"   Token: ...{TelegramConfig.BOT_TOKEN[-10:]}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Make sure your .env file has BOT_TOKEN set")
        return
    
    # Get bot info
    print("\n2. 🤖 Getting Bot Information...")
    if not get_bot_info():
        print("❌ Could not get bot information")
        return
    
    # Get recent updates
    print("\n3. 📨 Checking Recent Messages...")
    get_updates()
    
    # Test specific chat ID
    print("\n4. 🧪 Testing Specific Chat ID...")
    chat_id = input("Enter chat ID to test (or press Enter to skip): ").strip()
    
    if chat_id:
        test_send_to_chat(chat_id)
    
    print("\n✅ Debug complete!")
    print("\n💡 Next steps:")
    print("   1. Make sure you've sent a message to your bot")
    print("   2. Use a chat ID from the updates above")
    print("   3. Update your .telegram_users.csv file")
    print("   4. Test again with the interactive menu")

if __name__ == "__main__":
    main()