#!/usr/bin/env python3
"""
Interactive Telegram Testing Menu

This module provides an interactive menu for testing different Telegram atoms
and functionality without requiring integration with existing trading code.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.telegram.telegram_post import send_message, send_alert, add_user, get_active_users
from atoms.telegram.user_manager import UserManager
from atoms.telegram.config import TelegramConfig

class TelegramTestMenu:
    """Interactive menu for testing Telegram functionality."""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.running = True
        
    def display_banner(self):
        """Display the application banner."""
        print("\n" + "="*60)
        print("🚀 TELEGRAM ATOM TESTING MENU")
        print("="*60)
        print("Interactive testing for Telegram posting atoms")
        print("-"*60)
    
    def display_menu(self):
        """Display the main menu options."""
        print("\n📋 MENU OPTIONS:")
        print("1. 📤 Send Test Message")
        print("2. 🚨 Send Alert (with formatting)")
        print("3. 👥 Manage Users")
        print("4. ⚙️  Check Configuration")
        print("5. 📊 View Active Users")
        print("6. 🧪 Run Test Scenarios")
        print("7. ❌ Exit")
        print("-"*40)
    
    def get_user_input(self, prompt: str, default: str = None) -> str:
        """Get user input with optional default value."""
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ").strip()
    
    def send_test_message(self):
        """Interactive message sending."""
        print("\n📤 SEND TEST MESSAGE")
        print("-"*30)
        
        message = self.get_user_input("Enter message to send")
        if not message:
            print("❌ No message entered")
            return
        
        urgent = self.get_user_input("Mark as urgent? (y/n)", "n").lower() == 'y'
        
        print(f"\n📨 Sending message: '{message}'")
        print(f"🚨 Urgent: {urgent}")
        
        try:
            result = send_message(message, urgent=urgent)
            
            print(f"\n✅ Result:")
            print(f"   Success: {result['success']}")
            print(f"   Sent to: {result['sent_count']} users")
            print(f"   Failed: {result['failed_count']} users")
            
            if result['errors']:
                print(f"   Errors: {result['errors']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def send_test_alert(self):
        """Interactive alert sending."""
        print("\n🚨 SEND ALERT MESSAGE")
        print("-"*30)
        
        message = self.get_user_input("Enter alert message")
        if not message:
            print("❌ No message entered")
            return
        
        print("\nAlert levels:")
        print("1. ℹ️  info")
        print("2. ⚠️  warning") 
        print("3. ❌ error")
        print("4. ✅ success")
        
        level_choice = self.get_user_input("Select level (1-4)", "1")
        level_map = {"1": "info", "2": "warning", "3": "error", "4": "success"}
        level = level_map.get(level_choice, "info")
        
        chat_id = self.get_user_input("Specific chat ID (leave empty for all users)")
        chat_id = chat_id if chat_id else None
        
        print(f"\n📨 Sending alert: '{message}' (level: {level})")
        
        try:
            result = send_alert(message, level=level, chat_id=chat_id)
            print(f"✅ Alert sent: {result}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def manage_users(self):
        """Interactive user management."""
        print("\n👥 USER MANAGEMENT")
        print("-"*30)
        print("1. ➕ Add User")
        print("2. ❌ Disable User")
        print("3. ✅ Enable User")
        print("4. 📋 List All Users")
        print("5. 🔙 Back to Main Menu")
        
        choice = self.get_user_input("Select option (1-5)")
        
        if choice == "1":
            self._add_user()
        elif choice == "2":
            self._disable_user()
        elif choice == "3":
            self._enable_user()
        elif choice == "4":
            self._list_all_users()
        elif choice == "5":
            return
        else:
            print("❌ Invalid choice")
    
    def _add_user(self):
        """Add new user to CSV."""
        print("\n➕ ADD NEW USER")
        print("-"*20)
        
        chat_id = self.get_user_input("Chat ID")
        if not chat_id:
            print("❌ Chat ID required")
            return
            
        username = self.get_user_input("Username (optional)")
        notes = self.get_user_input("Notes (optional)")
        
        try:
            success = add_user(chat_id, username, enabled=True, notes=notes)
            if success:
                print(f"✅ User {chat_id} added successfully")
            else:
                print(f"❌ Failed to add user (may already exist)")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _disable_user(self):
        """Disable existing user."""
        self._show_active_users()
        chat_id = self.get_user_input("Chat ID to disable")
        if not chat_id:
            return
            
        try:
            success = self.user_manager.disable_user(chat_id)
            if success:
                print(f"✅ User {chat_id} disabled")
            else:
                print(f"❌ User {chat_id} not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _enable_user(self):
        """Enable existing user."""
        chat_id = self.get_user_input("Chat ID to enable")
        if not chat_id:
            return
            
        try:
            success = self.user_manager.enable_user(chat_id)
            if success:
                print(f"✅ User {chat_id} enabled")
            else:
                print(f"❌ User {chat_id} not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _list_all_users(self):
        """List all users from CSV."""
        try:
            all_users = self.user_manager._read_all_users()
            
            print(f"\n📋 ALL USERS ({len(all_users)} total):")
            print("-"*50)
            
            if not all_users:
                print("No users found in CSV file")
                return
                
            for user in all_users:
                if user.get('chat_id', '').startswith('#') or user.get('chat_id', '') == 'chat_id':
                    continue
                    
                enabled = "✅" if user.get('enabled', '').lower() == 'true' else "❌"
                print(f"{enabled} {user.get('chat_id', 'N/A')} | {user.get('username', 'N/A')} | {user.get('notes', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def check_configuration(self):
        """Check Telegram configuration."""
        print("\n⚙️  CONFIGURATION CHECK")
        print("-"*30)
        
        try:
            # Test configuration validation
            TelegramConfig.validate_config()
            print("✅ Configuration is valid")
            
            # Display current settings
            print(f"\n📋 Current Settings:")
            print(f"   Bot Token: {'*' * 20}...{TelegramConfig.BOT_TOKEN[-10:] if TelegramConfig.BOT_TOKEN else 'NOT SET'}")
            print(f"   Enabled: {TelegramConfig.ENABLED}")
            print(f"   Parse Mode: {TelegramConfig.PARSE_MODE}")
            print(f"   Timeout: {TelegramConfig.TIMEOUT}s")
            print(f"   CSV Path: {TelegramConfig.CSV_FILE_PATH}")
            print(f"   CSV Exists: {os.path.exists(TelegramConfig.CSV_FILE_PATH)}")
            
        except Exception as e:
            print(f"❌ Configuration Error: {e}")
            print("\n💡 Tips:")
            print("   1. Check your .env file exists")
            print("   2. Verify BOT_TOKEN is set correctly")
            print("   3. Token format: 123456:ABC-DEF...")
    
    def _show_active_users(self):
        """Display active users."""
        try:
            users = get_active_users()
            
            print(f"\n👥 ACTIVE USERS ({len(users)} total):")
            print("-"*40)
            
            if not users:
                print("No active users found")
                print("💡 Add users via option 3 in main menu")
                return
                
            for user in users:
                print(f"   📱 {user.get('chat_id', 'N/A')} | {user.get('username', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def view_active_users(self):
        """Show active users with details."""
        print("\n📊 ACTIVE USERS")
        print("-"*30)
        self._show_active_users()
    
    def run_test_scenarios(self):
        """Run predefined test scenarios."""
        print("\n🧪 TEST SCENARIOS")
        print("-"*30)
        print("1. 📝 Basic message test")
        print("2. 🚨 All alert levels test")
        print("3. 📊 Portfolio update simulation")
        print("4. 🔄 Error handling test")
        print("5. 🔙 Back to Main Menu")
        
        choice = self.get_user_input("Select test (1-5)")
        
        if choice == "1":
            self._test_basic_message()
        elif choice == "2":
            self._test_all_alerts()
        elif choice == "3":
            self._test_portfolio_update()
        elif choice == "4":
            self._test_error_handling()
        elif choice == "5":
            return
        else:
            print("❌ Invalid choice")
    
    def _test_basic_message(self):
        """Test basic message sending."""
        print("\n📝 Testing basic message...")
        test_message = f"🧪 Test message sent at {datetime.now().strftime('%H:%M:%S')}"
        
        try:
            result = send_message(test_message)
            print(f"✅ Basic test result: {result}")
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
    
    def _test_all_alerts(self):
        """Test all alert levels."""
        print("\n🚨 Testing all alert levels...")
        
        alerts = [
            ("info", "ℹ️ Information: System status normal"),
            ("warning", "⚠️ Warning: High volatility detected"),
            ("error", "❌ Error: Connection timeout occurred"),
            ("success", "✅ Success: Order executed successfully")
        ]
        
        for level, message in alerts:
            try:
                result = send_alert(message, level=level)
                print(f"   {level}: {result}")
            except Exception as e:
                print(f"   {level}: Failed - {e}")
    
    def _test_portfolio_update(self):
        """Test portfolio update message."""
        print("\n📊 Testing portfolio update simulation...")
        
        portfolio_message = """📊 Portfolio Update
💰 Total Value: $12,450.30 (+2.3%)
💵 Cash: $2,100.00
📈 Day P&L: +$287.50 (+2.37%)

Top Positions:
• AAPL: 10 shares @ $150.25 (+$45.00)
• TSLA: 5 shares @ $220.00 (-$12.50)"""
        
        try:
            result = send_message(portfolio_message)
            print(f"✅ Portfolio test result: {result}")
        except Exception as e:
            print(f"❌ Portfolio test failed: {e}")
    
    def _test_error_handling(self):
        """Test error handling scenarios."""
        print("\n🔄 Testing error handling...")
        
        # Test with invalid configuration
        original_token = os.environ.get('BOT_TOKEN')
        
        try:
            # Temporarily break the token
            os.environ['BOT_TOKEN'] = 'invalid_token'
            
            result = send_message("This should fail")
            print(f"   Invalid token test: {result}")
            
        except Exception as e:
            print(f"   Invalid token test caught error: {e}")
        finally:
            # Restore original token
            if original_token:
                os.environ['BOT_TOKEN'] = original_token
            elif 'BOT_TOKEN' in os.environ:
                del os.environ['BOT_TOKEN']
    
    def run(self):
        """Main menu loop."""
        self.display_banner()
        
        while self.running:
            try:
                self.display_menu()
                choice = self.get_user_input("Select option (1-7)")
                
                if choice == "1":
                    self.send_test_message()
                elif choice == "2":
                    self.send_test_alert()
                elif choice == "3":
                    self.manage_users()
                elif choice == "4":
                    self.check_configuration()
                elif choice == "5":
                    self.view_active_users()
                elif choice == "6":
                    self.run_test_scenarios()
                elif choice == "7":
                    print("\n👋 Goodbye!")
                    self.running = False
                else:
                    print("❌ Invalid choice. Please select 1-7.")
                
                if self.running:
                    input("\n⏸️  Press Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                input("⏸️  Press Enter to continue...")

def main():
    """Entry point for the interactive menu."""
    menu = TelegramTestMenu()
    menu.run()

if __name__ == "__main__":
    main()