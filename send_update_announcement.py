#!/usr/bin/env python3
"""
Send update announcement to active Telegram users.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, '.')

try:
    from atoms.telegram.telegram_post import TelegramPoster
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("❌ Telegram integration not available - missing dependencies")


def send_update_announcement():
    """Send the update.txt announcement to active Telegram users."""
    
    print("📨 Sending Update Announcement to Telegram Users")
    print("=" * 60)
    
    # Check if Telegram is available
    if not TELEGRAM_AVAILABLE:
        print("❌ Cannot send - Telegram integration not available")
        return False
    
    # Read the update announcement (use plain text version for Telegram)
    update_file = Path("update_plain.txt")
    if not update_file.exists():
        print("❌ update_plain.txt file not found")
        return False
    
    try:
        with open(update_file, 'r', encoding='utf-8') as f:
            update_content = f.read()
        
        print(f"✅ Loaded update announcement ({len(update_content)} characters)")
        
        # Initialize Telegram poster
        telegram_poster = TelegramPoster()
        
        print("📤 Sending announcement to active users...")
        
        # Send the update (not urgent, but informational)
        result = telegram_poster.send_message(update_content, urgent=False)
        
        # Report results
        if result['success']:
            print(f"✅ Announcement sent successfully!")
            print(f"   📊 Sent to: {result['sent_count']} users")
            if result['failed_count'] > 0:
                print(f"   ⚠️  Failed: {result['failed_count']} users")
                for error in result['errors']:
                    print(f"      • {error}")
        else:
            print(f"❌ Failed to send announcement")
            print(f"   📊 Sent: {result['sent_count']}, Failed: {result['failed_count']}")
            for error in result['errors']:
                print(f"   • {error}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error sending announcement: {e}")
        return False


def preview_announcement():
    """Preview the announcement before sending."""
    
    print("👁️  Preview of Update Announcement")
    print("=" * 60)
    
    update_file = Path("update_plain.txt")
    if not update_file.exists():
        print("❌ update_plain.txt file not found")
        return
    
    try:
        with open(update_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Show first 500 characters
        preview = content[:500]
        if len(content) > 500:
            preview += "\n\n[... truncated for preview ...]"
        
        print(preview)
        print("=" * 60)
        print(f"📊 Full announcement: {len(content)} characters")
        
    except Exception as e:
        print(f"❌ Error reading announcement: {e}")


if __name__ == "__main__":
    # Show preview first
    preview_announcement()
    
    print("\n" + "=" * 60)
    
    # Ask for confirmation if running interactively
    if sys.stdin.isatty():
        response = input("Send this announcement to active Telegram users? (y/N): ").strip().lower()
        if response != 'y':
            print("📝 Announcement not sent")
            sys.exit(0)
    
    # Send the announcement
    success = send_update_announcement()
    
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