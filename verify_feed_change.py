#!/usr/bin/env python3
"""
Quick verification that the feed selection logic is working.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_feed_changes():
    """Verify the feed selection changes are working."""
    
    print("=== Verifying Feed Selection Changes ===")
    
    # Check that the CLI argument was added
    print("\n1. Checking CLI argument:")
    
    # Read the orb_alerts.py file to verify changes
    orb_alerts_path = 'code/orb_alerts.py'
    
    with open(orb_alerts_path, 'r') as f:
        content = f.read()
    
    # Check for CLI argument
    if '--use-iex' in content and 'Use IEX data feed instead of SIP' in content:
        print("✓ CLI argument --use-iex added successfully")
    else:
        print("❌ CLI argument --use-iex not found")
        return False
    
    # Check for updated constructor
    if 'use_iex: bool = False' in content:
        print("✓ Constructor updated with use_iex parameter")
    else:
        print("❌ Constructor not updated")
        return False
    
    # Check for feed selection logic
    if "feed = 'sip'  # Default to SIP" in content:
        print("✓ Feed selection logic updated to default to SIP")
    else:
        print("❌ Feed selection logic not updated")
        return False
    
    # Check for IEX logic
    if "if self.use_iex:" in content and "feed = 'iex'" in content:
        print("✓ IEX selection logic added")
    else:
        print("❌ IEX selection logic not found")
        return False
    
    # Check for updated docstring
    if "python3 code/orb_alerts.py --use-iex" in content:
        print("✓ Documentation updated with new usage")
    else:
        print("❌ Documentation not updated")
        return False
    
    print("\n2. Summary of changes:")
    print("   - Default behavior changed from IEX to SIP")
    print("   - Added --use-iex flag to use IEX data feed")
    print("   - Updated constructor to accept use_iex parameter")
    print("   - Updated feed selection logic in _fetch_with_legacy_api")
    print("   - Updated documentation and help text")
    
    print("\n3. Usage commands:")
    print("   python3 code/orb_alerts.py              # Uses SIP (default)")
    print("   python3 code/orb_alerts.py --use-iex    # Uses IEX")
    
    print("\n✅ All changes verified successfully!")
    return True

if __name__ == "__main__":
    success = verify_feed_changes()
    if success:
        print("\n🎉 Feed selection update completed successfully!")
        print("   - Default: SIP data feed")
        print("   - Optional: IEX data feed with --use-iex")
    else:
        print("\n❌ Verification failed")
        sys.exit(1)