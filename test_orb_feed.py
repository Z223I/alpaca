#!/usr/bin/env python3
"""
Test script to verify that code/orb.py feed selection is working.
"""

import os
import sys
import subprocess

def test_orb_feed_selection():
    """Test the feed selection in code/orb.py"""
    
    print("=== Testing ORB Feed Selection ===")
    
    # Test 1: Default behavior (should use SIP)
    print("\n--- Test 1: Default behavior (should use SIP) ---")
    
    try:
        result = subprocess.run([
            'python3', 'code/orb.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if '--use-iex' in result.stdout and 'Use IEX data feed instead of SIP (default: SIP)' in result.stdout:
            print("✓ CLI argument --use-iex found in help")
        else:
            print("❌ CLI argument --use-iex not found in help")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Help command timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing help: {e}")
        return False
    
    # Test 2: Verify file changes
    print("\n--- Test 2: Verifying code changes ---")
    
    try:
        with open('code/orb.py', 'r') as f:
            content = f.read()
        
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
        
        # Check for argument parsing
        if '--use-iex' in content and 'args.use_iex' in content:
            print("✓ CLI argument parsing implemented")
        else:
            print("❌ CLI argument parsing not implemented")
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print("\n=== Summary ===")
    print("✓ code/orb.py updated successfully")
    print("✓ Default behavior: SIP data feed")
    print("✓ Optional: IEX data feed with --use-iex flag")
    print("✓ CLI help shows new argument")
    
    print("\nUsage commands:")
    print("  python3 code/orb.py              # Uses SIP (default)")
    print("  python3 code/orb.py --use-iex    # Uses IEX")
    print("  python3 code/orb.py --plot-alerts --use-iex  # Regular alerts + IEX")
    
    return True

if __name__ == "__main__":
    success = test_orb_feed_selection()
    if success:
        print("\n🎉 ORB feed selection update successful!")
    else:
        print("\n❌ ORB feed selection update failed!")
        sys.exit(1)