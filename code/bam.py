#!/usr/bin/env python3
"""
BAM - Bulk Account Manager

This script liquidates all positions and cancels all orders for accounts
where auto_trade="yes" in the configuration.

It processes accounts in the following order for both operations:
1. live
2. cash
3. paper

Usage:
    python3 code/bam.py [--dry-run]

Options:
    --dry-run    Show what would be done without executing (default: False)
"""

import subprocess
import sys
from typing import List, Tuple, Dict
from alpaca_config import get_current_config


def get_auto_trade_accounts() -> List[Tuple[str, str]]:
    """
    Get all account-name/account combinations where auto_trade="yes".

    Returns:
        List of tuples containing (account_name, environment)
    """
    config = get_current_config()
    auto_trade_accounts = []

    # Iterate through all providers (currently only "alpaca")
    for provider_name, provider_config in config.providers.items():
        # Iterate through all accounts
        for account_name, account_config in provider_config.accounts.items():
            # Check each environment (live, cash, paper) in priority order
            environments = ["live", "cash", "paper"]
            for env in environments:
                env_config = getattr(account_config, env)
                if env_config.auto_trade == "yes":
                    auto_trade_accounts.append((account_name, env))

    return auto_trade_accounts


def execute_command(account_name: str, environment: str,
                    dry_run: bool = False) -> bool:
    """
    Execute liquidation and order cancellation for a specific account.

    Args:
        account_name: Name of the account
        environment: Environment (live, cash, paper)
        dry_run: If True, only show what would be executed

    Returns:
        True if successful, False if failed
    """
    cmd = [
        "python3", "code/alpaca.py",
        "--account-name", account_name,
        "--account", environment,
        "--liquidate-all",
        "--cancel-orders",
        "--submit"
    ]
    action = "Liquidating all positions and cancelling all orders"

    print(f"🔄 {action} for {account_name}:{environment}")

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True)
        print(f"✅ Successfully completed liquidation and cancellation for "
              f"{account_name}:{environment}")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed liquidation/cancellation for "
              f"{account_name}:{environment}")
        print(f"   Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during liquidation/cancellation for "
              f"{account_name}:{environment}: {str(e)}")
        return False


def process_accounts(auto_trade_accounts: List[Tuple[str, str]],
                     dry_run: bool = False) -> Dict[str, int]:
    """
    Process all auto-trade accounts for liquidation and order cancellation.

    Args:
        auto_trade_accounts: List of (account_name, environment) tuples
        dry_run: If True, only show what would be executed

    Returns:
        Dictionary with success/failure counts
    """
    results = {
        "success": 0,
        "failed": 0
    }

    if not auto_trade_accounts:
        print("ℹ️  No accounts found with auto_trade='yes'")
        return results

    print(f"📋 Found {len(auto_trade_accounts)} auto-trade accounts to "
          f"process:")
    for account_name, env in auto_trade_accounts:
        print(f"   - {account_name}:{env}")
    print()

    # Process accounts in order: live, cash, paper
    environment_order = ["live", "cash", "paper"]

    # Liquidate positions and cancel orders together
    print("🚨 LIQUIDATING POSITIONS AND CANCELLING ORDERS")
    print("=" * 50)
    for env in environment_order:
        env_accounts = [(name, e) for name, e in auto_trade_accounts
                        if e == env]
        for account_name, environment in env_accounts:
            success = execute_command(account_name, environment, dry_run)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1

    return results


def print_summary(results: Dict[str, int], dry_run: bool = False):
    """Print execution summary."""
    print()
    print("📊 EXECUTION SUMMARY")
    print("=" * 40)

    mode = "[DRY RUN] " if dry_run else ""

    print(f"{mode}Account Operations:")
    print(f"   ✅ Successful: {results['success']}")
    print(f"   ❌ Failed: {results['failed']}")

    total_operations = sum(results.values())
    total_successful = results['success']

    if total_operations > 0:
        success_rate = (total_successful / total_operations) * 100
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% "
              f"({total_successful}/{total_operations})")
    print("\nNote: Each successful operation includes both position "
          "liquidation AND order cancellation for that account.")


def main():
    """Main entry point."""
    dry_run = "--dry-run" in sys.argv

    print("🚀 BAM - Bulk Account Manager")
    print("=" * 40)

    if dry_run:
        print("🔍 DRY RUN MODE - No actual operations will be performed")
        print()

    try:
        # Get accounts with auto_trade="yes"
        auto_trade_accounts = get_auto_trade_accounts()

        # Process the accounts
        results = process_accounts(auto_trade_accounts, dry_run)

        # Print summary
        print_summary(results, dry_run)

        # Exit with appropriate code
        if results['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
