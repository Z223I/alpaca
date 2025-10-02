#!/usr/bin/env python3
"""
Alpaca Configuration Modifier

This script allows you to modify the alpaca_config.py file by updating
trading parameters for specific accounts and environments.

Usage:
    python configure_alpaca.py --account-name Bruce --account paper --auto-trade yes --auto-amount 5000
    python configure_alpaca.py --account-name "Dale Wilson" --account live --trailing-percent 15.0
"""

import argparse
import re
import sys
from typing import Dict, Any


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Configure Alpaca trading parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --account-name Bruce --account paper --auto-trade yes --auto-amount 5000
  %(prog)s --account-name "Dale Wilson" --account live --trailing-percent 15.0
  %(prog)s --account-name Janice --account cash --max-trades-per-day 3
        """
    )

    # Account selection arguments
    parser.add_argument(
        "--account-name",
        default="Bruce",
        help="Account name to use (default: Bruce)"
    )

    parser.add_argument(
        "--account",
        choices=["paper", "live", "cash"],
        default="paper",
        help="Account environment to use: paper, live, cash (default: paper)"
    )

    # Configuration parameters
    parser.add_argument(
        "--auto-trade",
        choices=["yes", "no"],
        help="Enable or disable auto trading"
    )

    parser.add_argument(
        "--auto-amount",
        type=int,
        help="Auto trading amount"
    )

    parser.add_argument(
        "--trailing-percent",
        type=float,
        help="Trailing stop percentage"
    )

    parser.add_argument(
        "--take-profit-percent",
        type=float,
        help="Take profit percentage"
    )

    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        help="Maximum trades per day"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying the file"
    )

    return parser.parse_args()


def find_account_section(content: str, account_name: str, environment: str) -> tuple[int, int]:
    """
    Find the start and end positions of the account/environment section.

    Returns:
        Tuple of (start_pos, end_pos) or (-1, -1) if not found
    """
    # Escape special regex characters in account name
    escaped_account = re.escape(account_name)

    # Pattern to find the account section
    account_pattern = rf'"{escaped_account}":\s*AccountConfig\s*\('
    account_match = re.search(account_pattern, content)

    if not account_match:
        return -1, -1

    # Find the specific environment within this account
    # Start searching from the account match position
    search_start = account_match.start()

    # Pattern to find the environment section
    env_pattern = rf'{environment}=EnvironmentConfig\s*\('
    env_match = re.search(env_pattern, content[search_start:])

    if not env_match:
        return -1, -1

    # Adjust position to be relative to the full content
    env_start = search_start + env_match.start()

    # Find the end of this EnvironmentConfig block
    # We need to find the matching closing parenthesis
    open_parens = 0
    pos = search_start + env_match.end()

    for i, char in enumerate(content[pos:], pos):
        if char == '(':
            open_parens += 1
        elif char == ')':
            if open_parens == 0:
                return env_start, i + 1
            open_parens -= 1

    return -1, -1


def update_parameter(content: str, start_pos: int, end_pos: int, param_name: str, new_value: Any) -> str:
    """Update a parameter within the specified section."""
    section = content[start_pos:end_pos]

    # Format the new value appropriately
    if isinstance(new_value, str):
        formatted_value = f'"{new_value}"'
    elif isinstance(new_value, (int, float)):
        formatted_value = str(new_value)
    else:
        formatted_value = str(new_value)

    # Pattern to match the parameter line
    param_pattern = rf'(\s*{re.escape(param_name)}\s*=\s*)[^,\n]+([,\n])'

    # Try to update existing parameter
    match = re.search(param_pattern, section)
    if match:
        # Replace the existing value
        new_section = re.sub(
            param_pattern,
            rf'\g<1>{formatted_value}\g<2>',
            section
        )
        return content[:start_pos] + new_section + content[end_pos:]
    else:
        print(f"Warning: Parameter '{param_name}' not found in the configuration section")
        return content


def main():
    """Main function."""
    args = parse_arguments()

    # Check if any parameters were specified
    params_to_update = {}
    if args.auto_trade is not None:
        params_to_update["auto_trade"] = args.auto_trade
    if args.auto_amount is not None:
        params_to_update["auto_amount"] = args.auto_amount
    if args.trailing_percent is not None:
        params_to_update["trailing_percent"] = args.trailing_percent
    if args.take_profit_percent is not None:
        params_to_update["take_profit_percent"] = args.take_profit_percent
    if args.max_trades_per_day is not None:
        params_to_update["max_trades_per_day"] = args.max_trades_per_day

    if not params_to_update:
        print("Error: No parameters specified to update")
        print("Use --help to see available options")
        return 1

    # Read the configuration file
    config_file = "code/alpaca_config.py"
    try:
        with open(config_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        return 1
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return 1

    # Find the account/environment section
    start_pos, end_pos = find_account_section(content, args.account_name, args.account)

    if start_pos == -1:
        print(f"Error: Could not find configuration section for account '{args.account_name}' environment '{args.account}'")
        return 1

    print(f"Found configuration section for {args.account_name}/{args.account}")

    # Update each specified parameter
    modified_content = content
    for param_name, new_value in params_to_update.items():
        print(f"Updating {param_name} = {new_value}")
        modified_content = update_parameter(modified_content, start_pos, end_pos, param_name, new_value)
        # Update positions since content length may have changed
        start_pos, end_pos = find_account_section(modified_content, args.account_name, args.account)

    if args.dry_run:
        print("\n=== DRY RUN - Changes that would be made ===")
        # Show the modified section
        new_start, new_end = find_account_section(modified_content, args.account_name, args.account)
        if new_start != -1:
            print(modified_content[new_start:new_end])
        print("=== No changes written ===")
        return 0

    # Write the modified content back to the file
    try:
        with open(config_file, 'w') as f:
            f.write(modified_content)
        print(f"Successfully updated {config_file}")
        return 0
    except Exception as e:
        print(f"Error writing configuration file: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())