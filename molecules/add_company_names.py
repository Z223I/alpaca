#!/usr/bin/env python3
"""
Add Company Names and Float Shares to Stock Symbol CSV

Reads a CSV file with a 'symbol' column, fetches company names from Alpaca API,
fetches float shares from Yahoo Finance via yfinance, and adds columns for both.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

# Add the atoms directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'atoms', 'api'))

from init_alpaca_client import init_alpaca_client


def clean_company_name(full_name: str) -> str:
    """
    Clean company name by removing common suffixes like 'Common Stock', 'Inc.', etc.

    Args:
        full_name: Full asset name from Alpaca API

    Returns:
        Cleaned company name
    """
    if not full_name or full_name == "N/A":
        return full_name

    # List of suffixes to remove (order matters - more specific first)
    suffixes_to_remove = [
        ' Class A Ordinary Shares',
        ' Class B Ordinary Shares',
        ' Class C Ordinary Shares',
        ' Class A Common Stock',
        ' Class B Common Stock',
        ' Class C Common Stock',
        ' Common Stock',
        ' Ordinary Shares',
        ' American Depositary Shares',
        ' American Depositary Receipt',
        ' Depositary Shares',
        ' ETF Trust',
        ' Shares',
        ' Inc.',
        ' Inc',
        ' Corporation',
        ' Corp.',
        ' Corp',
        ' Limited',
        ' Ltd.',
        ' Ltd',
        ' L.P.',
        ' LP',
        ' LLC',
        ' PLC',
        ' plc',
        ' S.A.',
        ' SA',
        ' N.V.',
        ' NV',
        ' AG',
    ]

    cleaned = full_name.strip()

    # Remove suffixes
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
            # Check again for multiple suffixes (e.g., "Company Inc. Common Stock")
            for suffix2 in suffixes_to_remove:
                if cleaned.endswith(suffix2):
                    cleaned = cleaned[:-len(suffix2)].strip()
                    break
            break

    # Remove trailing comma if present
    cleaned = cleaned.rstrip(',').strip()

    return cleaned


def get_float_shares(symbols: List[str], verbose: bool = False) -> Dict[str, Optional[int]]:
    """
    Fetch float shares for a list of stock symbols from Yahoo Finance via yfinance.

    Note: This makes individual API calls for each symbol and can be slow for large lists.
    Consider using batch_size parameter or running during off-peak hours.

    Args:
        symbols: List of stock ticker symbols
        verbose: Enable verbose logging

    Returns:
        Dictionary mapping symbols to float shares (None if not available)
    """
    try:
        import yfinance as yf
    except ImportError:
        if verbose:
            print("ERROR: yfinance library not installed. Install with: pip install yfinance")
        return {symbol: None for symbol in symbols}

    if verbose:
        print(f"Fetching float shares for {len(symbols)} symbols from Yahoo Finance...")
        print(f"  Note: This requires individual API calls and may take several minutes")
        start_time = time.time()

    float_data = {}
    failed_count = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            float_shares = info.get('floatShares')
            float_data[symbol] = float_shares

            if verbose and i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(symbols) - i) / rate if rate > 0 else 0
                print(f"  Processed {i}/{len(symbols)} symbols ({rate:.1f}/sec, ~{remaining:.0f}s remaining)")

        except Exception as e:
            if verbose and i <= 5:  # Only show first few errors
                print(f"  Warning: Could not fetch float for {symbol}: {e}")
            float_data[symbol] = None
            failed_count += 1

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    if verbose:
        total_elapsed = time.time() - start_time
        success_count = len([v for v in float_data.values() if v is not None])
        print(f"Successfully fetched {success_count}/{len(symbols)} float values in {total_elapsed:.1f} seconds")
        if failed_count > 0:
            print(f"  Failed to fetch: {failed_count} symbols")

    return float_data


def get_company_names(symbols: List[str], provider: str = "alpaca",
                      account: str = "Bruce", environment: str = "paper",
                      verbose: bool = False) -> Dict[str, str]:
    """
    Fetch company names for a list of stock symbols from Alpaca API.

    Uses an efficient batch approach: fetches ALL assets once, then filters locally.
    This is much faster than making individual API calls for each symbol.

    Args:
        symbols: List of stock ticker symbols
        provider: API provider (default: "alpaca")
        account: Account name (default: "Bruce")
        environment: Environment type (default: "paper")
        verbose: Enable verbose logging

    Returns:
        Dictionary mapping symbols to company names
    """
    import time

    if verbose:
        print(f"Initializing Alpaca client (account: {account}, environment: {environment})")

    client = init_alpaca_client(provider, account, environment)

    if verbose:
        print(f"Fetching all assets in one batch request (much faster than {len(symbols)} individual calls)...")
        start_time = time.time()

    # Fetch ALL assets at once - this is a single API call!
    try:
        all_assets = client.list_assets(status='active', asset_class='us_equity')

        if verbose:
            elapsed = time.time() - start_time
            print(f"  Fetched {len(all_assets)} total assets in {elapsed:.2f} seconds")
            print(f"  Creating symbol-to-name mapping...")

        # Create lookup dictionary for fast access (with cleaned names)
        symbol_to_name = {asset.symbol: clean_company_name(asset.name) for asset in all_assets}

        if verbose:
            print(f"  Mapping {len(symbols)} requested symbols to company names...")

        # Map requested symbols to company names
        company_names = {}
        for symbol in symbols:
            company_names[symbol] = symbol_to_name.get(symbol, "N/A")

        found_count = len([v for v in company_names.values() if v != 'N/A'])
        if verbose:
            total_elapsed = time.time() - start_time
            print(f"Successfully matched {found_count}/{len(symbols)} symbols in {total_elapsed:.2f} seconds")
            if found_count < len(symbols):
                missing = [s for s, n in company_names.items() if n == 'N/A']
                print(f"  Missing symbols: {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))

        return company_names

    except Exception as e:
        if verbose:
            print(f"Error fetching assets: {e}")
        # Fallback: return N/A for all symbols
        return {symbol: "N/A" for symbol in symbols}


def process_csv(input_file: str, output_file: str, provider: str = "alpaca",
                account: str = "Bruce", environment: str = "paper",
                fetch_float: bool = False, verbose: bool = False):
    """
    Read CSV file, add company names and optionally float shares, and write to output file.

    Args:
        input_file: Path to input CSV file with 'symbol' column
        output_file: Path to output CSV file
        provider: API provider (default: "alpaca")
        account: Account name (default: "Bruce")
        environment: Environment type (default: "paper")
        fetch_float: Fetch float shares from Yahoo Finance (default: False, can be slow)
        verbose: Enable verbose logging
    """
    # Verify input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if verbose:
        print(f"Reading input file: {input_file}")

    # Read input CSV
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Check for symbol column
        if 'symbol' not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'symbol' column")

        # Read all rows
        rows = list(reader)
        original_fieldnames = reader.fieldnames

    if not rows:
        print("Warning: Input file contains no data rows")
        return

    # Extract unique symbols
    symbols = list(set(row['symbol'] for row in rows if row.get('symbol')))

    if verbose:
        print(f"Found {len(symbols)} unique symbols in input file")

    # Fetch company names
    company_names = get_company_names(symbols, provider, account, environment, verbose)

    # Fetch float shares if requested
    float_shares = {}
    if fetch_float:
        float_shares = get_float_shares(symbols, verbose)

    # Add company names and float shares to rows
    for row in rows:
        symbol = row.get('symbol', '')
        row['company_name'] = company_names.get(symbol, 'N/A')
        if fetch_float:
            float_val = float_shares.get(symbol)
            row['float'] = float_val if float_val is not None else 'N/A'

    # Create new fieldnames with company_name and float after symbol
    new_fieldnames = []
    for field in original_fieldnames:
        new_fieldnames.append(field)
        if field == 'symbol':
            new_fieldnames.append('company_name')
            if fetch_float:
                new_fieldnames.append('float')

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Write output CSV
    if verbose:
        print(f"Writing output file: {output_file}")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully processed {len(rows)} rows")
    print(f"Output written to: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Add company names and float shares to stock symbol CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python molecules/add_company_names.py -i input.csv -o output.csv
  python molecules/add_company_names.py -i data/stocks.csv -o data/stocks_with_names.csv -v
  python molecules/add_company_names.py -i input.csv -o output.csv --float -v
  python molecules/add_company_names.py -i input.csv -o output.csv --account-name Dale --account live --float
        """
    )

    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input CSV file path (must contain "symbol" column)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output CSV file path')

    # API configuration
    parser.add_argument('--provider', default='alpaca',
                        help='API provider (default: alpaca)')
    parser.add_argument('--account-name', default='Bruce',
                        help='Account name (default: Bruce)')
    parser.add_argument('--account', default='paper',
                        help='Account type (default: paper)')

    # Data fetching options
    parser.add_argument('--float', action='store_true',
                        help='Fetch float shares from Yahoo Finance (slow for large lists)')

    # Output options
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        process_csv(
            input_file=args.input,
            output_file=args.output,
            provider=args.provider,
            account=args.account_name,
            environment=args.account,
            fetch_float=getattr(args, 'float', False),
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
