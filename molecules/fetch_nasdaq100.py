#!/usr/bin/env python3
"""
Fetch NASDAQ 100 Symbols from Wikipedia

Reads the NASDAQ 100 constituent table from Wikipedia and saves
symbols, company names, industry, and subsector to a CSV file.
"""

import io
import os
import sys
import pandas as pd
import urllib.request


OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_master', 'nasdaq_100.csv')
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def fetch_nasdaq100(output_file: str = OUTPUT_FILE, verbose: bool = False) -> list:
    """
    Fetch NASDAQ 100 constituents from Wikipedia and save to CSV.

    Columns saved: symbol, company, industry, subsector

    Args:
        output_file: Path to save the output CSV
        verbose: Enable verbose logging

    Returns:
        List of symbols fetched
    """
    if verbose:
        print("Fetching NASDAQ 100 constituents from Wikipedia...")

    req = urllib.request.Request(WIKIPEDIA_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        html = response.read()
    tables = pd.read_html(io.BytesIO(html))

    # Find the table with a 'Ticker' column
    constituents = None
    for i, table in enumerate(tables):
        if 'Ticker' in table.columns:
            constituents = table
            if verbose:
                print(f"  Found constituents table (index {i}) with {len(table)} rows")
                print(f"  Columns: {list(table.columns)}")
            break

    if constituents is None:
        raise ValueError("Could not find a table with 'Ticker' column on Wikipedia page")

    constituents = constituents.dropna(subset=['Ticker'])

    # Normalize column names to lowercase, strip footnote references like [14]
    col_map = {}
    for col in constituents.columns:
        clean = col.split('[')[0].strip().lower().replace(' ', '_')
        col_map[col] = clean
    constituents = constituents.rename(columns=col_map)

    # Rename to standard output column names
    rename = {
        'ticker': 'symbol',
        'company': 'company',
        'icb_industry': 'industry',
        'icb_subsector': 'subsector',
    }
    constituents = constituents.rename(columns={k: v for k, v in rename.items() if k in constituents.columns})

    # Keep only known columns in preferred order
    ordered = [c for c in ['symbol', 'company', 'industry', 'subsector'] if c in constituents.columns]
    constituents = constituents[ordered]

    if verbose:
        print(f"  Found {len(constituents)} symbols")

    # Save to CSV
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    constituents.to_csv(output_file, index=False)

    print(f"Saved {len(constituents)} rows to: {output_file}")
    return constituents['symbol'].tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch NASDAQ 100 symbols from Wikipedia',
        epilog="Example: python molecules/fetch_nasdaq100.py -v"
    )
    parser.add_argument('-o', '--output', default=OUTPUT_FILE,
                        help=f'Output CSV file path (default: {OUTPUT_FILE})')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    try:
        symbols = fetch_nasdaq100(output_file=args.output, verbose=args.verbose)
        if args.verbose:
            print("Symbols:", ', '.join(symbols[:10]), '...' if len(symbols) > 10 else '')
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
