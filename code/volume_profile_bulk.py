#!/usr/bin/env python3
"""
Volume Profile Bulk Processing Script

Reads a CSV file from data/YYYYMMDD.csv for today's date, adds a POC column,
runs volume_profile.py for each symbol, extracts avg_poc from JSON output,
and saves the results to data/YYYYMMDD_POC.csv.

Usage:
    python code/volume_profile_bulk.py [--date YYYYMMDD] [--debug]
"""

import os
import sys
import csv
import json
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class VolumeProfileBulkProcessor:
    """Bulk processor for volume profile analysis"""

    def __init__(self, date: str = None, debug: bool = False):
        """
        Initialize bulk processor

        Args:
            date: Date string in YYYYMMDD format (defaults to today)
            debug: Enable debug output
        """
        self.debug = debug
        self.date = date or datetime.now().strftime('%Y%m%d')
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'volume_profile_output')
        self.volume_profile_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'code', 'volume_profile.py'
        )

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _log(self, message: str):
        """Log debug message if debug is enabled"""
        if self.debug:
            print(f"[DEBUG] {message}")

    def find_input_csv(self) -> Optional[str]:
        """
        Find the input CSV file for the specified date

        Returns:
            Path to CSV file or None if not found
        """
        csv_filename = f"{self.date}.csv"
        csv_path = os.path.join(self.data_dir, csv_filename)

        if os.path.exists(csv_path):
            self._log(f"Found input CSV: {csv_path}")
            return csv_path

        # If today's file doesn't exist, try the most recent file
        self._log(f"Input CSV {csv_path} not found, looking for most recent file...")

        csv_files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv') and len(filename) == 12:  # YYYYMMDD.csv format
                try:
                    date_part = filename[:8]
                    datetime.strptime(date_part, '%Y%m%d')  # Validate date format
                    csv_files.append((date_part, os.path.join(self.data_dir, filename)))
                except ValueError:
                    continue

        if csv_files:
            # Sort by date and get the most recent
            csv_files.sort(reverse=True)
            most_recent = csv_files[0][1]
            self._log(f"Using most recent CSV: {most_recent}")
            return most_recent

        return None

    def read_csv_data(self, csv_path: str) -> List[Dict]:
        """
        Read CSV data and add POC column after Signal column

        Args:
            csv_path: Path to input CSV file

        Returns:
            List of dictionaries representing CSV rows with POC column added
        """
        data = []

        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Get original fieldnames and insert POC after Signal
            original_fields = reader.fieldnames.copy()
            if 'Signal' in original_fields:
                signal_index = original_fields.index('Signal')
                new_fields = original_fields[:signal_index+1] + ['POC'] + original_fields[signal_index+1:]
            else:
                # If no Signal column, add POC at the beginning
                new_fields = ['POC'] + original_fields

            self._log(f"Original columns: {original_fields}")
            self._log(f"New columns: {new_fields}")

            # Read all rows and add empty POC column
            for row in reader:
                new_row = {}
                for field in new_fields:
                    if field == 'POC':
                        new_row[field] = ''  # Initialize POC as empty
                    else:
                        new_row[field] = row.get(field, '')
                data.append(new_row)

        self._log(f"Read {len(data)} rows from CSV")
        return data

    def run_volume_profile_for_symbol(self, symbol: str) -> Optional[float]:
        """
        Run volume_profile.py for a specific symbol and extract avg_poc

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Average POC value or None if failed
        """
        try:
            self._log(f"Running volume profile analysis for {symbol}")

            # Construct command
            cmd = [
                'python3', self.volume_profile_script,
                '--symbol', symbol,
                '--days', '1',
                '--timeframe', '5Min',
                '--time-per-profile', 'DAY',
                '--chart'
            ]

            self._log(f"Command: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Run from project root
            )

            if result.returncode != 0:
                print(f"❌ Error running volume profile for {symbol}: {result.stderr}")
                return None

            # Look for the JSON output file - it may use the actual data date, not the target date
            # First try with target date
            json_filename = f"{symbol}_volume_profile_{self.date}.json"
            json_path = os.path.join(self.output_dir, json_filename)

            if not os.path.exists(json_path):
                # Try to find any JSON file for this symbol (may have different date)
                self._log(f"JSON file not found with target date: {json_path}")
                for filename in os.listdir(self.output_dir):
                    if filename.startswith(f"{symbol}_volume_profile_") and filename.endswith('.json'):
                        json_path = os.path.join(self.output_dir, filename)
                        self._log(f"Found alternative JSON file: {json_path}")
                        break
                else:
                    print(f"❌ No JSON output file found for {symbol}")
                    return None

            # Parse JSON and extract avg_poc
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            avg_poc = json_data.get('summary', {}).get('avg_poc')
            if avg_poc is not None:
                self._log(f"Extracted avg_poc for {symbol}: {avg_poc}")
                return float(avg_poc)
            else:
                print(f"❌ avg_poc not found in JSON for {symbol}")
                return None

        except Exception as e:
            print(f"❌ Exception processing {symbol}: {e}")
            return None

    def process_all_symbols(self, data: List[Dict]) -> List[Dict]:
        """
        Process all symbols and populate POC values

        Args:
            data: List of CSV rows

        Returns:
            Updated data with POC values filled
        """
        total_symbols = len(data)
        processed = 0

        for i, row in enumerate(data):
            symbol = row.get('Symbol', '').strip()
            if not symbol:
                continue

            print(f"📊 Processing {symbol} ({i+1}/{total_symbols})...")

            avg_poc = self.run_volume_profile_for_symbol(symbol)
            if avg_poc is not None:
                row['POC'] = f"{avg_poc:.2f}"
                processed += 1
                print(f"✅ {symbol}: POC = {avg_poc:.2f}")
            else:
                row['POC'] = 'N/A'
                print(f"⚠️  {symbol}: POC could not be calculated")

        print(f"\n📈 Processed {processed}/{total_symbols} symbols successfully")
        return data

    def save_results(self, data: List[Dict], fieldnames: List[str]):
        """
        Save results to output CSV file

        Args:
            data: List of CSV rows with POC data
            fieldnames: List of column names
        """
        output_filename = f"{self.date}_POC.csv"
        output_path = os.path.join(self.data_dir, output_filename)

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"💾 Results saved to: {output_path}")

    def run(self):
        """Run the bulk processing workflow"""
        print("🚀 Starting Volume Profile Bulk Processing")
        print(f"📅 Target date: {self.date}")

        # Find input CSV
        input_csv = self.find_input_csv()
        if not input_csv:
            print(f"❌ No CSV file found for date {self.date}")
            return 1

        print(f"📁 Input file: {input_csv}")

        # Read CSV data
        try:
            data = self.read_csv_data(input_csv)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return 1

        if not data:
            print("❌ No data found in CSV file")
            return 1

        # Get fieldnames from first row
        fieldnames = list(data[0].keys())

        # Process all symbols
        updated_data = self.process_all_symbols(data)

        # Save results
        try:
            self.save_results(updated_data, fieldnames)
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return 1

        print("🎉 Bulk processing completed successfully!")
        return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Volume Profile Bulk Processing')
    parser.add_argument('--date', type=str, help='Date in YYYYMMDD format (default: today)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Validate date format if provided
    if args.date:
        try:
            datetime.strptime(args.date, '%Y%m%d')
        except ValueError:
            print("❌ Invalid date format. Use YYYYMMDD format.")
            return 1

    processor = VolumeProfileBulkProcessor(date=args.date, debug=args.debug)
    return processor.run()


if __name__ == '__main__':
    sys.exit(main())