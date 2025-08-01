import os
import re
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from atoms.utils.read_csv import read_csv


def build_symbol_list(data_directory: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build an accumulated symbol list from all YYYYMMDD.csv files in the data directory.

    This atom combines all CSV files with date-based names, eliminates duplicate symbols,
    and preserves full data from the most recent file while zeroing out fields from 
    older files (except Symbol column).

    Args:
        data_directory: Path to directory containing YYYYMMDD.csv files
        output_file: Optional path to write the accumulated symbol list CSV

    Returns:
        List of dictionaries containing accumulated symbol data with most recent
        data preserved and older data zeroed out (except Symbol column)

    Raises:
        FileNotFoundError: If data_directory doesn't exist
        ValueError: If no valid YYYYMMDD.csv files found
    """
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory not found: {data_directory}")

    # Find all YYYYMMDD.csv files
    csv_files = []
    date_pattern = re.compile(r'^(\d{8})\.csv$')

    for filename in os.listdir(data_directory):
        match = date_pattern.match(filename)
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d')
                csv_files.append({
                    'filename': filename,
                    'date': file_date,
                    'date_str': date_str
                })
            except ValueError:
                # Skip files with invalid date formats
                continue

    if not csv_files:
        raise ValueError(f"No valid YYYYMMDD.csv files found in {data_directory}")

    # Sort by date (most recent first)
    csv_files.sort(key=lambda x: x['date'], reverse=True)
    most_recent_file = csv_files[0]
    older_files = csv_files[1:]

    # Read the most recent file - preserve all data
    most_recent_path = os.path.join(data_directory, most_recent_file['filename'])
    most_recent_data = read_csv(most_recent_path)

    if not most_recent_data:
        raise ValueError(f"Most recent file {most_recent_file['filename']} is empty")

    # Get column names from the most recent file
    sample_row = most_recent_data[0]
    all_columns = list(sample_row.keys())

    # Build symbol dictionary with most recent data
    symbol_data = {}

    # Add most recent data (preserving all fields)
    for row in most_recent_data:
        symbol = row.get('Symbol')
        if symbol:
            symbol_data[symbol] = dict(row)

    # Process older files - zero out all fields except Symbol
    for file_info in older_files:
        file_path = os.path.join(data_directory, file_info['filename'])
        try:
            older_data = read_csv(file_path)

            for row in older_data:
                symbol = row.get('Symbol')
                if symbol and symbol not in symbol_data:
                    # Create zeroed row for this symbol
                    zeroed_row = {'Symbol': symbol}
                    for col in all_columns:
                        if col != 'Symbol':
                            zeroed_row[col] = 0
                    symbol_data[symbol] = zeroed_row

        except Exception as e:
            # Log but continue processing other files
            print(f"Warning: Could not process {file_info['filename']}: {e}")
            continue

    # Convert back to list and sort by symbol for consistency
    accumulated_data = list(symbol_data.values())
    accumulated_data.sort(key=lambda x: x.get('Symbol', ''))

    # Write to output file if specified
    if output_file and accumulated_data:
        _write_accumulated_csv(accumulated_data, output_file, all_columns)

    return accumulated_data


def _write_accumulated_csv(data: List[Dict[str, Any]], output_file: str, columns: List[str]) -> None:
    """
    Write accumulated symbol data to a CSV file.

    Args:
        data: List of symbol dictionaries to write
        output_file: Path to output CSV file
        columns: List of column names in desired order
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
    except Exception as e:
        raise IOError(f"Error writing to {output_file}: {e}")


def build_daily_accumulated_list(data_directory: str, accumulated_file: str) -> List[Dict[str, Any]]:
    """
    Build or update a daily accumulated symbol list file.

    This function is designed to be run daily. It either creates a new accumulated
    file or updates an existing one by adding new symbols from the most recent
    data file.

    Args:
        data_directory: Path to directory containing YYYYMMDD.csv files
        accumulated_file: Path to the accumulated symbols file to create/update

    Returns:
        List of dictionaries containing the updated accumulated symbol data
    """
    # Build the full symbol list
    accumulated_data = build_symbol_list(data_directory)

    # Write to the accumulated file
    if accumulated_data:
        sample_row = accumulated_data[0]
        columns = list(sample_row.keys())
        _write_accumulated_csv(accumulated_data, accumulated_file, columns)

    return accumulated_data