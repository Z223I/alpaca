import csv
from typing import List, Dict, Any


def read_csv(filename: str, delimiter: str = ',', encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Read a CSV file and return its contents as a list of dictionaries.

    Args:
        filename: Path to the CSV file to read
        delimiter: Field delimiter (default: ',')
        encoding: File encoding (default: 'utf-8')

    Returns:
        List of dictionaries where each dictionary represents a row
        with column headers as keys

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        csv.Error: If there's an error parsing the CSV file
        UnicodeDecodeError: If the file encoding is incorrect
    """
    try:
        with open(filename, 'r', encoding=encoding, newline='') as csvfile:
            # Use Sniffer to detect delimiter if not specified
            if delimiter == ',':
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample)
                    delimiter = dialect.delimiter
                except csv.Error:
                    # Fall back to comma if detection fails
                    delimiter = ','

            reader = csv.DictReader(csvfile, delimiter=delimiter)
            data = []

            for row in reader:
                # Convert numeric strings to appropriate types
                converted_row = {}
                for key, value in row.items():
                    if value is None or value == '':
                        converted_row[key] = None
                    else:
                        # Try to convert to float first, then int, otherwise keep as string
                        try:
                            if '.' in value:
                                converted_row[key] = float(value)
                            else:
                                converted_row[key] = int(value)
                        except ValueError:
                            converted_row[key] = value.strip()

                data.append(converted_row)

            return data

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {filename}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, f"Error decoding file {filename}: {e.reason}")
    except csv.Error as e:
        raise csv.Error(f"Error parsing CSV file {filename}: {e}")