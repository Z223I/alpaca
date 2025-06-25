import os
import glob
import sys
from typing import List, Dict, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atoms.utils.read_csv import read_csv


class ORB:
    """
    Open Range Breakout (ORB) class for analyzing trading opportunities
    based on breakout patterns from opening range data.
    """
    
    def __init__(self):
        """Initialize the ORB class."""
        self.data_directory = 'data'
        self.csv_data: Optional[List[Dict[str, Any]]] = None
        self.current_file: Optional[str] = None
    
    def _get_most_recent_csv(self) -> Optional[str]:
        """
        Get the most recent CSV file from the data directory.
        
        Returns:
            Path to the most recent CSV file, or None if no CSV files found
        """
        try:
            csv_pattern = os.path.join(self.data_directory, '*.csv')
            csv_files = glob.glob(csv_pattern)
            
            if not csv_files:
                return None
                
            # Sort by modification time, most recent first
            csv_files.sort(key=os.path.getmtime, reverse=True)
            return csv_files[0]
            
        except Exception as e:
            print(f"Error finding CSV files: {e}")
            return None
    
    def _prompt_user_for_file(self, filename: str) -> bool:
        """
        Prompt user to confirm if they want to use the suggested file.
        
        Args:
            filename: Name of the file to confirm
            
        Returns:
            True if user confirms, False otherwise
        """
        try:
            response = input(f"Use file '{filename}'? (y/n): ").strip().lower()
            return response in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            return False
    
    def _load_and_process_csv_data(self) -> bool:
        """
        Load and process CSV data from the most recent file.
        
        This method:
        1. Finds the most current CSV file in the data directory
        2. Prompts user for confirmation to use that file
        3. Reads the file if confirmed, exits if not
        
        Returns:
            True if successful, False otherwise
        """
        print("ORB - Open Range Breakout Analysis")
        print("=" * 40)
        
        # Find the most recent CSV file
        most_recent_file = self._get_most_recent_csv()
        
        if not most_recent_file:
            print("No CSV files found in the data directory.")
            return False
        
        print(f"Most recent CSV file: {most_recent_file}")
        
        # Prompt user for confirmation
        if not self._prompt_user_for_file(most_recent_file):
            print("Operation cancelled by user.")
            return False
        
        # Read the CSV file
        try:
            print(f"Reading file: {most_recent_file}")
            self.csv_data = read_csv(most_recent_file)
            self.current_file = most_recent_file
            
            if not self.csv_data:
                print("Warning: CSV file is empty or contains no data.")
                return False
            
            print(f"Successfully loaded {len(self.csv_data)} rows of data.")
            
            # Display basic file information
            if self.csv_data:
                columns = list(self.csv_data[0].keys())
                print(f"Columns: {', '.join(columns)}")
                print(f"Sample data from first row: {self.csv_data[0]}")
            
            return True
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

    def Exec(self) -> bool:
        """
        Execute the ORB analysis process.
        
        Returns:
            True if successful, False otherwise
        """
        success = self._load_and_process_csv_data()

        return success


def main():
    """
    Main function to setup and run the ORB class.
    Handles keyboard interrupts gracefully.
    """
    try:
        # Create and run ORB analysis
        orb = ORB()
        success = orb.Exec()
        
        if success:
            print("\nORB analysis completed successfully.")
        else:
            print("\nORB analysis failed or was cancelled.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()