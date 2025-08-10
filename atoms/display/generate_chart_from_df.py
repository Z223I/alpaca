"""
Atom for generating candlestick charts with MACD directly from a DataFrame.
"""

import os
import sys
import pandas as pd
import pytz
from typing import Optional, List
from datetime import datetime

# Handle imports for both standalone and module usage
try:
    from .plot_candle_chart import plot_candle_chart
except ImportError:
    # Add parent directory to path for standalone usage
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from atoms.display.plot_candle_chart import plot_candle_chart


class ChartFromDataFrame:
    """
    Generate candlestick charts with MACD from DataFrame data.
    """

    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None, output_dir: str = 'chart_output', alerts: Optional[List] = None):
        """
        Initialize the chart generator with DataFrame data.

        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            symbol: Stock symbol name (will be extracted from df if not provided)
            output_dir: Directory to save the chart (default: 'chart_output')
            alerts: Optional list of alert dictionaries
        """
        self.df = df.copy()
        self.symbol = symbol
        self.output_dir = output_dir
        self.alerts = alerts or []
        
        # Validate DataFrame
        if not self._validate_dataframe():
            raise ValueError("Invalid DataFrame structure")
        
        # Extract symbol if not provided
        if not self.symbol:
            self.symbol = self._extract_symbol()
        
        # Prepare DataFrame for charting
        self._prepare_dataframe()

    def _validate_dataframe(self) -> bool:
        """
        Validate that DataFrame has required columns and data.

        Returns:
            True if valid, False otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if self.df is None or self.df.empty:
            print("❌ DataFrame is empty or None")
            return False
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        if len(self.df) < 26:
            print(f"❌ Insufficient data for MACD calculation: {len(self.df)} < 26 periods")
            return False
        
        return True

    def _extract_symbol(self) -> str:
        """
        Extract symbol from DataFrame or use default.

        Returns:
            Symbol string
        """
        if 'symbol' in self.df.columns and not self.df['symbol'].isna().all():
            return str(self.df['symbol'].iloc[0])
        else:
            return 'UNKNOWN'

    def _prepare_dataframe(self) -> None:
        """
        Prepare DataFrame for charting by ensuring proper timestamp format.
        """
        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Add timezone information if not present
        if self.df['timestamp'].dt.tz is None:
            et_tz = pytz.timezone('America/New_York')
            self.df['timestamp'] = self.df['timestamp'].dt.tz_localize(et_tz)
        
        # Sort by timestamp to ensure proper order
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

    def generate_chart(self, verbose: bool = True) -> bool:
        """
        Generate the candlestick chart with MACD.

        Args:
            verbose: If True, print detailed information

        Returns:
            True if successful, False otherwise
        """
        if verbose:
            self._print_data_summary()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # Generate the chart using the existing plot_candle_chart function
            success = plot_candle_chart(self.df, self.symbol, self.output_dir, self.alerts)
            
            if success:
                # Determine the chart path
                chart_date = self.df['timestamp'].iloc[0].date()
                date_subdir = chart_date.strftime('%Y%m%d')
                chart_path = os.path.join(self.output_dir, date_subdir, f"{self.symbol}_candle_chart.png")
                
                if verbose:
                    if os.path.exists(chart_path):
                        print(f"✅ SUCCESS: Chart with MACD generated at {chart_path}")
                        self._print_chart_features()
                    else:
                        print(f"❌ Chart file not created at expected path: {chart_path}")
                        return False
                
                return True
            else:
                if verbose:
                    print("❌ Chart generation failed")
                return False
                
        except Exception as e:
            if verbose:
                print(f"❌ Error generating chart: {e}")
            return False

    def _print_data_summary(self) -> None:
        """Print summary of the data being used."""
        print(f"Generating MACD chart for {self.symbol}...")
        print(f"Data summary:")
        print(f"  🕒 Time period: {len(self.df)} minutes")
        print(f"  📅 Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"  💰 Price range: ${self.df['low'].min():.2f} - ${self.df['high'].max():.2f}")
        print(f"  📦 Volume range: {self.df['volume'].min():,} - {self.df['volume'].max():,}")
        print(f"  📦 Total volume: {self.df['volume'].sum():,} shares")
        print(f"✅ Sufficient data for MACD calculation ({len(self.df)} periods)")

    def _print_chart_features(self) -> None:
        """Print information about chart features."""
        print(f"\nChart features:")
        print(f"  📈 Price chart: Candlesticks, ORB levels, EMA(9), EMA(20), VWAP")
        print(f"  📊 MACD chart: MACD line, Signal line, Histogram (12,26,9)")
        print(f"  📊 Volume chart: Volume bars")

    def get_chart_path(self) -> str:
        """
        Get the expected path of the generated chart.

        Returns:
            Path to the chart file
        """
        chart_date = self.df['timestamp'].iloc[0].date()
        date_subdir = chart_date.strftime('%Y%m%d')
        return os.path.join(self.output_dir, date_subdir, f"{self.symbol}_candle_chart.png")


# Convenience function for quick chart generation
def generate_chart_from_dataframe(df: pd.DataFrame, symbol: Optional[str] = None, 
                                 output_dir: str = 'chart_output', alerts: Optional[List] = None,
                                 verbose: bool = True) -> bool:
    """
    Convenience function to generate a chart from a DataFrame.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol (optional, will be extracted from df)
        output_dir: Output directory for chart
        alerts: Optional list of alerts
        verbose: Print detailed information

    Returns:
        True if successful, False otherwise
    """
    try:
        chart_generator = ChartFromDataFrame(df, symbol, output_dir, alerts)
        return chart_generator.generate_chart(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"❌ Error: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    def test_with_verb_data():
        """Test the atom with VERB historical data."""
        print("Testing ChartFromDataFrame atom with VERB data...")
        
        # Path to the VERB data file (adjust based on current working directory)
        possible_paths = [
            "../../historical_data/2025-08-04/market_data/VERB_20250804_160655.csv",  # From atoms/display
            "historical_data/2025-08-04/market_data/VERB_20250804_160655.csv",       # From root
            "./historical_data/2025-08-04/market_data/VERB_20250804_160655.csv"      # From root with ./
        ]
        
        data_file = None
        for path in possible_paths:
            if os.path.exists(path):
                data_file = path
                break
        
        if data_file is None:
            print(f"❌ Test data file not found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            print("Please ensure the VERB data file exists")
            return False
        
        try:
            # Read the data
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} rows from {data_file}")
            
            # Test using the class
            print("\n" + "="*60)
            print("Testing ChartFromDataFrame class:")
            print("="*60)
            
            chart_gen = ChartFromDataFrame(df, output_dir='test_verb_output')
            success = chart_gen.generate_chart()
            
            if success:
                print(f"\n📊 Chart saved to: {chart_gen.get_chart_path()}")
            
            # Test using the convenience function (keep original symbol)
            print("\n" + "="*60)
            print("Testing convenience function:")
            print("="*60)
            
            success2 = generate_chart_from_dataframe(df, symbol='VERB', 
                                                   output_dir='test_verb_convenience')
            
            return success and success2
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    # Run the test
    test_success = test_with_verb_data()
    
    if test_success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed")
        sys.exit(1)