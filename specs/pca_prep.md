## PCA

Use the ORB class in orb.py.
Create a private method for PCA analysis prep.





For each stock:
Mirror
def plot_candle_chart(df: pd.DataFrame, symbol: str, output_dir: str = 'plots') -> bool:
   """
   Create a candlestick chart with volume and ORB rectangle for a single stock symbol.
  
   Args:
       df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
       symbol: Stock symbol name
       output_dir: Directory to save the chart
      
   Returns:
       True if successful, False otherwise
   """
   try:
       # Filter data for the specific symbol
       symbol_data = df[df['symbol'] == symbol].copy()
      
       if symbol_data.empty:
           print(f"No data found for symbol: {symbol}")
           return False
      
       # Sort by timestamp
       symbol_data = symbol_data.sort_values('timestamp')
      
       # Calculate ORB levels
       orb_high, orb_low = calculate_orb_levels(symbol_data)
Use df and symbol as parameters.
Call calculate_orb_levels
Filter the data from 9:30 to 10:15 ET
Verify there are 45 lines of data and return if not.





Create a private method for PCA analysis.
