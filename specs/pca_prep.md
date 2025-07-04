# PCA Data Prep

## High Level Requirements

You are a highly skilled Python developer and architect.
You are working in an atom - molecules architecture.

## Mid Level Requirements

Use the ORB class in orb.py.
Create a private method for PCA analysis data prep.
It is to conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

We will create another method later to call this new method.

### Step 1

Mirror
```python
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
        # Extract symbol data using the dedicated atom
        symbol_data = extract_symbol_data(df, symbol)
        
        if symbol_data is None:
            print(f"No data found for symbol: {symbol}")
            return False
        
        # Calculate ORB levels
        orb_high, orb_low = calculate_orb_levels(symbol_data)
        
        # Calculate EMA (9-period) for close prices
        ema_success, ema_values = calculate_ema(symbol_data, price_column='close', period=9)
        
        # Calculate VWAP using typical price (HLC/3)
        vwap_success, vwap_values = calculate_vwap_typical(symbol_data)
```

Use df and symbol as parameters.
Use all the code from the try block.

### Step 3

Update the new method: Right after retrieving symbol_data:
Filter the data from 9:30 to 10:15 ET;
Verify there are 45 lines of data and return if not.

### Step 4

Use calculate_vector_angle method from atoms/utils/calculate_vector_angle.py to calculate vector angle.

### Step 5

Create a dataframe as a class variable.
Use all the data already collected.
Most of the data will have 45 lines.  However, the vector angle is only one value; repeat it for all the lines.
Put all this data into the class variable.

