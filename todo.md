# TODO
- [ ] Focus on creating ORB filter.
- [X] ORB Create VWAP atom.
- [X] ORB Create EMA with 9 as the default parameter atom.
- [ ] ORB Create atom to calculate a momentum vector from ORB candlesticks.  Hmm. Is this two values? Yes.  Fit a line and use the angle.
- [X] ORB Method extract as an atom the symbol_data calculation in atoms/display/plot_candle_chart.py
- [X] ORB Create ORB.py to monitor ORB trading strategy.
- [X] ORB Create ORB._get_orb_market_data()
- [X] ORB Calculate the ORB for each stock in the first 15 minutes.
- [ ] Add float rotation calculator.  
- [X] Create .envpaper
- [ ] Create .envlive
- [X] Create '_future_bracket_order()'
- [ ] Update to do trailing stop. Create new method UStrailingStop()
- [ ] Check the return values of orders.
- [X] Create underscore buy.  New cli argument for this.
- [ ] Update print_active_orders to do all the prints.


## Bash

        ```bash
        python code/alpaca.py --bracket_order --symbol AAPL --quantity 10 --market_price 150.00
        python code/alpaca.py --get_latest_quote --symbol NINE
        python code/alpaca.py --buy --symbol NINE --take_profit 1.45
        python code/alpaca.py --buy --symbol AAPL --take_profit 210.00 --submit

        # quantity will be automatically calculated.
        python3 code/alpaca.py --future_bracket_order --symbol AAPL --limit_price 145.00 --stop_price 140.00 --take_profit 160.00 --submit
        ```

## MCP
        ```bash
        claude mcp add --transport http context7 https://mcp.context7.com/mcp
        ```



## PCA
Use the ORB class in orb.py.
Create a private method for PCA analysis.
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

Stack the data in each candlestick
Independent Variables
Use ‘close’ to calculate ratios
Resistance ratio
Gold support ratio
Support ratio
VWAP ratio
EMA 9 ratio
ORB low ratio
ORB high ratio
bool met ORB
bool met VWAP
bool met gold resistance
volume/float
Use ORB candlesticks to calculate a vector (momentum indicator)

TBD
Moving volume
, bool not too far above VWAP, bool above EMA 9, bool below next higher resistance line, use only 5 or 10 candlestick for AI training , not bool but ratios, moving volume ratio, where is the price within resistance lines, ratio to gold resistance, is there a momentum indicator? Yes, the change in VWAP and the candle color.

Dependent Variable
Price 5, 10 candlesticks ahead
Methodology
Build upon orb.py
Create a private method for PCA preparation



For each of the first 30 lines:
Get the close price;


Create lksdfjlsdkfj
And call the new method on a per stock basis.
Mirror lksdfjlsdkjf
Accumulate the PCA prep data to an object variable for later use
