from typing import Any, List, Union
from datetime import datetime
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def get_stock_data(api_client: Any, symbols: Union[str, List[str]], timeframe: TimeFrame, start_date: datetime, end_date: datetime) -> Any:
    """
    Collect historical data for multiple stock symbols

    Args:
        api_client: Alpaca StockHistoricalDataClient instance
        symbols (Union[str, List[str]]): Stock symbol or list of stock symbols ['AAPL', 'GOOGL', 'MSFT']
        timeframe (TimeFrame): Data interval
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection

    Returns:
        dict: Market data organized by symbol
    """
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        limit=1000  # Max 1000 bars per request
    )

    bars = api_client.get_stock_bars(request_params)
    return bars