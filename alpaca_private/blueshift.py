"""
    Title: Buy and Hold (NYSE)
    Description: This is a long only strategy which rebalances the
        portfolio weights every month at month start.
    Style tags: Systematic
    Asset class: Equities, Futures, ETFs, Currencies and Commodities
    Dataset: NYSE Daily or NYSE Minute
"""

"""
    Always use the absolute path method of import.
    from library.technicals.indicators import bollinger_band
"""

# I added a builtin Sector classifier called ZiplineTraderSector.
# https://zipline-trader.readthedocs.io/_/downloads/en/latest/pdf/

"""
    For backtests, all trades are simulated with as-traded prices (unadjusted data) to accurately
    capture actual trading conditions. However, an API call for historical data will return
    adjusted data for ease of strategy development (like computing moving averages). Adjustments
    are applied on EOD basis.
"""

"""
    Strategy is designed to be fault-tolerant for corrupt input data
        At the base level, check if the data you have received is very much different from last datapoints
        your strategy had. Most exchanges usually follow a market volatility control mechanism for publicly
        traded securities. These limits stock price movements and also enforce cool-off periods in such
        circumstances. If your algo received some extreme data points, it is highly likely they are wrong.
        Even if they are true, the market volatility control mechanism probably has already been triggered.
        If your strategy is not particularly designed to exploit these situations, it is a good practice to
        pause any trading activity till saner data arrive.

    Strategy has necessary risk controls in place
        This is, again, is an absolute must. At the minimum, it should control the max number of orders
        it can send (to control machine gunning), the max size of each order (machines do fat fingers too)
        and a kill switch (a percentage loss below which it should stop automatically). Blueshift® has all
        these features, and then some more. You can put controls based on the maximum position size, maximum
        leverage or even declare a white-list (black-list) of assets that the algo can (cannot) trade.

    A strategy trading close to the account capacity
        Margin calls can put an automated strategy out of gear. Always ensure the account is funded adequately,
        so that the algo runs in an expected way.

"""

# Zipline
from zipline.api import(    symbol,
                            order_target,
                            order_target_percent,
                            schedule_function,
                            date_rules,
                            time_rules,
                       )

"""
import pytz
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as yahoo_reader
from zipline.utils.calendars import get_calendar
from zipline.data import bundles
#from zipline import run_algorithm
"""


def initialize(context):
    """
        A function to define things to do at the start of the strategy
    """

    # universe selection
    # This will be overwritten daily.
    context.AT_portfolio = [
                               symbol('AMZN'),
                               symbol('AAPL'),
                               symbol('WMT'),
                               symbol('MU'),
                               symbol('BAC'),
                               symbol('KO'),
                               symbol('BA'),
                               symbol('AXP')
                             ]

    context.testStocks = [
                            symbol('AAPL'),
                            symbol('MSFT'),
                            symbol('AMZN'),
                            symbol('FB'),
                            symbol('GOOGL'),
                            symbol('GOOG')
                            ]

    context.SandP100 = [
                            symbol('AAPL'),
                            symbol('MSFT'),
                            symbol('AMZN'),
                            symbol('FB'),
                            symbol('GOOGL'),
                            symbol('GOOG'),
                            symbol('JPM'),
                            symbol('JNJ'),
                            symbol('V'),
                            symbol('UNH'),
                            symbol('HD'),
                            symbol('NVDA'),
                            symbol('PG'),
                            symbol('DIS'),
                            symbol('MA'),
                            symbol('BAC'),
                            symbol('PYPL'),
                            symbol('XOM'),
                            symbol('CMCSA'),
                            symbol('VZ'),
                            symbol('INTC'),
                            symbol('ADBE'),
                            symbol('T'),
                            symbol('CSCO'),
                            symbol('NFLX'),
                            symbol('PFE'),
                            symbol('KO'),
                            symbol('ABT'),
                            symbol('CVX'),
                            symbol('ABBV'),
                            symbol('PEP'),
                            symbol('CRM'),
                            symbol('MRK'),
                            symbol('WMT'),
                            symbol('WFC'),
                            symbol('TMO'),
                            symbol('ACN'),
                            symbol('AVGO'),
                            symbol('MCD'),
                            symbol('MDT'),
                            symbol('NKE'),
                            symbol('TXN'),
                            symbol('COST'),
                            symbol('DHR'),
                            symbol('HON'),
                            symbol('C'),
                            symbol('QCOM'),
                            symbol('UPS'),
                            symbol('LLY'),
                            symbol('UNP'),
                            symbol('PM'),
                            symbol('LOW'),
                            symbol('ORCL'),
                            symbol('AMGN'),
                            symbol('NEE'),
                            symbol('BMY'),
                            symbol('SBUX'),
                            symbol('IBM'),
                            symbol('MS'),
                            symbol('CAT'),
                            symbol('BA'),
                            symbol('GS'),
                            symbol('BLK'),
                            symbol('DE'),
                            symbol('AMAT'),
                            symbol('MMM'),
                            symbol('GE'),
                            symbol('CVS'),
                            symbol('AMT'),
                            symbol('INTU'),
                            symbol('SCHW'),
                            symbol('TGT'),
                            symbol('AXP'),
                            symbol('ISRG'),
                            symbol('CHTR'),
                            symbol('LMT'),
                            symbol('ANTM'),
                            symbol('MU'),
                            symbol('FIS'),
                            symbol('AMD'),
                            symbol('SPGI'),
                            symbol('BKNG'),
                            symbol('MO'),
                            symbol('CI'),
                            symbol('LRCX'),
                            symbol('MDLZ'),
                            symbol('TJX'),
                            symbol('PLD'),
                            symbol('PNC'),
                            symbol('USB'),
                            symbol('GILD'),
                            symbol('ADP'),
                            symbol('SYK')
                            ]

    # Data is for predictions from 5/5 to 5/7.  Need NYSE.
    context.buyListOfDicts = [
        {'AAPL': 105.0, 'MSFT': 16.57, 'NVDA': 19.53, 'INTC': 32.71, 'ADBE': 10.92, 'CSCO': 56.94},
        {'AAPL': 105.0, 'AVGO': 16.57, 'TXN': 19.53, 'QCOM': 32.71, 'INTU': 10.92},
        {'AAPL': 105.0, 'AVGO': 16.57, 'TXN': 19.53, 'QCOM': 32.71, 'INTU': 10.92},
    ]



    context.lookback = 40  # Look back 62 days
    context.history_range = 400  # Only consider the past 400 days' history

    context.tradeDayCounter = 0

    # Call rebalance function at market open
    # Trade at the start of every day
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes=1))

def createPortfolio(context):

    index = context.tradeDayCounter

    context.AT_portfolio = []

    for key in context.buyListOfDicts[index].keys():
        keyStr = str(key)
        context.AT_portfolio.append( symbol(keyStr) )

    if index < len(context.buyListOfDicts) - 1:
        context.tradeDayCounter += 1


def rebalance01(context, data):
    """
        A function to rebalance the portfolio, passed on to the call
        of schedule_function above.
    """

    createPortfolio(context)

    # Should use a percentage based on number of stocks.  I have seen 35 stocks at the top.
    for security in context.AT_portfolio:
        order_target_percent(security, 1.0 / float(len(context.AT_portfolio)))

"""
def handle_data(context, data):
    px1 = data.history(context.universe[0], "close", 10, "1m")
    px2 = data.history(context.universe, "close", 10, "1m")
    px3 = data.history(context.universe[0], ["open","close"], 10, "1m")
    px4 = data.history(context.universe, ["open","close"], 10, "1m")

px1: Pandas Series with date-time as index
px2: Pandas DataFrame with date-time index and assets as columns
px3: Pandas DataFrame with date-time index and price fields as columns
px4: Pandas Panel data2 in the current version. Pandas Multi-indexed dataframe in future version.

"""


def rebalance(context, data):
    """
        A function to rebalance the portfolio, passed on to the call
        of schedule_function above.
    """

    """
    # get the pending open orders
    open_orders = get_open_orders()
    # cancel all open orders
    for order_id in open_orders:
        cancel_order(order_id)

    asset = symbol("AAPL")
    order_id = order(asset, 10)   # a market order for 10 stocks of Apple Inc.
    order_id = order(asset, 10, 208.8)   # a limit order at 208.8 or better.
    """

    # Should use a percentage based on number of stocks.  I have seen 35 stocks at the top.
    for security in context.testStocks:
        order_target_percent(security, 1.0 / float(len(context.testStocks)))

    """
        WARNING:
        Always check if the return value is None or a valid order id. We return None for the following cases

        The order was invalid and was not placed with the broker
        The order type was market and resulted in unwinding of existing position(s). This is the
        case where the broker does not support fungible orders1. In such cases, only an order for
        the residual amount will be created (if any) and order ID of the same will be returned.
    """

#def before_trading_start(context, data):
    """
        This is the function that is called at a preset time before the market opens, every day.
        Your strategy should not depend on the exact time of this function call.
    """

#def after_market_hours(context, data):
    """
        Called once everyday, after the market closes. Your strategy should not depend on
        the exact time of this function call.
    """

#def analyze(context, performance):
    """
        This is a function called at the end of a strategy run.
    """


"""
https://github.com/drgee1000/stock_trading/blob/master/tests/test_TA.py
"""