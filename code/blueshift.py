"""
    Title: Buy and Hold (NYSE)
    Description: This is a long only strategy which rebalances the
        portfolio weights every month at month start.
    Style tags: Systematic
    Asset class: Equities, Futures, ETFs, Currencies and Commodities
    Dataset: NYSE Daily or NYSE Minute
"""
# Zipline
from zipline.api import(    symbol,
                            order_target_percent,
                            schedule_function,
                            date_rules,
                            time_rules,
                       )

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

    # Data is for predictions from 5/5 to 5/7.
    context.buyListOfDicts = [
        {'AKAM': 105.0, 'COMM': 16.57, 'DDD': 19.53, 'DXC': 32.71, 'GPRO': 10.92, 'INTC': 56.94, 'MSI': 186.57, 'ORCL': 78.54, 'SNPS': 237.07, 'SPWR': 23.0, 'VIAV': 15.99, 'WDAY': 237.15, 'STX': 88.47, 'PSTG': 18.89, 'BB': 8.36},
        {'AVYA': 29.57, 'CLDR': 11.86, 'COMM': 16.65, 'FTV': 71.74, 'GRPN': 51.11, 'HPE': 15.985, 'INTC': 56.9, 'LSCC': 48.67, 'OKTA': 242.28, 'SPWR': 23.34, 'SSNC': 73.05, 'VMW': 159.66, 'XRX': 23.98, 'TRMB': 81.14, 'IIVI': 65.15},
        {'ACN': 291.17, 'ADSK': 284.21, 'BB': 8.14, 'COMM': 17.21, 'CTSH': 75.11, 'DBX': 24.52, 'FEYE': 19.08, 'FIS': 151.81, 'FTNT': 206.37, 'GPRO': 10.45, 'HPQ': 34.99, 'INFY': 18.6, 'LSCC': 49.52, 'NOW': 485.84, 'NTAP': 77.9, 'OKTA': 234.35, 'SABR': 12.39, 'SPLK': 116.78, 'VIAV': 16.2, 'VMW': 162.36},
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


def rebalance(context, data):
    """
        A function to rebalance the portfolio, passed on to the call
        of schedule_function above.
    """

    createPortfolio(context)

    # Should use a percentage based on number of stocks.  I have seen 35 stocks at the top.
    for security in context.AT_portfolio:
        order_target_percent(security, 1.0 / float(len(context.AT_portfolio)))

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