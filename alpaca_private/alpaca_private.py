#import requests
import json
import math
import time
import sys

import alpaca_trade_api as tradeapi   # pip3 install alpaca-trade-api -U

import parseArgs

## Enter description
class Portfolio:

    ##
    def __init__(self, assets=dict()):
        self.assets = assets

        if isinstance(self.assets, list):
            self.assets = {asset: None for asset in self.assets}

        print("assets:")
        print(str(self.assets))


    ## Returns two dicts the first contains stocks to sell the second contains stocks to buy
    def compare(self, other):
        sell  = self.assets.copy()
        buy = {
            asset: other.assets[asset]
            for asset in other.assets
            if sell.pop(asset, True)
        }

        return sell, buy


    def adjust(self, symbols, buy):
        for asset in symbols:
            if buy:
                self.assets[asset] = None
            else:
                del self.assets[asset]

## Enter description
class alpaca_private:


    ##
    def __init__(self):
        self.history = {}
        self.holdings = Portfolio()

        self.key = 'AKS04WXLXNUF9LX4XD1Y'
        self.secret = '3kCOo9dblurb9CMIyhArbHvFofnKWoumeZ4tTO8b'
        self.headers = {'APCA-API-KEY-ID':self.key, 'APCA-API-SECRET-KEY':self.secret}

        self.baseURL = 'https://api.alpaca.markets'
        self.accountURL = "{}/v2/account".format(self.baseURL)
        self.ordersURL = "{}/v2/orders".format(self.baseURL)

        self.core =  tradeapi.REST(self.key, self.secret, self.baseURL)
        self.active_orders = []
        self.current_id = 0


    ##
    def signal(self, signals):

        signals  = Portfolio(signals)

        sellStocksDict, buyStocksDict  = self.holdings.compare(signals)

        print("Sell:")
        print(str(sellStocksDict.keys()))

        self.action(sellStocksDict, False)
        self.delay()

        print("Buy:")
        print(str(buyStocksDict.keys()))

        self.action(buyStocksDict, True)

        self.holdings.adjust(sellStocksDict, False)
        self.holdings.adjust(buyStocksDict, True)

    def delay(self):
        while len(self.getActiveOrders()) > 0:
            time.sleep(1)


    ##
    def getActiveOrders(self):
        try:
            return self.core.list_orders(
                status='open',
                limit=100)
        except:
            return 1

    ##
    def getCash(self):
        return self.core.get_account().cash

    ##
    def getState(self):
        return self.core.list_positions()

    ##
    def printState(self):
        state = self.getState()
        print(f"state: {state}")


    ##
    def action(self, symbols, buy):
        if buy:
            direction = "buy"
            cash  = float(self.core.get_account().cash)
            print("cash")
            print(cash)
            if cash < 1:
                return
        else:
            direction = "sell"

        for asset in symbols:
            if buy:
                self.order(asset, (cash/len(symbols))/symbols[asset], direction)
            else:
                self.order(asset, self.core.get_position(asset).qty, direction)

            self.active_orders.append(self.current_id)
            self.current_id += 1

        print("Trade")
        self.printState()


    ##
    def order(self, symbol, quat, side, type="market", time_in_force="gtc"):

        self.core.submit_order(
            symbol=symbol,
            qty=math.floor(quat),
            side=side,
            type=type,
            time_in_force=time_in_force )



        #orderObject = {
        #"symbol":symbol,
        #"notional":quat,
        #"side":side,
        #"type":"market",
        #"time_in_force":"gtc"
        #}
        #return json.loads(requests.post(self.ordersURL, headers=self.headers, json=orderObject).content)


    ## @brief Main code of the alpaca_private object.
    def main( self, userArgs=None ):

        self.printState()
        print(self.getActiveOrders())

        return 0


## Parse CLI arguments.
def getArgs(userArgs=None):
    # sourcery skip: inline-immediately-returned-variable

    print(f'userArgs: {userArgs}')

    parser = parseArgs.setupParser()

    args = parseArgs.parseArgs( parser, userArgs )

    return args

"""
api = alpaca_private()
#api.printState()
#api.signal({"AAPL":134, "NKE":131})
#api.signal({"AAPL":134, "NKE":131})
#api.signal({"AAPL":134, "GME":170})
#api.signal({"AMC":11.5, "GME":170})
#api.signal({"AAPL":134, "NKE":131})
#api.order("AAPL", 1, "sell")

#api.order("AAPL", 390, "sell")
#api.order("NKE", 380, "sell")
api.printState()
print(api.getActiveOrders())
api.delay()
#print(str(api.getCash()))
"""

"""
api = alpaca_private()
"""



## @brief Main exec of the file.
def execMain( userArgs=None ):
    # sourcery skip: inline-immediately-returned-variable

    alpacaObj = alpaca_private()

    exitValue = alpacaObj.main()

    return exitValue

if __name__ == '__main__':
   try:
      retVal = execMain()
   except KeyboardInterrupt:
      print('Received <Ctrl+c>')
      sys.exit(-1)

   sys.exit(retVal)

"""
python3 -m pdb code/alpaca_private.py
"""


"""
Can use <F5> or <Ctrl+F5> by doing the following:
dlvenv
"""