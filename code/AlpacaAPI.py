import requests
import json
import math
import time 
import alpaca_trade_api as tradeapi 

class Portfolio:
    def __init__(self,assets=dict()):
        self.assets = assets
        if isinstance(self.assets,list):
            temp = dict()
            for asset in self.assets:
                temp[asset] = None
            self.assets = temp 
        print("assets:")
        print(str(self.assets))
    #returns two dicts the first contains stocks to sell the second contains stocks to buy
    def compare(self,other):
        sell  = self.assets.copy()
        buy  = dict()
        for asset in other.assets:
            if sell.pop(asset,True):
                buy[asset] = other.assets[asset]
        return sell, buy
    def adjust(self,symbols,buy):
        for asset in symbols:
            if not buy:
                del self.assets[asset]
            else:
                self.assets[asset] = None

class AlpacaAPI:
    def __init__(self):
        self.history = dict()
        self.holdings = Portfolio()
        self.key = "PKE2MB78HZHHEGSWYM36"
        self.secret = "ihWtpSr1hzTK4lAWyEBZAPQF9841aiATY8zWEQUy"
        self.headers = {'APCA-API-KEY-ID':self.key, 'APCA-API-SECRET-KEY':self.secret}
        self.baseURL = "https://paper-api.alpaca.markets"
        self.accountURL = "{}/v2/account".format(self.baseURL)
        self.ordersURL = "{}/v2/orders".format(self.baseURL)
        self.core =  tradeapi.REST(self.key,self.secret,self.baseURL)
        self.active_orders = []
        self.current_id = 0

    def signal(self,signals):
        signals  = Portfolio(signals)
        sell, buy  = self.holdings.compare(signals)
        print("sell:")
        print(str(sell.keys()))
        self.action(sell,False)
        self.delay()
        print("buy:")
        print(str(buy.keys()))
        self.action(buy,True)
        self.holdings.adjust(sell,False)
        self.holdings.adjust(buy,True)

    def delay(self):
        while len(self.getActiveOrders()) > 0:
            time.sleep(1)

    def getActiveOrders(self):
        try:
            return self.core.list_orders(
                status='open',
                limit=100)
        except:
            return 1
    
    def getCash(self):
        return self.core.get_account().cash

    def printState(self):
        print(str(self.core.list_positions()))

    def action(self,symbols,buy):
        direction = "buy"
        if not buy:
            direction="sell"
        else:
            cash  = float(self.core.get_account().cash)
            print("cash")
            print(cash)
            if cash<1:
                return
        for asset in symbols:
            if not buy:
                self.order(asset,self.core.get_position(asset).qty,direction)
            else:
                self.order(asset,(cash/len(symbols))/symbols[asset],direction)
            self.active_orders.append(self.current_id)
            self.current_id +=1
        #print("Trade")
        self.printState()

    def order(self,symbol,quat,side): 
        self.core.submit_order(    
            symbol=symbol,
            qty=math.floor(quat),
            side=side,
            type='market',
            time_in_force='gtc' )
        #orderObject = {
        #"symbol":symbol,
        #"notional":quat,
        #"side":side,
        #"type":"market",
        #"time_in_force":"gtc"
        #}
        #return json.loads(requests.post(self.ordersURL,headers=self.headers,json=orderObject).content)

"""
api = AlpacaAPI()
#api.printState()
#api.signal({"AAPL":134,"NKE":131})
#api.signal({"AAPL":134,"NKE":131})
#api.signal({"AAPL":134,"GME":170})
#api.signal({"AMC":11.5,"GME":170})
#api.signal({"AAPL":134,"NKE":131})
#api.order("AAPL",1,"sell")

#api.order("AAPL",390,"sell")
#api.order("NKE",380,"sell")
api.printState()
print(api.getActiveOrders())
api.delay()
#print(str(api.getCash()))
"""