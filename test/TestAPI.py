#!/usr/bin/env python
# coding: utf-8

import requests
import json



#paper trade 
key = "PK55VLCPE1HGTITEXNRA"
secret = "7Td8XyMXFVKB46szhHwPzI8ov9w7zeTK5ctcSMMx"
baseURL = "https://paper-api.alpaca.markets"
accountURL = "{}/v2/account".format(baseURL)
ordersURL = "{}/v2/orders".format(baseURL)

headers = {'APCA-API-KEY-ID':key, 'APCA-API-SECRET-KEY':secret}
requests.get(accountURL, headers=headers).content


def order(symbol,quat,side,type,time):
    orderObject = {
        "symbol":symbol,
        "qty":quat,
        "side":side,
        "type":type,
        "time_in_force":time
    }
    
    return json.loads(requests.post(ordersURL,headers=headers,json=orderObject).content)

order("AAPL",1,"buy","market","gtc")





