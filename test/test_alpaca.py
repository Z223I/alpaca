# coding: utf-8

import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

import requests
import json

from alpaca_private import getArgs



## Test the alpaca_private object.
class TestAlpaca(unittest.TestCase):

   def setUp(self):

      #userArgs = None
      userArgs = ['-k', 'demo']
      args = getArgs(userArgs)

      print(f'userArgs: {userArgs}')

      self.test_alpaca = alpaca_private(args)

   def tearDown(self):



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





