# coding: utf-8

import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

import requests
import json

from alpaca_private import getArgs
from alpaca_private import alpaca_private



## Test the alpaca_private object.
class TestAlpaca(unittest.TestCase):

    def setUp(self):

        #userArgs = None
        #userArgs = ['-k', 'demo']
        #args = getArgs(userArgs)
        #print(f'userArgs: {userArgs}')

        #self.test_alpacaObj = alpaca_private(args)
        self.test_alpacaObj = alpaca_private()


        #paper trade
        key = "PK55VLCPE1HGTITEXNRA"
        secret = "7Td8XyMXFVKB46szhHwPzI8ov9w7zeTK5ctcSMMx"
        baseURL = "https://paper-api.alpaca.markets"
        self.accountURL = f"{baseURL}/v2/account"
        self.ordersURL = f"{baseURL}/v2/orders"

        self.headers = {'APCA-API-KEY-ID':key, 'APCA-API-SECRET-KEY':secret}
        #requests.get(self.accountURL, headers=self.headers).content

    def tearDown(self):
        pass


    ##
    def test_order(self):
        """
python test/test_Alpaca.py  TestAlpaca.test_order
        """

        symbol = 'AAPL'
        # Quat = quantity
        quat   = 1
        side   = "buy"

        self.test_alpacaObj.order(symbol, quat, side)
        state = self.test_alpacaObj.getState()
        print()
        print(state)

    """
    def order(symbol,quat,side,type,time):
        orderObject = {
            "symbol":symbol,
            "qty":quat,
            "side":side,
            "type":type,
            "time_in_force":time
        }

        return json.loads(requests.post(self.ordersURL,headers=self.headers,json=orderObject).content)
    """



"""
Can use <F5> or <Ctrl+F5> by doing the following:
dlvenv
export PYTHONPATH=$PYTHONPATH:/home/bwilson/DL/alpaca/code

If ran from alpaca directory, you can:
python -m unittest discover -s test

python -m pdb test/test_Alpaca.py  TestAlpaca.test_order
"""

if __name__ == "__main__":

    unittest.main()


