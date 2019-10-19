#!/usr/bin/python
import pandas as pd
import requests
import sys


currency1 = 'USD'
currency2 = 'JPY'
date_from = '2018-01-01'
date_to = '2018-01-30'

def fetch_currency(currency,beg,end):
    url = 'http://api.nbp.pl/api/exchangerates/rates/A/' + currency + "/" + date_from + "/" + date_to + "/"
    currency_req = requests.get(url)
    currency_data = currency_req.json()
    return currency_data['rates']

rate1 = fetch_currency(currency1,date_from,date_to)
rate2 = fetch_currency(currency2,date_from,date_to)

print(rate1)
print(rate2)