#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

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

usd_rates_january = pd.DataFrame.from_dict(rate1)
usd_rates_january.head()

jpy_rates_january = pd.DataFrame.from_dict(rate2)
jpy_rates_january.head()

plot_data1 = usd_rates_january.set_index(['effectiveDate'])['mid']
plot_data2 = jpy_rates_january.set_index(['effectiveDate'])['mid']

plt.plot(plot_data1)
plt.show()

plt.plot(plot_data2)
plt.show()


print(rate1)
print(rate2)

correlation = np.corrcoef (plot_data1, plot_data2)[0][1]


print("{} {}".format("Corelation: ", correlation))