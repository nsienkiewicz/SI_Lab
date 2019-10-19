#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

currency1 = 'USD'
currency2 = 'EUR'
date_from = '2018-01-01'
date_to = '2018-01-30'

def fetch_currency(currency,beg,end):
    url = 'http://api.nbp.pl/api/exchangerates/rates/A/' + currency + "/" + date_from + "/" + date_to + "/"
    currency_req = requests.get(url)
    currency_data = currency_req.json()
    return currency_data['rates']

rate1 = fetch_currency(currency1,date_from,date_to)
rate2 = fetch_currency(currency2,date_from,date_to)

usd_rates_january = pd.DataFrame.from_dict(rate1).head(10)
usd_rates_january.head()

jpy_rates_january = pd.DataFrame.from_dict(rate2).head(10)
jpy_rates_january.head()

plot_data1 = usd_rates_january.set_index(['effectiveDate'])['mid']
plot_data2 = jpy_rates_january.set_index(['effectiveDate'])['mid']

correlation = np.corrcoef (plot_data1, plot_data2)[0][1]

plt.plot(plot_data1, 'g--', plot_data2,'b--')
plt.ylim(ymin=0)
plt.title('Korelacja {} do {} = {}'.format(currency1, currency2, correlation))
plt.ylabel('Wartość w PLN')
plt.xlabel('Data')
plt.legend([currency1, currency2], loc='lower right')
plt.show()