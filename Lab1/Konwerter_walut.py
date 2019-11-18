#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Dane wejściowe.
currency1 = 'USD'
currency2 = 'EUR'
date_from = '2018-01-01'
date_to = '2018-01-30'

# Funkcja wczytująca walutę z pliku .json, jako słownik.
def fetch_currency(currency,beg,end):
    url = 'http://api.nbp.pl/api/exchangerates/rates/A/' + currency + "/" + date_from + "/" + date_to + "/"
    currency_req = requests.get(url)
    currency_data = currency_req.json()
    return currency_data['rates']

# Użycie funkcji wczytującej walutę.
rate1 = fetch_currency(currency1,date_from,date_to)
rate2 = fetch_currency(currency2,date_from,date_to)

# Tworzenie obiektów DataFrame ze słowników i ograniczenie danych do pierwszych 10 wpisów.
rate_dataframe1 = pd.DataFrame.from_dict(rate1).head(10)
rate_dataframe2 = pd.DataFrame.from_dict(rate2).head(10)

#Indeksy ustawione na datę.
plot_data1 = rate_dataframe1.set_index(['effectiveDate'])['mid']
plot_data2 = rate_dataframe2.set_index(['effectiveDate'])['mid']

# Użycie funkcji obliczającej korelację dwóch kursów.
correlation = np.corrcoef (plot_data1, plot_data2)[0][1]

# Rysowanie wykresu pokazującego wyliczoną korelację oraz wartość obydwu kursów w porównaniu do PLN.
plt.plot(plot_data1, 'g--', plot_data2,'b--')
plt.ylim(ymin=0)
plt.title('Korelacja {} do {} = {}'.format(currency1, currency2, correlation))
plt.ylabel('Wartość w PLN')
plt.xlabel('Data')
plt.legend([currency1, currency2], loc='lower right')
plt.show()