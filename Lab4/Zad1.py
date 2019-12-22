from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

samochody = fetch_openml('cars1')
print("Klucze: ", samochody.keys())
print(samochody ['feature_names'])
print (samochody ['feature_names'][3])

# data[:, [3]] - horsepower
# data[:, [5]] - time-to-sixty
x = samochody.data[:, [1, 3]]
y = samochody['target']
y = [int(elem) for elem in y]

# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Tworzymy klasyfikator z trzema klastrami (klasami)
kmn = KMeans (n_clusters=7)
kmn.fit(x_train)

# Wyciągamy punkty centralne klastrów - pokażemy je na wykresie obok punktów ze zbioru uczącego
centra = kmn.cluster_centers_
fig, ax = plt.subplots(1, 2)

# Pierwszy wykres to nasz zbiór uczący, z prawdziwymi klasami
ax[0].scatter (x_train[:, 0], x_train[:, 1], c=y_train, s=20)

# Teraz używamy danych treningowych żeby sprawdzić co klasyfikator o nich myśli
y_pred_train = kmn.predict(x_train)
ax[1].scatter (x_train[:, 0], x_train [:, 1], c=y_pred_train, s=20)

# Dokładamy na drugim wykresie centra klastrów
ax[1].scatter(centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()

# Próbujemy przewidzieć gatunki dla zbioru testowego
y_pred = kmn.predict(x_test)

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, s=20)

#Nowe gatunki przewidziane przez klastrowanie
plt.scatter (centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()

# Zadanie 2
# Opisz własnymi słowami, jakie klasy samochodów wg Ciebie znalazły się w zbiorze
# W zbiorze znalazły się następujące klasy samochodów, posiadające:
# - 1 cylinder oraz 1,24s do 60-tki
# - 3 cylindry oraz 1,28s do 60-tki
# - 4 cylindry oraz 2,26s do 60-tki