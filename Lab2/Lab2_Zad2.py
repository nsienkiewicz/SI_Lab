from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# Ta sekcja jest tylko na potrzeby zobrazowania zbioru danych
# wykresy będą tworzone przy pomocy pakietu seaborn
# %matplotlib inline
# konwersja na obiekt pandas.DataFrame
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# funkcja która nam zamieni wartości 0, 1, 2 na pełny opis tekstowy dla gatunku
targets = map(lambda x: iris['target_names'][x], iris['target'] )

# doklejenie informacji o gatunku do reszty dataframe
iris_df['species'] = np.array(list(targets))

# wykres
# sns.pairplot(iris_df, hue='species')
# plt.show()
print(iris_df.head(10))

# Podzielmy zbiór na cechy oraz etykiety
# Konwencja, często spotykana w dokumentacji sklearn to X dla cech oraz y dla etykiet
X = iris.data
y = iris.target

# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Tworzymy klasyfikator k-NN używając parametru 5 sąsiadów
knn = KNeighborsClassifier(n_neighbors = 5)

# Uczymy klasyfikator na zbiorze - zaskoczenie - uczącym
knn.fit(X_train, y_train)

# Przewidujemy wartości dla zbioru testowego
y_pred = knn.predict(X_test)

# Sprawdzamy kilka pierwszych wartości przewidzianych
print(y_pred[:5])

# Sprawdzamy dokładność klasyfikatora
print(knn.score(X_test, y_test))

# Tworzymy płaszczyznę wszystkich możliwych wartości dla cechy 0 oraz 2, z krokiem 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

print("----")

# Uczymy klasyfikator na tylko dwóch wybranych cechach
knn.fit(X_train[:, [0, 2]], y_train)

# Przewidujemy każdy punkt na płaszczyźnie
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tworzymy contourplot
# plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.bwr)
# plt.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
# plt.show()

# Zadanie 2:
# Stwórz listę kilku wybranych przez siebie wartości dla parametru n_neighbors
# W pętli 'for' użyj kolejnych wartości parametru do stworzenia klasyfikatora
# Następnie naucz go na danych uczących
# Zapisz wynik scoringu na danych testowych do osobnej listy

lista_n = [1,2,3,4,5,6,7,8,9]
dokladnosci = []

for n_neighb in lista_n:
    knn = KNeighborsClassifier(n_neighbors=n_neighb)
    # Uczymy klasyfikator na zbiorze uczącym
    knn.fit(X_train, y_train)
    # Przewidujemy wartości dla zbioru testowego
    y_pred = knn.predict(X_test)
    # Sprawdzamy kilka pierwszych wartości przewidzianych
    print(y_pred[:5])
    # Sprawdzamy dokładność klasyfikatora
    dokladnosc = knn.score(X_test, y_test)
    print("N: %d dokladnosc: %7.30f" % (n_neighb, dokladnosc))
    dokladnosci.append(dokladnosc)

# Wyświetl wykres zależności między liczbą sąsiadów a dokładnością.
plt.scatter(lista_n, dokladnosci, edgecolor='k')
plt.show()