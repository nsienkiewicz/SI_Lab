from sklearn import datasets
import matplotlib.pyplot as plt


# Zadanie 3:
# wczytaj dane o winach za pomocą funkcji poniżej
from sklearn.datasets import load_wine

# Zbadaj zbiór danych. Stwórz wykresy obrazujące ten zbiór danych.
# Podziel zbiór danych na uczący i testowy.
# Wytrenuj klasyfikator kNN
# Dokonaj predykcji na zbiorze testowym
# Wypisz raport z uczenia: confusion_matrix oraz classification_report


###################################################

from sklearn import datasets

# Wczytaj przykładowy zbiór danych - dane dotyczące trzech gatunków win
wine = datasets.load_wine()

# Zobaczmy jakie dane mamy w zbiorze
print('Elementy zbioru: ', list(wine.keys()))

# Zobaczmy jak wyglądają elementy zbioru
print('Typ pierwszego elementu z \'data\': ', type(wine['data'][0]))
print('Kilka pierwszych elementów:')
print(wine['data'][0:5])

# Wina mają swoje etykiety numeryczne...
print('Pierwszy kwiat w zbiorze to: ', wine['target'][0])

# ... a odpowiadające im nazwy są osobno
print('Pierwszy kwiat w zbiorze (słownie) to: ', wine['target_names'][0])

# Etykiety które występują
print('Cechy irysów w zbiorze to: ', wine['feature_names'])

##############################################3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = wine.data
y = wine.target

# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Tworzymy klasyfikator k-NN używając parametru 5 sąsiadów
knn = KNeighborsClassifier(n_neighbors = 5)

# Uczymy klasyfikator na zbiorze uczącym
knn.fit(X_train, y_train)

# Przewidujemy wartości dla zbioru testowego
y_pred = knn.predict(X_test)

# Sprawdzamy kilka pierwszych wartości przewidzianych
print(y_pred[:5])

# Sprawdzamy dokładność klasyfikatora
print(knn.score(X_test, y_test))

##########################


from sklearn.metrics import classification_report, confusion_matrix

# Różnica wartości

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))