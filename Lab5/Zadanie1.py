from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Wczytaj przykładowy zbiór danych - dane dotyczące trzech gatunków Irysów
iris = datasets.load_iris()

# Podzielmy zbiór na cechy oraz etykiety
# Zostawiamy tym razem wszystkie cechy - będziemy próbować odgadnąć które cechy są najważniejsze
X = iris.data
y = iris.target

# Inicjalizacja. Można od razu wypełnić n_components, wykorzystujemy wszystkie cechy
# pca = PCA(n_components=3)

pca = PCA()
pca.fit(X)

# Analiza (dekompozycja) PCA tworzy nam n nowych "sztucznych" cech, które starają się jak najlepiej
# odzwierciedlić zmienność oryginalnego zbioru
print("Liczba komponentów: ", pca.n_components_)

# Dodatkowo możemy sprawdzić jaki wpływ nasze oryginalne cechy mają na wywnioskowane, nowe cechy
print("Skład nowych cech:")
print(pca.components_)

# Na koniec możemy określić które nowe, wywnioskowane cechy mają największy wpływ na ogólną zmienność zbioru
print(pca.explained_variance_ratio_)

# Jedna *nowa* cecha tłumaczy prawie wszystko? Sprawdźmy!
# Czy potrafisz określić kierunki największej zmienności danych?

# wykresy będą tworzone przy pomocy pakietu seaborn
import seaborn as sns

# konwersja na obiekt pandas.DataFrame
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# funkcja która nam zamieni wartości 0, 1, 2 na pełny opis tekstowy dla gatunku
targets = map(lambda x: iris['target_names'][x], iris['target'] )

# doklejenie informacji o gatunku do reszty dataframe
iris_df['species'] = np.array(list(targets))

# wykres
sns.pairplot(iris_df, hue='species')
plt.show()

# Spróbujmy zredukować nasz zbiór cech do tej jednej, najlepszej

pca_limit = PCA(n_components = 1)

X_new = pca_limit.fit_transform(X)

# Cechy:
print("Liczba komponentów: ", print(pca_limit.n_components_))

# Wpływ oryginalnych cech na wywnioskowaną cechę
print("Skład nowej cechy:")
print(pca_limit.components_)

# "Wytłumaczalność" nowej cechy dalej jest bardzo wysoka
print(pca_limit.explained_variance_ratio_)

# Po użyciu funkcji transform (lub fit_transform) dekompozycja pozostawiła nam tylko liczbę cech, którą skonfigurowaliśmy
# Dodatkowo została od nich odjęta średnia, więc dane zawierają tylko wariancję

X_new[:5]

plt.scatter(X_new, y)
plt.show()

# Na podstawie ostatniego przedotatniego wykresu, na którym widnieją wartości do 7 cm, cechą prawdopodobnie wybraną została cecha petal lenght. Ponadto jeden z targetów
# jest oddzielony od reszty, co pozwala lepiej rozróżnić rośliny wedle danej cechy.
