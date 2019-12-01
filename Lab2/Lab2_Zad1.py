from sklearn import datasets

# Wczytaj przykładowy zbiór danych - dane dotyczące trzech gatunków Irysów
iris = datasets.load_iris()

# Zobaczmy jakie dane mamy w zbiorze
print('Elementy zbioru: ', list(iris.keys()))

# Zobaczmy jak wyglądają elementy zbioru
print('Typ pierwszego elementu z \'data\': ', type(iris['data']))
print('Kilka pierwszych elementów:')
print(iris['data'])

# Kwiaty mają swoje etykiety numeryczne...
print('Pierwszy kwiat w zbiorze to: ', iris['target'])

# ... a odpowiadające im nazwy są osobno
print('Pierwszy kwiat w zbiorze (słownie) to: ', iris['target_names'])

# Etykiety które występują
print('Cechy irysów w zbiorze to: ', iris['feature_names'])

# Opis, którego brakuje w treści zadania.
print('Ukryty opis: ', iris['DESCR'])

# Opis, którego brakuje w treści zadania.
print('Nazwa pliku: ', iris['filename'])


#Zbiór zawiera dane dotyczące trzech gatunków irysów. Klucz "data" określa wymiary kwiatu danego gatunku, które zostały wyszczególnione w kluczu "feature_names".
# Każda tabela wymiarów w "data" jest przypisana do każdego kwiatu w "target". Description zawiera informacje dotyczące bazy danych, użytej w zadaniu,
# natomiast klucz filename to ścieżka,która prowadzi do pliku iris.csv.



