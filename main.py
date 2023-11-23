from ucimlrepo import fetch_ucirepo 
from id3 import ID3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# fetch dataset 
print("Program start")
mushroom = fetch_ucirepo(id=73) 
print("Set downloaded")
# data (as pandas dataframes) 
X = mushroom.data.features
y = mushroom.data.targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
print(X_train)
# Pobieranie listy atrybutów (kolumn) z DataFrame
tree = ID3(ID3.TreeType.BINARY)
tree.build(X_train, y_train, X_train.columns.tolist())
# tree.print()
print("Node count:", tree.get_tree_node_count())
y_pred = tree.predict_set(X_test)
y_test = [item for sublist in y_test.values for item in sublist]
# Oblicz i wypisz różne metryki
# print("Y TEST:")
# print(y_test)
# print("Y PRED:")
# print(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='p'))
print("Recall:", recall_score(y_test, y_pred, pos_label='p'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='p'))
# Wypisz macierz pomyłek (confusion matrix)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# # for ind, data in enumerate(X.iloc):
#     result = tree.predict(data)
#     if result != y.iloc[ind]['poisonous']:
#         print("ERROR FOUND!")
#         #print(data)
#         print(result)
#         print(y.iloc[ind])
#         print("      ")