from ucimlrepo import fetch_ucirepo 
from id3 import ID3
# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
# Pobieranie listy atrybutów (kolumn) z DataFrame
tree = ID3(ID3.TreeType.BINARY)
tree.build(X, y, X.columns.tolist())
tree.print()