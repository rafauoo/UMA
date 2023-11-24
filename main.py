'''
Created by
Rafal Budnik
Zuzanna Damszel
'''

from ucimlrepo import fetch_ucirepo 
from id3 import ID3
from experiments import experiment
from io_data import get_balance_scale_set

mushroom = fetch_ucirepo(id=73) 
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
# Pobieranie listy atrybutów (kolumn) z DataFrame
tree = ID3(ID3.TreeType.BINARY)
tree.build(X, y)
tree.print()