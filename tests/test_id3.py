'''
Created by
Rafal Budnik
Zuzanna Damszel
'''

from ucimlrepo import fetch_ucirepo 
from id3 import ID3

def test_mushroom_binary():
    """
    Testing Binary ID3 tree on Mushroom set
    """
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 
    # Pobieranie listy atrybutów (kolumn) z DataFrame
    tree = ID3(ID3.TreeType.BINARY)
    tree.build(X, y)
    #tree.print()
    same_count = 0
    for ind, data in enumerate(X.iloc):
        result = tree.predict(data)
        if result == y.iloc[ind]['poisonous']:
            same_count += 1
    assert same_count == len(y)

def test_mushroom_identity():
    """
    Testing Binary ID3 tree on Mushroom set
    """
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 
    # Pobieranie listy atrybutów (kolumn) z DataFrame
    tree = ID3(ID3.TreeType.IDENTITY)
    tree.build(X, y)
    #tree.print()
    same_count = 0
    for ind, data in enumerate(X.iloc):
        result = tree.predict(data)
        if result == y.iloc[ind]['poisonous']:
            same_count += 1
    assert same_count == len(y)

def test_mushroom_binary_node_count():
    """
    Testing Binary ID3 tree on Mushroom set
    """
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 
    # Pobieranie listy atrybutów (kolumn) z DataFrame
    tree = ID3(ID3.TreeType.BINARY)
    tree.build(X, y)
    node_count = tree.get_tree_node_count()
    assert node_count == 23

def test_mushroom_identity_node_count():
    """
    Testing Binary ID3 tree on Mushroom set
    """
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 
    # Pobieranie listy atrybutów (kolumn) z DataFrame
    tree = ID3(ID3.TreeType.IDENTITY)
    tree.build(X, y)
    node_count = tree.get_tree_node_count()
    assert node_count == 28