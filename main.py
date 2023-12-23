'''
Created by
Rafal Budnik
Zuzanna Damszel
'''

from ucimlrepo import fetch_ucirepo
from id3 import ID3
from experiments import experiment
from io_data import get_balance_scale_set
from exp_sklearn_tree_our_cross import exp_sklearn_tree_our_cross
from exp_sklearn_tree_sklearn_cross import exp_sklearn_tree_sklearn_cross

#dataset = fetch_ucirepo(id=19) # Car Evaluation
dataset = fetch_ucirepo(id=14) # Breast Cancer
#dataset = get_balance_scale_set() # Balance Scale


print("=======================")
print(dataset.metadata.name)
print("=======================")
print("OUR TREE BINARY")
experiment(ID3(ID3.TreeType.BINARY), dataset, 10, 10)
print("")
print("OUR TREE IDENTITY")
experiment(ID3(ID3.TreeType.IDENTITY), dataset, 10, 10)
print("")
print("SKLEARN TREE")
exp_sklearn_tree_our_cross(dataset, 10, 10)
