'''
Created by
Rafal Budnik
Zuzanna Damszel
'''

from ucimlrepo import fetch_ucirepo 
from id3 import ID3
from experiments import experiment
from io_data import get_balance_scale_set

dataset = fetch_ucirepo(id=19)
#dataset = get_balance_scale_set()
tree = ID3(ID3.TreeType.BINARY)
experiment(tree, dataset, 10, 10)