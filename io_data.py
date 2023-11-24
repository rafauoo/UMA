'''
Created by
Rafal Budnik
Zuzanna Damszel
'''


import pandas as pd
from ucimlrepo.dotdict import dotdict

def get_balance_scale_set():
    '''
    Balance Scale set is not available through fetch set command.
    Code of this function is modified code from ucimlrepo library.
    '''
    data_path = './data/balance_scale/balance-scale.data'
    column_names = ['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance']
    df = pd.read_csv(data_path, names=column_names)
    X = df.drop('Class', axis=1)
    y = df[['Class']]
    data = {
        'features': X,
        'targets': y,
        'original': df,
    }
    result = {
        'data': dotdict(data),
    }
    return dotdict(result)