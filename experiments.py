from ucimlrepo import fetch_ucirepo 
from id3 import ID3
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from random import randint
import numpy as np

def get_subsets(dataset, num_subsets):
    X = dataset.data.features
    y = dataset.data.targets
    # Wymieszaj dane
    X, y = shuffle(X, y, random_state=randint(0,1000))

    subset_size = len(X) // num_subsets

    # Inicjalizuj listy do przechowywania podzbiorów
    X_subsets = []
    y_subsets = []

    # Dzieli dane na podzbiory
    for i in range(num_subsets):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i < num_subsets - 1 else None
        
        X_subset = X.iloc[start_idx:end_idx]
        y_subset = y.iloc[start_idx:end_idx]
        
        X_subsets.append(X_subset)
        y_subsets.append(y_subset)
    
    return X_subsets, y_subsets

def sum_matrix(matrix):
    sums = [sum(row) for row in matrix]
    return sum(sums)

def micro_average(matrix):
    num_classes = len(matrix)
    true_positive = []
    false_positive = []
    true_negative = []
    false_negative = []
    for ind in range(0, num_classes):
        true_positive.append(matrix[ind][ind])
        false_positive.append(sum(matrix[:, ind])-matrix[ind][ind])
        false_negative.append(sum(matrix[ind])-matrix[ind][ind])
        true_negative.append(sum_matrix(matrix)-sum(matrix[:, ind])-sum(matrix[ind])+matrix[ind][ind])
    
    TP = sum(true_positive)
    FP = sum(false_positive)
    FN = sum(false_negative)
    TN = sum(true_negative)
    return np.array([
        [TP,FN],
        [FP,TN]
    ])

def get_accuracy(matrix):
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TP + TN) / (TP + FP + FN + TN)

def get_sensivity(matrix):
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TP) / (TP + FN)

def get_specificity(matrix):
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TN) / (TN + FP)

def get_precision(matrix):
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TP) / (TP + FP)

def get_f_measure(matrix):
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (2 * TP) / (2 * TP + FP + FN)

def cross_validation(tree: ID3, dataset, k):
    num_classes = len(set([a[0] for a in dataset.data.targets.values.tolist()]))
    X_subsets, Y_subsets = get_subsets(dataset, k)
    
    conf = 0

    for i in range(0, k):
        X_train = X_subsets[i]
        Y_train = Y_subsets[i]
        X_test_subsets = [subset for ind, subset in enumerate(X_subsets) if ind != i] 
        Y_test_subsets = [subset for ind, subset in enumerate(Y_subsets) if ind != i] 
        X_test = pd.concat(X_test_subsets)
        Y_test = pd.concat(Y_test_subsets)
        Y_test = [item for el in Y_test.values for item in el]

        tree.build(X_train, Y_train)
        Y_pred = tree.predict_set(X_test)

        if num_classes == 2:
            conf += confusion_matrix(Y_test, Y_pred)
        else:
            conf += micro_average(confusion_matrix(Y_test, Y_pred))
    
    return (conf/k, get_accuracy(conf), get_sensivity(conf), get_specificity(conf), get_precision(conf), get_f_measure(conf))

def experiment(tree, dataset, k, epochs):
    for i in range(0, epochs):
        result = cross_validation(tree, dataset, k)