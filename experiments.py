'''
Created by
Rafal Budnik
Zuzanna Damszel
'''

from id3 import ID3
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from random import randint
import numpy as np

def get_subsets(dataset, num_subsets):
    '''
    Function used for cross validation. It divides set into k subsets.
    '''
    X = dataset.data.features
    y = dataset.data.targets
    X, y = shuffle(X, y, random_state=randint(0,1000))

    subset_size = len(X) // num_subsets

    X_subsets = []
    y_subsets = []

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
    if matrix.shape[0] > 2:
        matrix = micro_average(matrix)
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TP + TN) / (TP + FP + FN + TN)

def get_sensivity(matrix):
    if matrix.shape[0] > 2:
        matrix = micro_average(matrix)
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TP) / (TP + FN)

def get_specificity(matrix):
    if matrix.shape[0] > 2:
        matrix = micro_average(matrix)
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TN) / (TN + FP)

def get_precision(matrix):
    if matrix.shape[0] > 2:
        matrix = micro_average(matrix)
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (TP) / (TP + FP)

def get_f_measure(matrix):
    if matrix.shape[0] > 2:
        matrix = micro_average(matrix)
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    return (2 * TP) / (2 * TP + FP + FN)

def cross_validation(tree: ID3, dataset, k, classes):
    X_subsets, Y_subsets = get_subsets(dataset, k)
    conf = 0
    nodes_count = 0
    leaf_count = 0

    for i in range(0, k):
        X_test = X_subsets[i]
        Y_test = Y_subsets[i]
        X_train_subsets = [subset for ind, subset in enumerate(X_subsets) if ind != i] 
        Y_train_subsets = [subset for ind, subset in enumerate(Y_subsets) if ind != i] 
        X_train = pd.concat(X_train_subsets)
        Y_train = pd.concat(Y_train_subsets)
        tree.clear()
        tree.build(X_train, Y_train)
        Y_pred = tree.predict_set(X_test)
        nodes_count += tree.get_tree_node_count()
        leaf_count += tree.get_tree_leaf_count()
        conf += confusion_matrix(Y_test, Y_pred, labels=classes)
    
    return (conf/k, get_accuracy(conf), get_sensivity(conf), get_specificity(conf), get_precision(conf), get_f_measure(conf), nodes_count/k, leaf_count/k)

def experiment(tree, dataset, k, epochs):
    '''
    Experiment is repeating cross_validation n times.
    '''
    conf = 0
    accuracy = 0
    sensivity = 0
    specificity = 0
    precision = 0
    nodes_count = 0
    f_measure = 0
    leaf_count = 0
    classes = sorted(list(set(dataset.data.targets.iloc[:, 0].tolist())))
    for i in range(0, epochs):
        result = cross_validation(tree, dataset, k, classes)
        conf += result[0]
        accuracy += result[1]
        sensivity += result[2]
        specificity += result[3]
        precision += result[4]
        f_measure += result[5]
        nodes_count += result[6]
        leaf_count += result[7]

    print("Mean Confusion Matrix")
    print(classes)
    np.set_printoptions(suppress=True, precision=4)
    print(np.array(conf/epochs))
    print(f"Mean Accuracy: {(accuracy*100/epochs):.2f}%")
    print(f"Mean Sensivity: {(sensivity*100/epochs):.2f}%")
    print(f"Mean Specificity: {(specificity*100/epochs):.2f}%")
    print(f"Mean Precision: {(precision*100/epochs):.2f}%")
    print(f"Mean F_measure: {(f_measure*100/epochs):.2f}%")
    print(f"Mean nodes count: {(nodes_count/epochs):.2f}")
    print(f"Mean leaf count: {(leaf_count/epochs):.2f}")
