'''
Created by
Rafal Budnik
Zuzanna Damszel
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from experiments import get_subsets, get_accuracy

def cross_validation(tree: DecisionTreeClassifier, dataset, k, classes):
    X_subsets, Y_subsets = get_subsets(dataset, k)
    
    conf = 0
    leaf_count = 0

    for i in range(0, k):
        X_test = X_subsets[i]
        Y_test = Y_subsets[i]
        X_train_subsets = [subset for ind, subset in enumerate(X_subsets) if ind != i] 
        Y_train_subsets = [subset for ind, subset in enumerate(Y_subsets) if ind != i] 
        X_train = pd.concat(X_train_subsets)
        Y_train = pd.concat(Y_train_subsets)
        from sklearn.preprocessing import OneHotEncoder
        # Combine training and test sets
        combined_data = pd.concat([X_train, X_test], axis=0)

        # Convert categorical columns to one-hot encoding
        combined_data_encoded = pd.get_dummies(combined_data)

        # Split the combined data back into training and test sets
        X_train_encoded = combined_data_encoded.iloc[:len(X_train)]
        X_test_encoded = combined_data_encoded.iloc[len(X_train):]

        # Example using OneHotEncoder
        tree.fit(X_train_encoded, Y_train)
        leaf_count += tree.get_n_leaves()
        Y_pred = tree.predict(X_test_encoded)
        conf += confusion_matrix(Y_test, Y_pred, labels=classes)

    return (conf/k, leaf_count/k, get_accuracy(conf))

def exp_sklearn_tree_our_cross(dataset, k, epochs):
    tree = DecisionTreeClassifier(criterion='entropy')
    classes = sorted(list(set(dataset.data.targets.iloc[:, 0].tolist())))
    conf = 0
    acc = 0
    leaf_count = 0
    for i in range(0, epochs):
        result = cross_validation(tree, dataset, k, classes)
        conf += result[0]
        leaf_count += result[1]
        acc += result[2]
    print("Mean Confusion Matrix")
    print(classes)
    np.set_printoptions(suppress=True, precision=4)
    print(np.array(conf/epochs))
    print(f"Mean leaf count: {(leaf_count/epochs):.2f}")
    print(f"Mean Accuracy: {(acc*100/epochs):.2f}%")
