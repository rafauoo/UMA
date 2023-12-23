'''
Created by
Rafal Budnik
Zuzanna Damszel
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from experiments import get_accuracy
from random import randint
import numpy as np

def exp_sklearn_tree_sklearn_cross(dataset, k, epochs):
    classes = sorted(list(set(dataset.data.targets.iloc[:, 0].tolist())))
    X = dataset.data.features
    y = dataset.data.targets
    categorical_columns = dataset.data.features.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder='passthrough'
    )
    conf = 0
    acc = 0
    for i in range(0, epochs):
        tree = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(criterion='entropy'))
        ])
        X_shuffled, y_shuffled = shuffle(X, y, random_state=randint(0,1000))
        y_pred = cross_val_predict(tree, X_shuffled, y_shuffled, cv=k)
        matrix = confusion_matrix(y, y_pred, labels=classes)
        acc += get_accuracy(matrix)
        conf += matrix / k
    print("Mean Confusion Matrix")
    print(classes)
    np.set_printoptions(suppress=True, precision=4)
    print(np.array(conf/epochs))
    print(f"Mean Accuracy: {(acc*100/epochs):.2f}%")
