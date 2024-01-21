import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
from graphics import write_to_csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor


def read_csv(file_path='corr-based.csv'):
    data = pd.read_csv(file_path)
    return data


def split_dataset(dataset, test_size=0.2, random_state=None):
    y_column = dataset.columns[-1]
    X = dataset.drop(y_column, axis=1)
    y = dataset[y_column]
    num_test_samples = int(len(dataset) * test_size)
    test_indices = X.sample(n=num_test_samples, random_state=random_state).index
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    X_train = X.drop(test_indices)
    y_train = y.drop(test_indices)

    print("X_test :", X_test.shape)
    print("X_train :", X_train.shape)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    dataset = read_csv()
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    rf = RandomForestRegressor()
    crf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred = list(map(lambda element: round(element * 5), y_pred))
    # write_to_csv("predictions_RandomForest_sklearn.csv", dataset.columns.tolist(), X_test, y_test, y_pred)

    train2 = list(map(lambda element: round(element * 5), y_train))
    cnn = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, train2)
    y_pred = cnn.predict(X_test)
    # write_to_csv("predictions_NeuralNetwork_sklearn.csv", dataset.columns.tolist(), X_test, y_test, y_pred)
