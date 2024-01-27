import csv
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor


def write_to_csv(csvfile_path, feature_names, X_test, y_test, y_pred):
    y_column_name = 'Quality of patient care star rating'
    with open(csvfile_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        feature_names.remove(y_column_name)
        csv_writer.writerow(feature_names + [y_column_name, 'Predicted Values'])
        for i in range(len(y_pred)):
            row = []
            for j in range(len(X_test.iloc[i])):
                row.append(X_test.iloc[i][j])
            row.extend([y_test.iloc[i], y_pred[i]])
            csv_writer.writerow(row)


def read_csv(file_path='output_file.csv'):
    data = pd.read_csv(file_path)
    return data


def custom_round(value):
    value *= 5
    integer_part = int(value)
    decimal_part = value - integer_part
    if decimal_part < 0.5:
        return integer_part
    else:
        return integer_part + 1


def split_dataset(dataset, test_size=0.2, random_state=None):
    y_column = dataset.columns[8]
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
    y_pred = list(map(lambda element: custom_round(element), y_pred))
    write_to_csv("all-data/predictions_all-data_RandomForest_sklearn.csv", dataset.columns.tolist(), X_test, y_test, y_pred)

    train2 = list(map(lambda element: custom_round(element), y_train))
    cnn = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, train2)
    y_pred = cnn.predict(X_test)
    write_to_csv("all-data/predictions_all-data_NeuralNetwork_sklearn.csv", dataset.columns.tolist(), X_test, y_test, y_pred)
