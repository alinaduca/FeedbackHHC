import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from graphics import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error, \
    ConfusionMatrixDisplay
from RandomForest import RandomForestRegressorCustom
import matplotlib.pyplot as plt


def read_csv(file_path='corr-based.csv'):
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
    
    custom_rf = RandomForestRegressorCustom(n_trees=1, max_depth=100, min_samples_split=2)
    custom_rf.fit(X_train, y_train)
    y_pred = custom_rf.predict(X_test)
    mse1 = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error1:", mse1)
    print(y_pred)
    csvfile_path = "predictions_RandomForest_1_100.csv"
    write_to_csv(csvfile_path, dataset.columns.tolist(), X_test, y_test, y_pred)

    custom_rf = RandomForestRegressorCustom(n_trees=100, max_depth=100, min_samples_split=2)
    custom_rf.fit(X_train, y_train)
    y_pred = custom_rf.predict(X_test)
    mse2 = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error2:", mse2)
    print(y_pred)
    csvfile_path = "predictions_RandomForest_100_100.csv"
    write_to_csv(csvfile_path, dataset.columns.tolist(), X_test, y_test, y_pred)

    custom_rf = RandomForestRegressorCustom(n_trees=100, max_depth=None, min_samples_split=2)
    custom_rf.fit(X_train, y_train)
    y_pred = custom_rf.predict(X_test)
    mse3 = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error3:", mse3)
    print(y_pred)
    csvfile_path = "predictions_RandomForest_100_None.csv"
    write_to_csv(csvfile_path, dataset.columns.tolist(), X_test, y_test, y_pred)

    with open('mse_values.txt', 'w') as file:
        file.write("mse1: " + str(mse1) + "\n")
        file.write("mse2: " + str(mse2) + "\n")
        file.write("mse3: " + str(mse3) + "\n")

    predicted_dataset = read_csv("predictions_RandomForest_1_100.csv")
    true_values = predicted_dataset['Quality of patient care star rating']
    predicted_values = predicted_dataset['Predicted Values']
    predicted_values = predicted_values.apply(custom_round)
    true_values = true_values.apply(custom_round)
    cm = confusion_matrix(true_values, predicted_values)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig('confusion_matrix.png')
