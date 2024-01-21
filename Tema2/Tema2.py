import pandas as pd
import numpy as np
from graphics import write_to_csv
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from RandomForest import RandomForestRegressorCustom


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

    '''
    X_test_first_5 = X_test.head(5)
    y_test_first_5 = y_test.head(5)
    custom_rf = RandomForestRegressorCustom(n_trees=3, max_depth=5, min_samples_split=2)
    custom_rf.fit(X_train, y_train)
    y_pred_first_5 = custom_rf.predict(X_test_first_5)
    print("Predictions for the first 5 rows:", y_pred_first_5)
    '''

    '''
    custom_rf = RandomForestRegressorCustom(n_trees=50, max_depth=100, min_samples_split=2)
    custom_rf.fit(X_train, y_train)
    y_pred = custom_rf.predict(X_test)
    csvfile_path = "predictions_RandomForest_1_100.csv"
    write_to_csv(csvfile_path, dataset.columns.tolist(), X_test, y_test, y_pred)
    '''

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred = list(map(lambda element: round(element * 5), y_pred))
    write_to_csv("predictions_RandomForest_sklearn.csv", dataset.columns.tolist(), X_test, y_test, y_pred)


    train2 = list(map(lambda element: round(element * 5), y_train))
    clf = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, train2)
    y_pred = clf.predict(X_test)
    write_to_csv("predictions_NeuralNetwork_sklearn.csv", dataset.columns.tolist(), X_test, y_test, y_pred)


    # custom_rf = RandomForestRegressorCustom(n_trees=100, max_depth=100, min_samples_split=2)
    # custom_rf.fit(X_train, y_train)
    # y_pred = custom_rf.predict(X_test)
    # mse2 = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error2:", mse2)
    # print(y_pred)
    # csvfile_path = "predictions_RandomForest_100_100.csv"
    # write_to_csv(csvfile_path, dataset.columns.tolist(), X_test, y_test, y_pred)
    #
    # custom_rf = RandomForestRegressorCustom(n_trees=100, max_depth=50, min_samples_split=2)
    # custom_rf.fit(X_train, y_train)
    # y_pred = custom_rf.predict(X_test)
    # mse3 = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error3:", mse3)
    # print(y_pred)
    # csvfile_path = "predictions_RandomForest_100_50.csv"
    # write_to_csv(csvfile_path, dataset.columns.tolist(), X_test, y_test, y_pred)
