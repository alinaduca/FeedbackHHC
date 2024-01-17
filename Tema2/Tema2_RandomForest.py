import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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


    custom_rf = RandomForestRegressorCustom(n_trees=100, max_depth=None, min_samples_split=2)
    custom_rf.fit(X_train, y_train)

    y_pred = custom_rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

