import numpy as np


class RandomForestRegressorCustom:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for tree_num in range(self.n_trees):
            print(f"\nBuilding Tree {tree_num + 1}...")
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap, y_bootstrap = X.iloc[indices, :], y.iloc[indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            print(f"Best Split for Tree {tree_num + 1}: Feature={tree.index}, Value={tree.value}")

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions / self.n_trees


class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.index = None
        self.value = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.leaf_value = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or (
                self.max_depth is not None and depth == self.max_depth):
            self.is_leaf = True
            self.leaf_value = np.mean(y)
            return self

        best_index, best_value = self.find_best_split(X, y)
        if best_index is None or best_value is None:
            self.is_leaf = True
            self.leaf_value = np.mean(y)
            return self

        left_mask = X.loc[:, best_index] <= best_value
        right_mask = ~left_mask

        self.index = best_index
        self.value = best_value
        self.left = DecisionTreeRegressorCustom(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        self.right = DecisionTreeRegressorCustom(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right.fit(X[right_mask], y[right_mask], depth + 1)

        return self

    def find_best_split(self, X, y):
        best_index, best_value, best_mse = None, None, float('inf')
        for column in X.columns:
            values = X[column].unique()
            for value in values:
                left_mask = X[column] <= value
                right_mask = ~left_mask
                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue
                mse = self.calculate_mse(y[left_mask]) + self.calculate_mse(y[right_mask])
                if mse < best_mse:
                    best_index, best_value, best_mse = column, value, mse
        return best_index, best_value

    def calculate_mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def predict(self, X):
        return [self.predict_single(x) for _, x in X.iterrows()]

    def predict_single(self, x):
        if self.is_leaf:
            return self.leaf_value
        elif x[self.index] <= self.value:
            return self.left.predict_single(x)
        else:
            return self.right.predict_single(x)
