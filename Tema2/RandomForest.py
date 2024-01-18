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
            X_bootstrap, y_bootstrap = X.iloc[indices], y.iloc[indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions / self.n_trees


class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or (
                self.max_depth is not None and depth == self.max_depth):
            print(f"Leaf Node: Depth={depth}, Value={np.mean(y)}")
            self.tree = {'value': np.mean(y)}  # Set self.tree for leaf nodes
            return self.tree

        best_index, best_value = self.find_best_split(X, y)
        if best_index is None or best_value is None:
            # Return a leaf node with the average value if a split cannot be found
            print(f"Leaf Node: Depth={depth}, Value={np.mean(y)}")
            self.tree = {'value': np.mean(y)}  # Set self.tree for leaf nodes
            return self.tree
        left_mask = X.loc[:, best_index] <= best_value
        right_mask = ~left_mask

        print(f"Split Node: Depth={depth}, Feature={best_index}, Value={best_value}")
        left_subtree = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.fit(X[right_mask], y[right_mask], depth + 1)
        self.tree = {'index': best_index, 'value': best_value, 'left': left_subtree, 'right': right_subtree}
        return self.tree

    def find_best_split(self, X, y):
        best_index, best_value, best_mse = None, None, float('inf')
        for i in X.columns:
            values = X[i].unique()
            for value in values:
                left_mask = X[i] <= value
                right_mask = ~left_mask
                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue
                mse = self.calculate_mse(y[left_mask]) + self.calculate_mse(y[right_mask])
                if mse < best_mse:
                    best_index, best_value, best_mse = i, value, mse
        return best_index, best_value

    def calculate_mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def predict(self, X):
        return [self.predict_single(x, self.tree) for _, x in X.iterrows()]

    def predict_single(self, x, tree):
        if 'value' in tree:
            return tree['value']
        else:
            if x[tree['index']] <= tree['value']:
                return self.predict_single(x, tree['left'])
            else:
                return self.predict_single(x, tree['right'])
