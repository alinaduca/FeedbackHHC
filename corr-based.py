import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from math import sqrt


class Queue:
    def __init__(self):
        self.queue = []

    def isEmpty(self):
        return len(self.queue) == 0

    def push(self, item, priority):
        for index, (i, p) in enumerate(self.queue):
            if set(i) == set(item):
                if p >= priority:
                    break
                del self.queue[index]
                self.queue.append((item, priority))
                break
        else:
            self.queue.append((item, priority))

    def pop(self):
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if self.queue[max_idx][1] < p:
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return item, priority


def computeMerit(subset, label):
    n = len(subset)
    rcf = []
    for feature in subset:
        coefficient = pointbiserialr(data[label], data[feature])
        rcf.append(abs(coefficient.correlation))
    rcf_mean = np.mean(rcf)
    correlation = data[subset].corr()
    correlation.values[np.tril_indices_from(correlation.values)] = np.nan
    correlation = abs(correlation)
    rff = correlation.unstack().mean()
    return (n * rcf_mean) / sqrt(n + n * (n - 1) * rff)


if __name__ == '__main__':
    with open('output_file.csv') as file:
        first_line = file.readline().strip()
        column_names = first_line.split(',')
    column_names = [name.replace("'", '').replace('`', '').replace('"', '') for name in column_names]
    data = pd.read_csv('output_file.csv', names=column_names, skiprows=1)
    data.replace(',', '', regex=True, inplace=True)
    label = 'Quality of patient care star rating'
    features = data.columns.tolist()
    features.remove(label)
    best_val = -1
    best_feature = ''
    for feature in features:
        coefficient = pointbiserialr(data[label], data[feature])
        coefficient = abs(coefficient.correlation)
        if coefficient > best_val:
            best_val = coefficient
            best_feature = feature
    queue = Queue()
    queue.push([best_feature], best_val)
    visited = []
    i = 0
    maxi = 10
    best_subset = []
    while not queue.isEmpty():
        subset, prio = queue.pop()
        if prio < best_val:
            i += 1
        else:
            best_val = prio
            best_subset = subset
        if i == maxi:
            break
        for feature in features:
            if feature != subset[0]:
                temp_subset = subset + [feature]
                for node in visited:
                    if set(node) == set(temp_subset):
                        break
                else:
                    visited.append(temp_subset)
                    merit = computeMerit(temp_subset, label)
                    queue.push(temp_subset, merit)
    output_data = data[best_subset]
    output_data.to_csv('corr-based.csv', index=False)
