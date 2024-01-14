import numpy as np
import math


def softmax(x):
    nr = np.exp(x)
    return nr / np.sum(nr)


def softmax_derivative(output):
    s = output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def mean_squared_error(predictions, targets):
    return sum((predictions[i] - targets[i]) ** 2 for i in range(len(predictions))) / len(predictions)


def sum_error(predictions, targets):
    sum([(targets[i] - predictions[i]) ** 2 for i in range(len(targets))])


def cross_entropy_loss(predictions, targets):
    loss = 0
    for i in range(len(predictions)):
        epsilon = 1e-15
        loss += -targets[i] * math.log(max(predictions[i], epsilon)) - (1 - targets[i]) * math.log(
            max(1 - predictions[i], epsilon))
    return loss / len(predictions)


def calculate_accuracy(predictions, expected):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == expected[i]:
            correct += 1
    return correct / len(predictions) * 100_000 // 10 / 100
