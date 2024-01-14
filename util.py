import numpy as np
import math


# def sigmoid(x):
#     return 1.0 / (1.0 + math.exp(-x))

# def softmax(x):  # x -> vector de scoruri, buna pentru a calcula "cea mai importanta" valoare
#     exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# def softmax(x):
#     exps = np.exp(x)
#     return exps / np.sum(exps)

def softmax(x):
    nr = np.exp(x)
    # exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtracting max(x) for numerical stability
    # return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return nr / np.sum(nr)


def softmax_derivative(output):
    # Derivative of softmax function with respect to the input
    # The output parameter is the output of the softmax function
    s = output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


# def sigmoid_derivative(x):
#     return x * (1 - x)


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
