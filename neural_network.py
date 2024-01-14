import random
from util import *


class NeuralNetwork:
    def __init__(self, input_size, layers, learning_rate=0.01):
        random.seed(0)
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.error_plot = []
        self.network = []
        for i in range(1, len(layers)):
            layer = [{'weights': [random.random() for _ in range(layers[i - 1] + 1)],
                      'output': 0, 'delta': 0} for _ in range(layers[i] + 1)]
            self.network.append(layer)

    def forward_propagation(self, inputs):
        for layer in self.network:
            outputs = []  # outputs of the current layer
            local_weights = []
            for neuron in layer:
                weights = neuron['weights']
                for i in range(len(weights) - 1):
                    neuron['output'] += weights[i] * inputs[i]
                local_weights.append(neuron['output'])
                # print(neuron['output'])
            local_weights = softmax(local_weights)
            for index, neuron in enumerate(layer):
                neuron['output'] = local_weights[index]
            outputs = local_weights
            inputs = outputs
        return inputs

    def backward_propagation(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * softmax_derivative(neuron['output'])

    def train(self, x, epochs):
        for i in range(epochs):
            predictions = [0 for _ in range(len(x))]
            for trail in range(len(x)):
                inputs = x[trail]
                outputs = self.forward_propagation(inputs).tolist()
                predictions[trail] = outputs.index(max(outputs))
                expected = [0 for i in range(len(outputs))]
                expected[int(inputs[-1])] = 1
                self.backward_propagation(expected)
                self.update_weights(inputs[:-1])
            targets = [int(i[-1]) for i in x]
            error_value = mean_squared_error(predictions, targets)
            self.error_plot.append([error_value, i])
            print("Epoch: ", i + 1, ", loss: ", error_value, sep='')

    def update_weights(self, inputs):
        for neuron in self.network[0]:
            for j in range(len(inputs)):
                neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += self.learning_rate * neuron['delta']
        for i in range(1, len(self.network)):
            for neuron in self.network[i]:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']  # bias

    def predict(self, data):
        predictions = []
        for row in data:
            output = self.forward_propagation(row).tolist()
            print("Output:")
            print(output)
            ans = output.index(max(output))
            print("Ans:")
            print(ans)
            predictions.append(ans)
        return predictions
