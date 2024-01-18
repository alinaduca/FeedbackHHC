import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Layer():
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def tanh(x):
    return np.tanh(x);


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2;


def relu(x):
    x[x < 0] = 0
    return x


def relu_prime(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return 1


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None
        self.errlist = []

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            displayerr = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                displayerr += self.loss(y_train[j], output)
                self.errlist.append(displayerr)
                error = self.loss_deriv(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            displayerr /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, displayerr))

    def returnerr(self):
        return self.errlist


if __name__ == '__main__':
    data = np.genfromtxt('corr-based.csv', delimiter=',', skip_header=1)
    train, test = train_test_split(data, test_size=0.4)

    xtrain = train[:, :-1].reshape(train[:, :-1].shape[0], 1, 7)
    ytrain = train[:, -1:]
    xtest = test[:, :-1].reshape(test[:, :-1].shape[0], 1, 7)
    ytest = test[:, -1:]

    net = Network()
    net.add(Layer(7, 50))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(Layer(50, 10))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(Layer(10, 1))

    net.use(mse, mse_deriv)
    net.fit(xtrain, ytrain, epochs=500, learning_rate=0.1)

    out = net.predict(xtest)
    total = 0
    count = 0
    testerrlist = []
    for i in range(len(out)):
        total += 1
        testerrlist.append(mse(out[i], ytest[i]))

    plt.plot([x for x in range(len(net.returnerr()))], net.returnerr())
    plt.savefig('error.png')
    plt.close()
    plt.plot([x for x in range(len(testerrlist))], testerrlist)
    plt.savefig('testerror.png')
    plt.close()

    with open("predictions_NeuralNetwork.csv", mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["How often patients got better at walking or moving around", "How often patients got better at taking their drugs correctly by mouth", "How often patients got better at bathing", "How often the home health team began their patients care in a timely manner", "How often patients breathing improved", "How often home health patients had to be admitted to the hospital", "How often patients got better at getting in and out of bed", "Quality of patient care star rating", "Predicted Values"])
        for i in range(len(xtest)):
            combined_row = np.concatenate((xtest[i].reshape(-1), ytest[i], out[i][0]))
            csv_writer.writerow(combined_row)
