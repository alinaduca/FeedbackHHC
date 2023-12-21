import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt, inf


def read_csv(file_path):
    with open(file_path) as file:
        first_line = file.readline().strip()
        column_names = first_line.split(',')
    column_names = [name.replace("'", '').replace('`', '').replace('"', '') for name in column_names]
    '''
    with open("initial_column_names.txt", 'w') as file:
       for name in column_names:
            file.write(name + '\n')
    '''
    data = pd.read_csv(file_path, names=column_names, skiprows=1)
    data.replace(',', '', regex=True, inplace=True)
    return data


data = read_csv('output_file.csv')
df = pd.DataFrame(data)
df = df.fillna(0)
df.insert(0, 'Bias', 1)
X = df.iloc[:].values
X = np.delete(X, 9, axis=1)
print(X)
y = df.iloc[:, 9].values.reshape(-1, 1)
theta = np.zeros((X.shape[1], 1))
w_y = []
w_0 = []
w_1 = []
w_2 = []
w_3 = []
w_4 = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))   # sigma(x_i) = 1/(1+e^(-x_i))


def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))  # funcția de verosimilitate condiționată
    return J


def log_likelihood(X, y, theta):
    h = sigmoid(np.dot(X, theta))
    likelihood = y * np.log(h) + (1 - y) * np.log(1 - h)
    likelihood = likelihood[~np.isnan(likelihood)]
    return np.sum(likelihood)


def gradient_descent(X, y, theta, alpha, num_iterations):
    likelihood_history = []
    for iteration in range(num_iterations):
        likelihood = log_likelihood(X, y, theta)
        print(y)
        theta -= alpha * np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))
        w_y.append(iteration + 1)
        w_0.append(theta[0, 0])
        w_1.append(theta[1, 0])
        w_2.append(theta[2, 0])
        w_3.append(theta[3, 0])
        w_4.append(theta[4, 0])
        likelihood_history.append(likelihood)
    return theta, likelihood_history


alpha = 0.5
num_iterations = 300
trained_theta, likelihood_history = gradient_descent(X, y, theta, alpha, num_iterations)

plt.xlabel('Numărul de iterații')
plt.ylabel('Valorile ponderilor')
plt.plot(w_y, w_0, marker='^', label='w_0')
plt.plot(w_y, w_1, marker='*', label='w_1')
plt.plot(w_y, w_2, marker='+', label='w_2')
plt.plot(w_y, w_3, marker='H', label='w_3')
plt.plot(w_y, w_4, marker='.', label='w_4')
plt.title('Graficul ponderilor')
plt.legend()
plt.savefig("ponderi.png")
plt.clf()


cols = []
ponderi = theta
theta = []
for pond in ponderi:
    theta.append(pond)
while len(cols) < len(ponderi):
    maxx = -inf
    index = -1
    for ind, val in enumerate(theta):
        if val[0] > maxx:
            maxx = val[0]
            index = ind
    ponderi[index][0] = -inf
    cols.append(data.columns[index])


for index, column_name in enumerate(cols):
    print(index + 1, ". ", column_name, sep='')
