import pandas as pd
from neural_network import NeuralNetwork


def read_csv(file_path):
    with open(file_path) as file:
        first_line = file.readline().strip()
        column_names = first_line.split(',')
    column_names = [name.replace("'", '').replace('`', '').replace('"', '') for name in column_names]
    data = pd.read_csv(file_path, names=column_names, skiprows=1)
    data.replace(',', '', regex=True, inplace=True)
    return data


def main():
    data = read_csv("corr-based.csv")
    input_size = 7
    layer_config = [8, 7, 6]
    learning_rate = 0.2
    network = NeuralNetwork(input_size, layer_config, learning_rate)
    network.train(data.values, epochs=1)
    predictions = network.predict(data.values)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
