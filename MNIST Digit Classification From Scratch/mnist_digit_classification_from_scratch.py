import numpy as np
import pandas as pd

from typing import Callable


def create_bias(shape: (int, int) = (0, 0)):
    """
       Initialize bias based on 'shape'
    """
    # assert if shape is not set
    assert shape != (0, 0)

    m, n = shape
    b = np.random.rand(m, 1) - 0.5
    return b


def create_weights(shape: (int, int) = (0, 0)):
    """
    Initialize weights based on 'shape'
    """
    # assert if shape is not set
    assert shape != (0, 0)

    m, n = shape
    w = np.random.rand(m, n) - 0.5
    return w


def init_weights_and_biases(num_layers: int = 2, input_shape: (int, int) = (1, 784),
                            hidden_units: int = 10, output_units: int = 10):
    """
    Initialize weights and biases
    """
    m, n = input_shape
    layers = {}
    for i in range(num_layers):
        # create weights and biases between input layer and first hidden layer
        if i == 0:
            layers[i] = {
                'weights': create_weights(shape=(hidden_units, n)),
                'bias': create_bias(shape=(hidden_units, n))
            }
        # create weights and biases between last hidden layer and output layer
        elif i == num_layers:
            layers[i] = {
                'weights': create_weights(shape=(output_units, n)),
                'bias': create_bias(shape=(output_units, n))
            }
        # create weights and biases between hidden layers
        else:
            layers[i] = {
                'weights': create_weights(shape=(hidden_units, hidden_units)),
                'bias': create_bias(shape=(hidden_units, hidden_units))
            }

    return layers


def ReLU(z):
    """
    Return z if z>0 else return 0
    """
    return np.maximum(z, 0)


def deriv_ReLu(z):
    """
    Return the derivative of ReLu
    """
    return z > 0


def one_hot(Y):
    """
    Convert the labels into one hot encodings
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def softmax(Z):
    """
    Apply softmax function to input array
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def calculate_state(weights: np.array, activations: np.array, bias: np.array):
    """
    Calculate the state of the neurons, often known as variable 'z'
    """
    z = weights.dot(activations) + bias
    return z


def calculate_activation(z: np.array, activation_function: Callable = ReLU):
    """
    Calculate activation of neurons based on given activation function
    """
    a = activation_function(z)
    return a


def update_params(layers, derivatives, alpha):
    """
    Update the weights and biases
    """
    for layer in layers:
        W = layers[layer]['weights'] - alpha * derivatives[layer]['dW']
        b = layers[layer]['bias'] - alpha * derivatives[layer]['db']
        layers[layer] = {
            'weights': W,
            'bias': b
        }
    return layers


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def forward_prop(layers: dict, x: np.array):
    # create dict to hold calculated values
    neurons = {}
    # Loop through the layers
    for layer in layers:
        # extract data
        W = layers[layer]['weights']
        b = layers[layer]['bias']
        if layer == 0:
            # calculate z and a
            z = W.dot(x) + b
            a = ReLU(z)
        else:
            z = W.dot(neurons[layer-1]['a']) + b

            if layer == len(layers) - 1:
                a = softmax(z)
            else:
                a = ReLU(z)
        neurons[layer] = {'z': z, 'a': a}

    return neurons


def back_prop(neurons, layers, X, Y):

    m = Y.size
    one_hot_Y = one_hot(Y)
    derivatives = {}
    # Loop in reversed
    for layer in reversed(neurons):
        # output layer
        if layer == len(neurons) - 1:
            dZ = neurons[layer]['a'] - one_hot_Y
            dW = 1 / m * dZ.dot(neurons[layer - 1]['a'].T)
            db = 1 / m * np.sum(dZ)
        # input layer
        elif layer == 0:
            dZ = layers[layer + 1]['weights'].T.dot(derivatives[layer + 1]['dZ']) * deriv_ReLu(neurons[layer]['z'])
            dW = 1 / m * dZ.dot(X.T)
            db = 1 / m * np.sum(dZ)
        # if hidden layer
        # TODO: This does not work, if network is > 3 layers learning goes bad
        else:
            dZ = layers[layer + 1]['weights'].T.dot(derivatives[layer + 1]['dZ']) * deriv_ReLu(neurons[layer]['z'])
            dW = 1 / m * dZ.dot(neurons[layer - 1]['a'].T)
            db = 1 / m * np.sum(dZ)
        derivatives[layer] = {'dZ': dZ, 'db': db, 'dW': dW}

    return derivatives


def gradient_descent(X, Y, iterations, alpha):
    layers = init_weights_and_biases()

    for i in range(iterations):
        neurons = forward_prop(layers, X)
        derivatives = back_prop(neurons, layers, X, Y)
        layers = update_params(layers, derivatives, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(neurons[1]['a'])
            print(get_accuracy(predictions, Y))
    return None


def main():
    data = pd.read_csv('train.csv')
    # We don't want to use pandas but rather np arrays
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)  # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _, m_train = X_train.shape

    ok = gradient_descent(X_train, Y_train, iterations=100, alpha=0.1)


if __name__ == '__main__':
    main()
