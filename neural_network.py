import numpy as np

from data_preparation import load_file, class_conversion, data_scaling, splitting_into_training_and_test


def initialize_data(file_name):
    lines = load_file(file_name)
    lines_converted = class_conversion(lines)
    data_array = data_scaling(lines_converted)
    training_data, test_data = splitting_into_training_and_test(data_array)

    training_data_t = training_data.T
    y_data_train = training_data_t[:4]
    x_data_train = training_data_t[4:]

    test_data_t = test_data.T
    y_data_test = test_data_t[:4]
    x_data_test = test_data_t[4:]

    return y_data_train, x_data_train, y_data_test, x_data_test


def initialize_network(n_input, n_hidden, n_output):
    weights_hidden = np.random.randn(n_hidden, n_input)
    weights_output = np.random.randn(n_output, n_hidden)

    return weights_hidden, weights_output


def forward_propagation(weights_hidden, weights_output, net_input):
    hidden_activation = weights_hidden.dot(net_input)
    hidden_output = np.tanh(hidden_activation)

    output_activation = weights_output.dot(hidden_output)
    output_output = np.tanh(output_activation)

    return hidden_activation, hidden_output, output_activation, output_output


def tanh_derivative(x):
    tanh_x = np.tanh(x)
    derivative = 1 - np.power(tanh_x, 2)
    return derivative


def backward_propagation(net_input, hidden_activation, hidden_output, output_activation,
                         output_output, y_data, weights_hidden, weights_output, ni):
    error_output = (y_data - output_output) * tanh_derivative(output_activation)
    error_hidden = tanh_derivative(hidden_activation) * (weights_output.T.dot(error_output))

    weights_output += ni * error_output.dot(hidden_output.T)
    weights_hidden += ni * error_hidden.dot(net_input.T)

    return weights_hidden, weights_output


def gradient_descend(net_input, y_data, iterations, ni, n_input, n_hidden, n_output):
    weights_hidden, weights_output = initialize_network(n_input, n_hidden, n_output)

    for i in range(iterations):
        hidden_activation, hidden_output, output_activation, output_output = \
            forward_propagation(weights_hidden, weights_output, net_input)
        weights_hidden, weights_output = backward_propagation(net_input, hidden_activation, hidden_output,
                                                              output_activation, output_output, y_data, weights_hidden,
                                                              weights_output, ni)

        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(output_output), y_data))

    return weights_hidden, weights_output

    # print(output_output)
    # print(y_data)


# def get_predictions(output_output):
#     return np.where(output_output < 0, 0, 1)
#
#
# def get_accuracy(predictions, y_data):
#     return np.sum(predictions == y_data)/y_data.size

def get_accuracy(predictions, y_data):
    return np.sum(np.all(predictions == y_data, axis=0)) / y_data.shape[1]


# def get_predictions(A2):
#     return np.argmax(A2, 0)

def get_predictions(output_output):
    maxes = np.argmax(output_output, 0)
    num_columns = len(maxes)
    predictions = np.zeros((4, num_columns))

    for i, index in enumerate(maxes):
        predictions[index, i] = 1
    return predictions
