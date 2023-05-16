import numpy as np

from data_preparation import load_file, class_conversion, data_scaling, splitting_into_training_and_test


def initialize_data(file_name):
    lines = load_file(file_name)
    lines_converted = class_conversion(lines)
    data_array = data_scaling(lines_converted)
    training_data, test_data = splitting_into_training_and_test(data_array)

    training_data_t = training_data.T
    y_data_train = training_data_t[0:3]
    x_data_train = training_data_t[4:8]

    test_data_t = test_data.T
    y_data_test = test_data_t[0:3]
    x_data_test = test_data_t[4:8]

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



