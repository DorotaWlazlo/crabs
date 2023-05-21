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


def gradient_descend(net_input, y_data, num_iterations, ni, n_input, n_hidden, n_output):
    weights_hidden, weights_output = initialize_network(n_input, n_hidden, n_output)
    accuracies = []
    iterations = []

    for i in range(num_iterations):
        hidden_activation, hidden_output, output_activation, output_output = \
            forward_propagation(weights_hidden, weights_output, net_input)
        weights_hidden, weights_output = backward_propagation(net_input, hidden_activation, hidden_output,
                                                              output_activation, output_output, y_data, weights_hidden,
                                                              weights_output, ni)

        if i % 10 == 0:
            print("Iteration: ", i)
            accuracy = get_accuracy(get_predictions(output_output), y_data)
            print("Accuracy: ", accuracy)
            accuracies.append(accuracy)
            iterations.append(i)

    return weights_hidden, weights_output, accuracies, iterations


def get_accuracy(predictions, y_data):
    return np.sum(np.all(predictions == y_data, axis=0)) / y_data.shape[1]


def get_predictions(output_output):
    maxes = np.argmax(output_output, 0)
    num_columns = len(maxes)
    predictions = np.zeros((4, num_columns))

    for i, index in enumerate(maxes):
        predictions[index, i] = 1
    return predictions


def check_class(code):
    first_class = np.array([1, 0, 0, 0])

    second_class = np.array([0, 1, 0, 0])

    third_class = np.array([0, 0, 1, 0])

    fourth_class = np.array([0, 0, 0, 1])

    if np.array_equal(code, first_class):
        return 0
    if np.array_equal(code, second_class):
        return 1
    if np.array_equal(code, third_class):
        return 2
    if np.array_equal(code, fourth_class):
        return 3


def test_predictions(weights_hidden, weights_output,y_data_test, x_data_test):
    _, _, _, output_output = forward_propagation(weights_hidden, weights_output, x_data_test)
    predictions = get_predictions(output_output)
    accuracy = get_accuracy(predictions, y_data_test)
    confusion_matrix = np.zeros((4, 4))

    for i in range(y_data_test.shape[1]):
        x = check_class(y_data_test[:, i])
        y = check_class(predictions[:, i])
        confusion_matrix[x][y] += 1

    return accuracy, confusion_matrix


def calculate_sensitivity(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    sensitivities = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        sensitivities[i] = true_positives / (true_positives + false_negatives)

    return sensitivities


def calculate_specificity(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    specificities = np.zeros(num_classes)

    for i in range(num_classes):
        true_negatives = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + \
                         confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
        specificities[i] = true_negatives / (true_negatives + false_positives)

    return specificities

