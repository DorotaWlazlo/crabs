import numpy as np

from data_preparation import load_file, class_conversion, data_scaling, splitting_into_training_and_test


def initialize_data(file_name):
    """
    Initializes the data for training and testing network.

    :param file_name: The name of the file containing the data.
    :return: Four numpy arrays representing the training and testing data.
    """

    lines = load_file(file_name)  # loading data file
    lines_converted = class_conversion(lines)  # converting class labels
    data_array = data_scaling(lines_converted)  # scaling numeric data
    training_data, test_data = splitting_into_training_and_test(data_array)

    # transposing training and testing data and splitting them into input and output arrays
    training_data_t = training_data.T
    y_data_train = training_data_t[:4]
    x_data_train = training_data_t[4:]

    test_data_t = test_data.T
    y_data_test = test_data_t[:4]
    x_data_test = test_data_t[4:]

    return y_data_train, x_data_train, y_data_test, x_data_test


def initialize_network(n_input, n_hidden, n_output):
    """
    Initializes the neural network weights.

    :param n_input: Number of input neurons.
    :param n_hidden: Number of hidden neurons.
    :param n_output: Number of output neurons.
    :return: Two numpy arrays representing the weights for the hidden and output layers.
    """

    weights_hidden = np.random.randn(n_hidden, n_input)
    weights_output = np.random.randn(n_output, n_hidden)

    return weights_hidden, weights_output


def forward_propagation(weights_hidden, weights_output, net_input):
    """
    Performs forward propagation through the neural network.

    :param weights_hidden: Weights for the hidden layer.
    :param weights_output: Weights for the output layer.
    :param net_input: Input to the neural network.
    :return: Outputs of the hidden layer and output layer and there activation rates.
    """

    hidden_activation = weights_hidden.dot(net_input)
    hidden_output = np.tanh(hidden_activation)

    output_activation = weights_output.dot(hidden_output)
    output_output = np.tanh(output_activation)

    return hidden_activation, hidden_output, output_activation, output_output


def tanh_derivative(x):
    """
    Calculates the derivative of the hyperbolic tangent (tanh) function.

    :param x: Input value.
    :return: Derivative of the tanh function at the given input.
    """

    tanh_x = np.tanh(x)
    derivative = 1 - np.power(tanh_x, 2)
    return derivative


def backward_propagation(net_input, hidden_activation, hidden_output, output_activation,
                         output_output, y_data, weights_hidden, weights_output, ni):
    """
    Performs backward propagation to update the weights of the neural network.

    :param net_input: Input to the neural network.
    :param hidden_activation: Activation values of the hidden layer.
    :param hidden_output: Outputs of the hidden layer.
    :param output_activation: Activation values of the output layer.
    :param output_output: Outputs of the output layer.
    :param y_data: Target outputs.
    :param weights_hidden: Weights for the hidden layer.
    :param weights_output: Weights for the output layer.
    :param ni: Learning rate.
    :return: Updated weights for the hidden and output layers.
    """

    # computing errors
    error_output = (y_data - output_output) * tanh_derivative(output_activation)
    error_hidden = tanh_derivative(hidden_activation) * (weights_output.T.dot(error_output))

    # updating weights
    weights_output += ni * error_output.dot(hidden_output.T)
    weights_hidden += ni * error_hidden.dot(net_input.T)

    return weights_hidden, weights_output


def gradient_descend(net_input, y_data, num_iterations, ni, n_input, n_hidden, n_output):
    """
    Performs gradient descent to train the neural network.

    :param net_input: Input to the neural network.
    :param y_data: Target outputs.
    :param num_iterations: Number of iterations to perform.
    :param ni: Learning rate.
    :param n_input: Number of input neurons.
    :param n_hidden: Number of hidden neurons.
    :param n_output: Number of output neurons.
    :return: Updated weights for the hidden and output layers, list of accuracies, list of iterations.
    """

    # initializing weights
    weights_hidden, weights_output = initialize_network(n_input, n_hidden, n_output)
    accuracies = []
    iterations = []

    for i in range(num_iterations):
        # forward and backward propagation
        hidden_activation, hidden_output, output_activation, output_output = \
            forward_propagation(weights_hidden, weights_output, net_input)
        weights_hidden, weights_output = backward_propagation(net_input, hidden_activation, hidden_output,
                                                              output_activation, output_output, y_data, weights_hidden,
                                                              weights_output, ni)

        if i % 10 == 0:
            # calculating accuracy
            print("Iteration: ", i)
            accuracy = get_accuracy(get_predictions(output_output), y_data)
            print("Accuracy: ", accuracy)
            accuracies.append(accuracy)
            iterations.append(i)

    return weights_hidden, weights_output, accuracies, iterations


def get_accuracy(predictions, y_data):
    """
    Calculates the accuracy of predictions.

    :param predictions: Predicted outputs.
    :param y_data: Target outputs.
    :return: Accuracy as a decimal value.
    """

    return np.sum(np.all(predictions == y_data, axis=0)) / y_data.shape[1]


def get_predictions(output_output):
    """
    Converts the output layer activations into one-hot encoded predictions.

    :param output_output: Outputs of the output layer.
    :return: Predictions as a numpy array.
    """

    maxes = np.argmax(output_output, 0)
    num_columns = len(maxes)
    predictions = np.zeros((4, num_columns))

    for i, index in enumerate(maxes):
        predictions[index, i] = 1
    return predictions


def check_class(code):
    """
    Checks the class represented by the given code.

    :param code: One-hot encoded class code.
    :return: Class index.
    """

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
    """
    Tests the neural network predictions on the test data after finished learning.

    :param weights_hidden: Weights for the hidden layer.
    :param weights_output: Weights for the output layer.
    :param y_data_test: Target outputs for the test data.
    :param x_data_test: Input data for the test data.
    :return: Accuracy and confusion matrix.
    """

    _, _, _, output_output = forward_propagation(weights_hidden, weights_output, x_data_test)
    predictions = get_predictions(output_output)
    accuracy = get_accuracy(predictions, y_data_test) # final accuracy
    confusion_matrix = np.zeros((4, 4))

    # calculating confusion matrix
    for i in range(y_data_test.shape[1]):
        x = check_class(y_data_test[:, i])
        y = check_class(predictions[:, i])
        confusion_matrix[x][y] += 1

    return accuracy, confusion_matrix


def calculate_sensitivity(confusion_matrix):
    """
    Calculates sensitivity for each class.

    :param confusion_matrix: Confusion matrix.
    :return: Sensitivities for each class.
    """

    num_classes = confusion_matrix.shape[0]
    sensitivities = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        sensitivities[i] = true_positives / (true_positives + false_negatives)

    return sensitivities


def calculate_specificity(confusion_matrix):
    """
    Calculates specificity for each class.

    :param confusion_matrix: Confusion matrix.
    :return: Specificities for each class.
    """

    num_classes = confusion_matrix.shape[0]
    specificities = np.zeros(num_classes)

    for i in range(num_classes):
        true_negatives = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + \
                         confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
        specificities[i] = true_negatives / (true_negatives + false_positives)

    return specificities

