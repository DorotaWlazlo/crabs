from neural_network import gradient_descend, initialize_data, test_predictions, calculate_sensitivity, \
    calculate_specificity
from matplotlib import pyplot as plt


def main():
    ni_list = [0.1, 0.01, 0.001]
    hidden_list = [5, 50, 500]
    # Initialize training and test data
    y_data_train, x_data_train, y_data_test, x_data_test = initialize_data('crabs.dat')

    for hidden in hidden_list:
        accuracies_list = []
        iterations = 0
        for ni in ni_list:
            # Perform gradient descent to train the neural network
            weights_hidden, weights_output, accuracies, iterations = \
                gradient_descend(x_data_train, y_data_train, 10000, ni, 5, hidden, 4)
            accuracies_list.append(accuracies)
            # Test the trained network on the test data
            accuracy, confusion_matrix = test_predictions(weights_hidden, weights_output, y_data_test, x_data_test)
            print(f'Parameters for ni = {ni} and number of neurons in hidden layer = {hidden}:')
            print(f'Accuracy of test data: {accuracy}')
            # Calculate sensitivity for each class
            sensitivity = calculate_sensitivity(confusion_matrix)
            print(f'Sensitivity of each class: {sensitivity}')
            # Calculate specificity for each class
            specificity = calculate_specificity(confusion_matrix)
            print(f'Specificity of each class: {specificity}')

        # Plot the change of accuracy over iterations
        fig = plt.figure()
        plt.plot(iterations, accuracies_list[0], color='r', label=ni_list[0])
        plt.plot(iterations, accuracies_list[1], color='b', label=ni_list[1])
        plt.plot(iterations, accuracies_list[2], color='c', label=ni_list[2])
        plt.xlabel('Iteracje')
        plt.ylabel('Dokładność')
        plt.title(f'Zmiana dokładności predykcji w czasie \n dla {hidden} neuronów w warstwie ukrytej')
        plt.legend()
        plt.savefig(f'plots/accuracy{hidden}.png')


if __name__ == '__main__':
    main()
