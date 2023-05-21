
from neural_network import gradient_descend, initialize_data, test_predictions, calculate_sensitivity, \
    calculate_specificity
from matplotlib import pyplot as plt


def main():
    y_data_train, x_data_train, y_data_test, x_data_test = initialize_data('crabs.dat')
    weights_hidden, weights_output, accuracies, iterations = \
        gradient_descend(x_data_train, y_data_train, 1500, 0.01, 5, 5, 4)
    accuracy, confusion_matrix = test_predictions(weights_hidden, weights_output, y_data_test, x_data_test)
    print(f'Accuracy of test data: {accuracy}')
    sensitivity = calculate_sensitivity(confusion_matrix)
    print(f'Sensitivity of each class: {sensitivity}')
    specificity = calculate_specificity(confusion_matrix)
    print(f'Sensitivity of each class: {specificity}')
    plt.plot(iterations, accuracies)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Change of accuracy in iterations')
    plt.savefig('plots/accuracy.png')


if __name__ == '__main__':
    main()


