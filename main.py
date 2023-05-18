
from neural_network import gradient_descend, initialize_data


def main():
    y_data_train, x_data_train, y_data_test, x_data_test = initialize_data('crabs.dat')

    weights_hidden, weights_output = gradient_descend(x_data_train, y_data_train, 500, 0.01, 5, 5, 4)

    print(weights_hidden)
    print(weights_output)


if __name__ == '__main__':
    main()


