from data_preparation import load_file, class_conversion, data_scaling, splitting_into_training_and_test
from histogram_generation import generate_all_histograms


def main():
    lines = load_file('crabs.dat')
    lines_converted = class_conversion(lines)
    data_array = data_scaling(lines_converted)
    print(data_array[100])
    print(data_array.shape)
    print(data_array.size)
    training_data, test_data = splitting_into_training_and_test(data_array)
    print(training_data.shape)
    print(test_data.shape)
    print(training_data)
    print(test_data.shape)
    print(test_data)

if __name__ == '__main__':
    main()


