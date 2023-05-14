from data_preparation import load_file, class_convertion, data_scaling
from histogram_generation import generate_all_histograms


def main():
    lines = load_file('crabs.dat')
    lines_converted = class_convertion(lines)
    data_array = data_scaling(lines_converted)
    print(data_array[100])


if __name__ == '__main__':
    main()


