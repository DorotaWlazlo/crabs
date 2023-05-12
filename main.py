from data_preparation import load_file, data_scaling
from histogram_generation import generate_all_histograms


def main():
    lines = load_file('crabs.dat')
    #generate_all_histograms(lines)
    data_scaling(lines)


if __name__ == '__main__':
    main()


