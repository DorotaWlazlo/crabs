import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_file(file_name):
    """
    Loads a file and returns its content as a list of lines.

    :param file_name: The name of the file to load.
    :return: A list of lines from the file.
    """
    with open(file_name, 'r') as f:
        message = f.read()
    f.close()
    lines = message.splitlines()
    return lines


def class_conversion(lines):
    """
    Converts class labels in the lines list to a one-hot encoded format.

    :param lines: A list of lines representing data.
    :return: The lines list with class labels converted.
    """
    lines = lines[1:]  # deleting header
    i = 0
    for line in lines:
        if line[0] == 'B' and line[2] == 'M':
            lines[i] = '1 0 0 0 ' + line
        if line[0] == 'B' and line[2] == 'F':
            lines[i] = '0 1 0 0 ' + line
        if line[0] == 'O' and line[2] == 'M':
            lines[i] = '0 0 1 0 ' + line
        if line[0] == 'O' and line[2] == 'F':
            lines[i] = '0 0 0 1 ' + line
        i += 1
    return lines


def data_scaling(lines):
    """
    Scales numeric columns in the lines list to be in -1 to 1 range.

    :param lines: A list of lines representing data.
    :return: The array of lines with numeric columns scaled.
    """

    data = [line.split() for line in lines]
    array = np.array(data)
    columns_to_remove = [4, 5, 6]  # removing columns with letter labels and index
    array = np.delete(array, columns_to_remove, axis=1)
    array = array.astype(np.float64)
    numeric_columns = array[:, [4, 5, 6, 7, 8]]  # columns to scale

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(numeric_columns)
    scaled_columns = scaler.transform(numeric_columns)
    array[:, [4, 5, 6, 7, 8]] = scaled_columns

    return array


def splitting_into_training_and_test(array):
    """
    Splits the array into training and test data.

    :param array: A numpy array representing data.
    :return: Two numpy arrays, training_data and test_data.
    """
    training_data = np.empty((0, 9))
    test_data = np.empty((0, 9))
    split_arrays = np.array_split(array, 4)

    # splitting data from each class into test and train data sets
    for i in range(4):
        np.random.shuffle(split_arrays[i])
        total_rows = split_arrays[i].shape[0]
        train_rows = int(0.8 * total_rows)  # train data contains 80% of data of each class
        training_data = np.concatenate((training_data, split_arrays[i][:train_rows]), axis=0)
        test_data = np.concatenate((test_data, split_arrays[i][train_rows:]), axis=0)

    np.random.shuffle(training_data)
    np.random.shuffle(test_data)

    return training_data, test_data
