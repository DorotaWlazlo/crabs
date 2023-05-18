import numpy as np


def load_file(file_name):
    with open(file_name, 'r') as f:
        message = f.read()
    f.close()
    lines = message.splitlines()
    return lines


def class_conversion(lines):
    lines = lines[1:]
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

    data = [line.split() for line in lines]
    array = np.array(data)
    columns_to_remove = [4, 5, 6]
    array = np.delete(array, columns_to_remove, axis=1)
    numeric_columns = [4, 5, 6, 7, 8]
    array = array.astype(np.float64)

    for col in numeric_columns:
        min_val = np.min(array[:, col])
        max_val = np.max(array[:, col])
        array[:, col] = (array[:, col] - min_val) / (max_val - min_val) * 2 - 1

    return array


def splitting_into_training_and_test(array):
    training_data = np.empty((0, 9))
    test_data = np.empty((0, 9))
    split_arrays = np.array_split(array, 4)
    for i in range(4):
        np.random.shuffle(split_arrays[i])
        total_rows = split_arrays[i].shape[0]
        train_rows = int(0.8 * total_rows)
        training_data = np.concatenate((training_data, split_arrays[i][:train_rows]), axis=0)
        test_data = np.concatenate((test_data, split_arrays[i][train_rows:]), axis=0)

    np.random.shuffle(training_data)
    np.random.shuffle(test_data)

    return training_data, test_data
