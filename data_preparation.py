import numpy as np


def load_file(file_name):
    with open(file_name, 'r') as f:
        message = f.read()
    f.close()
    lines = message.splitlines()
    return lines


def data_scaling(lines):
    lines = lines[1:]

    data = [line.split() for line in lines]
    array = np.array(data)
    numeric_columns = [1, 2, 3, 4, 5]
    columns_to_remove = [0, 1]
    array = np.delete(array, columns_to_remove, axis=1)
    array = array.astype(np.float64)

    for col in numeric_columns:
        min_val = np.min(array[:, col])
        max_val = np.max(array[:, col])
        array[:, col] = (array[:, col] - min_val) / (max_val - min_val) * 2 - 1

    print(array)


def float_converter(value):
    return float(value)

