

def print_hi(name):
    with open('crabs.dat', 'r') as f:
        message = f.read()

    print(message)


if __name__ == '__main__':
    print_hi('PyCharm')


