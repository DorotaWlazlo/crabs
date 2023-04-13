from matplotlib import pyplot as plt


def get_attribute(data_list, attribute_number):
    result = []
    for i in data_list:
        line = i.split()
        result.append(float(line[attribute_number]))
    return result


def draw_histogram(data, name, suffix):
    plt.clf()
    plt.hist(data)
    plt.xlabel("Value (mm)")
    plt.ylabel("Frequency")
    result = suffix.replace(' ', '_')
    plt.title(name)
    result = 'plots/' + result + '.png'
    plt.savefig(result)
    plt.close()


def generate(data_list, name):
    attribute_names = [" frontal lip of carapace", " rear width of carapace", " length along the midline of carapace",
                       " maximum width of carapace", " body depth"]
    suffix = [" FL", " RW", " CL", " CW", " BD"]
    for i in range(3, 8):
        j = i - 3
        long_name = name + attribute_names[j]
        short_name = name + suffix[j]
        attribute_values = get_attribute(data_list, i)
        draw_histogram(attribute_values, long_name, short_name)



def main():
    with open('crabs.dat', 'r') as f:
        message = f.read()
    f.close()
    lines = message.splitlines()
    generate(lines[1:51], "Blue male")
    generate(lines[51:101], "Blue female")
    generate(lines[101:151], "Orange male")
    generate(lines[151:201], "Orange female")


if __name__ == '__main__':
    main()


