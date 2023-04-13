from matplotlib import pyplot as plt


def get_attribute(data_list, attribute_number):
    result = []
    for i in data_list:
        line = i.split()
        result.append(line[attribute_number])
    return result


def draw_histogram(data, name):
    plt.hist(data, 15)
    plt.xlabel("Wartość (mm)")
    plt.ylabel("Częstość")
    result = name.replace(' ', '_')
    plt.title(name)
    result = 'plots/' + result + '.png'
    plt.savefig(result)


def generate(data_list, name):
    attribute_names = [" frontal lip of carapace", " rear width of carapace", " length along the midline of carapace",
                       " maximum width of carapace", " body depth"]
    for i in range(3, 8):
        name = name + attribute_names[i-3]
        attribute_values = get_attribute(data_list, i)
        draw_histogram(attribute_values, name)




def print_hi(name):
    with open('crabs.dat', 'r') as f:
        message = f.read()
    f.close()
    lines = message.splitlines()
    blue_male_fl = get_attribute(lines[1:51], 3)
    blue_male_rw = get_attribute(lines[1:51], 4)
    blue_male_cl = get_attribute(lines[1:51], 5)
    blue_male_cw = get_attribute(lines[1:51], 6)
    blue_male_bd = get_attribute(lines[1:51], 7)

    print(blue_male_bd)
    print(len(blue_male_fl))


if __name__ == '__main__':
    print_hi('PyCharm')


