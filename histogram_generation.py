from matplotlib import pyplot as plt
from numpy import mean, std


def get_attribute(data_list, attribute_number):
    result = []
    for i in data_list:
        line = i.split()
        result.append(float(line[attribute_number]))
    return result


def draw_histogram(data, name, suffix, color):
    plt.clf()
    plt.hist(data, bins=15, color=color, ec="black")
    plt.xlabel("Value (mm)")
    plt.ylabel("Frequency")
    result = suffix.replace(' ', '_')
    plt.title(name)
    result = 'plots/' + result + '.png'
    plt.savefig(result)
    plt.close()


def generate(data_list, name, color):
    attribute_names = [" frontal lip of carapace", " rear width of carapace", " length along the midline of carapace",
                       " maximum width of carapace", " body depth"]
    suffix = [" FL", " RW", " CL", " CW", " BD"]
    results = []
    for i in range(3, 8):
        j = i - 3
        long_name = name + attribute_names[j]
        short_name = name + suffix[j]
        attribute_values = get_attribute(data_list, i)
        statistic = short_name + " Average: " + str(round(mean(attribute_values), 2)) + " Stan. dev.: " + \
                    str(round(std(attribute_values), 2))
        results.append(statistic)
        draw_histogram(attribute_values, long_name, short_name, color)
    return results


def generate_all_histograms(lines):
    print(generate(lines[1:51], "Blue male", "#00BFFF"))
    print(generate(lines[51:101], "Blue female", "#104E8B"))
    print(generate(lines[101:151], "Orange male", "#FFA500"))
    print(generate(lines[151:201], "Orange female", "#EE4000"))
