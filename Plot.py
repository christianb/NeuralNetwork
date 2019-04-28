import matplotlib.pyplot as plt


def plot(x_list, y_list, x_name, y_name, title):
    plt.plot(x_list, y_list, label=title)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    #plt.title(title)
    #plt.savefig(out_file)

    pass

def plot_list(list, x_name, y_name, title, out_file):
    plt.plot(*list)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.title(title)
    plt.savefig(out_file)
    pass