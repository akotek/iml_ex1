import matplotlib.pyplot as plt


def plot_graph(title_x, y1, title_y1, y2=None, title_y2=None):
    if y2 is None and title_y2 is None:
        plot_one_func(title_x, y1,title_y1)
        return
    plt.title('Plotting {0} and {1} and {2}'.format(title_x, title_y1, title_y2), fontsize=8)
    plt.xlabel(title_x)
    plt.ylabel("{0} and {1}".format(title_y1, title_y2))
    plt.plot(y1, 'r')
    plt.plot(y2, 'b')
    plt.legend([title_y1, title_y2], loc='center right')
    plt.show()
    return


def plot_one_func(title_x, y1, title_y1):
    plt.title('Plotting {0} and {1}'.format(title_x, title_y1))
    plt.xlabel(title_x)
    plt.ylabel(title_y1)
    plt.plot(y1, 'b')

    plt.show()
    return


def plot_five_func(x_axis_list, y_func_list, x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    title_list = []
    for i in range(len(y_func_list)):
        plt.plot(x_axis_list, y_func_list[i])
        title_list.append("line " + str(i))
    plt.legend(title_list, loc='upper right')
    plt.show()
    return


def plot_images_from_list(image_list, k_list, forb_dist, comp_ratio):

    fig, axeslist = plt.subplots(ncols=3, nrows=2)
    fig.suptitle("5 Images Plot", fontsize=15)

    for i in range(len(image_list)):
        axeslist.ravel()[i].imshow(image_list[i], 'gray')
        axeslist.ravel()[i].set_title(
            "K: {0},\n Forb Dist: {1:.1f}, Comp Ratio: {2:.1f}".format(
                k_list[i],forb_dist[i], comp_ratio[i]), fontsize=10)
        axeslist.ravel()[i].set_axis_off()
    plt.show()

