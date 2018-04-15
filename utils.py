import matplotlib.pyplot as plt


def plot_graph(x, title_x, y1, title_y1, y2=None, title_y2=None):
    plt.title('Plotting {0} and {1}'.format(title_x, title_y1))
    plt.xlabel(title_x)
    plt.ylabel(title_y1)
    plt.plot(x, y1)
    if y2 is not None and title_y2 is not None:
        plt.ylabel(title_y2)
        plt.plot(x, y2)
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
