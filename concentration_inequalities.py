import numpy as np
import math
import utils
import matplotlib.pyplot as plt


def calc_mean(toss_list):
    mean_list = [toss_list[0]]
    for m in range(1, len(toss_list)):
        mean_list.append(np.mean(toss_list[0:m]))
    return mean_list


def calc_cheby(eps, m):
    res = 1 / (4 * m * (eps ** 2))
    if res > 1:
        return 1
    return res


def calc_hoefding(eps, m):
    res = 2 * math.exp(-2 * m * (eps ** 2))
    if res > 1:
        return 1
    return res


def main():
    data = np.random.binomial(1, 0.25, (100000, 1000))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    m_list = list(range(1000))

    # -------- 1st Plot --------
    x_mean_list = []
    for m in range(5):
        x_mean_list.append(calc_mean(data[m]))

    utils.plot_five_func(m_list, x_mean_list, 'm', 'mean_of_m')
    # -------------------------

    # -------- 2/3nd Plot --------
    P = 0.25
    for eps in epsilon:
        chebyshev_list = []
        hoeffding_list = []
        bad_percentage = []

        cumsum_arr = data.cumsum(axis=1)

        for m in range(1000):
            chebyshev_list.append(calc_cheby(eps, m + 1))
            hoeffding_list.append(calc_hoefding(eps, m + 1))

            bad_data_counter = 0
            for seq in range(100000):
                if np.fabs((cumsum_arr[seq][m] / (m + 1)) - P) >= eps:
                    bad_data_counter += 1

            bad_percentage.append(bad_data_counter / 100000)

        # utils.plot_graph(m_list, "m", chebyshev_list, "chebyshev",
        #                  hoeffding_list, "hoeffding")
        plt.plot(m_list, chebyshev_list, label="Chebyshev bound")
        plt.plot(m_list, hoeffding_list, label="Hoeffding bound")
        plt.plot(m_list, bad_percentage, label="Bad percentage")
        plt.legend()
        plt.show()
    # -------------------------


if __name__ == '__main__':
    main()
