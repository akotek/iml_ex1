import numpy as np
import utils


def calc_mean(toss_list):
    mean_list = [toss_list[0]]
    for m in range(1, len(toss_list)):
        mean_list.append(np.mean(toss_list[0:m]))
    return mean_list


def main():
    data = np.random.binomial(1, 0.25, (100000, 1000))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    m_list = list(range(1,1000))

    # -------- 1st Plot --------
    # x_mean_list = []
    # for m in range(5):
    #     x_mean_list.append(calc_mean(data[m]))
    #
    # utils.plot_five_func(m_list, x_mean_list, 'm', 'mean_of_m')
    # -------------------------

    chebyshev_list = []
    hoeffding_list = []
    for eps in epsilon:
        for m in range(5):
            chebyshev_list.append(1 / (4*(m+1)*eps*eps))
            hoeffding_list.append(2*np.exp(-2*(m+1)*eps*eps))
        utils.plot_graph(m_list, "m", chebyshev_list, "chebyshev",
                         hoeffding_list, "hoeffding")
    # -------- 2nd Plot --------



if __name__ == '__main__':
    main()
