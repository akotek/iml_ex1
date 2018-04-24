import numpy as np
import pandas as pd
import utils


def load_data(input):
    return pd.read_csv(input)


def create_categorial_features(data_set, col):
    return pd.get_dummies(data=data_set, columns=col, drop_first=True)


def remove_empty_data(data_set):
    data_set.dropna(inplace=True)


def remove_cols(data_set, cols):
    return data_set.drop(cols, axis=1)


def get_col(data_set, col):
    return data_set[col]


def get_psuedo_inverse(matrix):
    return np.linalg.pinv(matrix)


def create_intercept(data_set, col_name):
    data_set.insert(0, col_name, 1)
    return data_set


def update_col(data_set, col, value):
    #data_set.loc[data_set[col] == 0, value] = data_set[value]
    data_set[col][data_set[col] == 0] = data_set[value]


def remove_outliers(data_set):
    return data_set[data_set.apply(lambda x: np.abs(x - x.mean()) /
                                                 x.std() < 4).all(axis=1)]


def pre_process_data(data_set):
    remove_empty_data(data_set)
    data_set = remove_cols(data_set, ["id", "date"])
    update_col(data_set, 'yr_renovated', 'yr_built')
    # a = data_set.apply(lambda y: np.abs(y - y.mean()) / y.std() < 4).all(
    #     axis=1)
    # print(a.type)
    data_set = create_intercept(data_set, 'w0')
    data_set = create_categorial_features(data_set, ['zipcode'])
   # data_set = remove_outliers(data_set)
    # data_set = create_categorial_features(data_set, 'zipcode')
    return data_set


def get_rmse(y_train, y_predicted):
    return np.sqrt(((y_train - y_predicted) ** 2).mean())


def main():
    data_set = load_data("kc_house_data.csv")
    data_set = pre_process_data(data_set)

    #print(data_set.head)
    # print(data_set.at[0,'yr_renovated'])
    # print(data_set.at[4,'yr_renovated'])
    # x_arr = []
    # train_error = []
    # test_error = []
    # for x in range(1, 100):
    #     x_arr.append(x)
    #     rows = np.random.rand(len(data_set)) < x / 100
    #     train = data_set[rows]
    #     test = data_set[~rows]
    #     y_train = get_col(train, 'price') #y price
    #     train = remove_cols(train, ['price'])
    #     w_train = np.dot(get_psuedo_inverse(train), y_train)
    #     y_hat_train = train.dot(w_train) #yHAT = Xw
    #
    #     y_test = get_col(test, 'price')
    #     test = remove_cols(test, ['price'])
    #     y_hat_test = test.dot(w_train)
    #     train_error.append(get_rmse(y_train, y_hat_train))
    #     test_error.append(get_rmse(y_test, y_hat_test))
    #
    # utils.plot_graph(x_arr, "X", train_error, "mse_train", test_error,
    # "mse_test")


if __name__ == '__main__':
    main()
