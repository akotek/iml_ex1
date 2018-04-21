import numpy as np
import pandas as pd
import utils


def load_data(input):
    return pd.read_csv(input)


def create_categorial_features(data_set, col):
    return pd.get_dummies(data_set, col, drop_first=True)


def remove_empty_data(data_set):
    data_set.dropna(inplace=True)


def remove_cols(data_set, cols):
    return data_set.drop(cols, axis=1)


def get_col(data_set, col):
    return data_set[col]


def get_psuedo_inverse(matrix):
    return np.linalg.pinv(matrix)


def create_intercept(data_set, col_name, value):
    data_set.insert(0, col_name, value)
    return data_set


def pre_process_data(data_set):
    remove_empty_data(data_set)
    data_set = create_categorial_features(data_set, 'zipcode')
    data_set = create_intercept(data_set, 'w0', 1)
    data_set = remove_cols(data_set, ["id", 'lat', 'long'])
    return data_set


def get_rmse(y_train, y_predicted):
    return ((y_train - y_predicted) ** 2).mean()


def main():
    data_set = load_data("kc_house_data.csv")
    data_set = pre_process_data(data_set)

    #todo check how date is removed??
    #todo improve score
    x_arr = []
    train_error = []
    test_error = []
    for x in range(1, 100):
        x_arr.append(x)
        rows = np.random.rand(len(data_set)) < x / 100
        train = data_set[rows]
        test = data_set[~rows]
        y_train = get_col(train, 'price')
        train = remove_cols(train, ['price'])
        y_test = get_col(test, 'price')
        test = remove_cols(test, ['price'])
        w_train = np.dot(get_psuedo_inverse(train), y_train)
        y_hat_train = train.dot(w_train)
        w_test = np.dot(get_psuedo_inverse(test), y_test)
        y_hat_test = test.dot(w_test)
        train_error.append(get_rmse(y_train, y_hat_train))
        test_error.append(get_rmse(y_test, y_hat_test))

    utils.plot_graph(x_arr, "X", train_error, "mse_train", test_error, "mse_test")


if __name__ == '__main__': main()
