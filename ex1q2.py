import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def load_data(input):
    return pd.read_csv(input)


def create_categorial_features(data_set, col):
    return pd.get_dummies(data_set[col], drop_first=True)


def remove_empty_data(data_set):
    data_set.dropna(inplace=True)


def concat_col(data_set, col):
    return pd.concat([data_set, col], axis=1)


def remove_cols(data_set, cols):
    return data_set.drop(cols, axis=1)


def get_col(data_set, col):
    return data_set[col]


def get_psuedo_inverse(matrix):
    return np.linalg.pinv(matrix)


def main():
    data_set = load_data("kc_house_data.csv")
    remove_empty_data(data_set)
    y = get_col(data_set, 'price')
    zip_code_cat_col = create_categorial_features(data_set, 'zipcode')
    data_set = remove_cols(data_set, ["id", "date", "zipcode"])
    data_set = concat_col(data_set, zip_code_cat_col)

    print(data_set.shape)

    pseudo_inv_mat = get_psuedo_inverse(data_set)
    w_hat = pseudo_inv_mat.dot(y)

    print(w_hat.shape)
    x_arr = []
    train_error = []
    test_error = []
    for x in range(1, 100):
        x_arr.append(x)
        train_data = data_set.sample(frac=x / 100)
        y = get_col(train_data, 'price')


        # remove price
        #((y_train - y_predicted) ** 2).mean()
        pass

    utils.plot_graph(x_arr, "X", train_error, "mse_train", test_error,
                     "mse_test")

if __name__ == '__main__': main()
