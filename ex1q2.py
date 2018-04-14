import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(input):
    return pd.read_csv(input)


def pre_process_data(data_set):
    data_set.dropna(inplace=True)
    data_set = data_set.drop(["id", "date", "price", "zipcode"], axis=1)

    return data_set


def create_categorial_features(data, col):
    return pd.get_dummies(data[col], drop_first=True)


def concat_col(data_set, col):
    return pd.concat([data_set, col], axis=1)


def get_col(data_set, col):
    return data_set[col]


def main():
    data_set = load_data("kc_house_data.csv")
    zip_code_cat_col = create_categorial_features(data_set, 'zipcode')
    y = get_col(data_set, 'price')
    data_set = pre_process_data(data_set)
    data_set = concat_col(data_set, zip_code_cat_col)  # add categorical zip code

    psuedo_inverse_mat = np.linalg.pinv(data_set)
    print(psuedo_inverse_mat.shape)

    # print(data_set.head)


if __name__ == '__main__': main()
