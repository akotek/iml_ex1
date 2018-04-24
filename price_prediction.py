import numpy as np
import pandas as pd
import utils


def load_data(input):
    return pd.read_csv(input)


def create_categorial_features(df, col):
    return pd.get_dummies(data=df, columns=col, drop_first=True)


def remove_data(df):
    df.drop(df[(df.price <= 0) | (df.bedrooms <= 0) |
               (df.bathrooms <= 0) | (df.sqft_living <= 0) |
               (df.sqft_lot <= 0) | (df.floors <= 0) |
               (df.condition <= 0) | (df.grade <= 0) |
               (df.sqft_above <= 0) | (df.sqft_basement <= 0)].index,
            inplace=True)
    df.dropna(inplace=True)


def remove_cols(df, cols):
    return df.drop(cols, axis=1)


def get_col(df, col):
    return df[col]


def get_psuedo_inverse(matrix):
    return np.linalg.pinv(matrix)


def create_intercept(df, col_name):
    df.insert(0, col_name, 1)
    return df


def update_col(df, col, value):
    df[col][df[col] == 0] = df[value]


def pre_process_data(df):
    remove_data(df)
    df = remove_cols(df, ["id", "date"])
    update_col(df, 'yr_renovated', 'yr_built')
    df = create_intercept(df, 'w0')
    df = create_categorial_features(df, ['zipcode'])
    return df


def get_rmse(y_train, y_predicted):
    return np.sqrt(((y_train - y_predicted) ** 2).mean())


def main():
    df = load_data("kc_house_data.csv")
    df = pre_process_data(df)

    train_error = []
    test_error = []
    for x in range(1, 100):
        rows = np.random.rand(len(df)) < x / 100
        train = df[rows]
        test = df[~rows]
        y_train = get_col(train, 'price')
        train = remove_cols(train, ['price'])
        w_train = np.dot(get_psuedo_inverse(train), y_train)
        y_hat_train = train.dot(w_train)

        y_test = get_col(test, 'price')
        test = remove_cols(test, ['price'])
        y_hat_test = test.dot(w_train)

        train_error.append(get_rmse(y_train, y_hat_train))
        test_error.append(get_rmse(y_test, y_hat_test))

    utils.plot_graph("X", train_error, "mse_train", test_error,
                     "mse_test")

if __name__ == '__main__':
    main()
