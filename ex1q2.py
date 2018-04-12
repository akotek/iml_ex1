import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def load_data(input):
    return pd.read_csv(input)


def pre_process_data(data_set):

    data_set = data_set.dropna()
    data_set = data_set.drop(["id", "date", "price", "zipcode"], axis=1)
    # axis1 is COL

    return data_set



def create_categorial_features(data, col):
    return pd.get_dummies(data[col], drop_first=True)


def main():
    data_set = load_data("kc_house_data.csv")
    cat_feat_mat = create_categorial_features(data_set, 'zipcode')
    data_set = pre_process_data(data_set)
    data_set = pd.concat([data_set, cat_feat_mat], axis=1) # add categorial
    # zip code to data


    print(data_set.shape)
    # data_set = pre_process_data(data_set)
    # cols will be different values in zip code, rows will sign 0/1 (exist,not)
    # categorical only on numeric vals so date is not relevant
    # views is not categorical as well - it has meaning to it values (more
    # views, less value?? )




if __name__ == '__main__': main()