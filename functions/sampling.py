import pandas as pd
from sklearn.model_selection import train_test_split


def split_testing_dataset(Clinic, test_size):
    # The Clinic dataset must contain the column to describe Responders/Non-Responders
    R = [i for i in Clinic.index if Clinic.loc[i, "Response"] == 1]
    NR = [i for i in Clinic.index if Clinic.loc[i, "Response"] == 0]
    dataset_R_train, dataset_R_test = train_test_split(Clinic.loc[R,:], test_size=test_size)
    dataset_NR_train, dataset_NR_test = train_test_split(Clinic.loc[NR,:], test_size=test_size)
    train_all = pd.concat([dataset_R_train,dataset_NR_train], join="inner", axis=0)
    test_all = pd.concat([dataset_R_test,dataset_NR_test], join="inner", axis=0)
    return train_all.index.tolist(), test_all.index.tolist()


def balance_subclass(df_train):
    if df_train[df_train['Response'] == 1].shape[0] > df_train[df_train['Response'] == 0].shape[0]:
        data_train_equal = pd.concat(
            [df_train.loc[df_train.loc[:, "Response"] == 1].sample(df_train[df_train['Response'] == 0].shape[0]),
             df_train.loc[df_train.loc[:, "Response"] == 0]], axis=0, join="inner")
    else:
        data_train_equal = pd.concat(
            [df_train.loc[df_train.loc[:, "Response"] == 0].sample(df_train[df_train['Response'] == 1].shape[0]),
             df_train.loc[df_train.loc[:, "Response"] == 1]], axis=0, join="inner")
    print('Balanced samples: {}'.format(data_train_equal.shape[0]))
    return data_train_equal
