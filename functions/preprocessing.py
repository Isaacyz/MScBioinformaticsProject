import numpy as np
import pandas as pd
import qnorm
from sklearn.preprocessing import MinMaxScaler

def log2_transformation(Exp_Dataset):
     return np.log2(Exp_Dataset + 1)

def Quantile_Normalization(Exp_Dataset):
    Exp_Dataset = qnorm.quantile_normalize(Exp_Dataset, axis=1)
    return Exp_Dataset

def Standardization(Exp_Dataset):
    index, column = Exp_Dataset.index, Exp_Dataset.columns
    Exp_Dataset = MinMaxScaler().fit_transform(Exp_Dataset)
    Exp_Dataset = pd.DataFrame(Exp_Dataset, columns=column, index=index)
    return Exp_Dataset

def Power(Exp_Dataset, times = 2):
    temp = input("Power the Expression dataset by 2?(Y/N): ")
    if temp == "N":
        times = input("Provide the number: ")
    print("Expression dataset is by {}".format(times))
    return Exp_Dataset**times

def preProcessing_Steps(Exp_Dataset, pre_processing_order):
    # The default order is gene following log2 transformations than quantile normalizing within samples and Standardized them into 0-1 scales
    # Also can set your own preprocessing order such as without log2 transformation
    if isinstance(pre_processing_order, list) and all([True if i in ["log2","Standardiaztion","QN","power"] else False for i in pre_processing_order]):
        for i in pre_processing_order:
            print(i)
            if i == "log2":
                Exp_Dataset = log2_transformation(Exp_Dataset)
            if i == "QN":
                Exp_Dataset = Quantile_Normalization(Exp_Dataset)
            if i == "Standardiaztion":
                Exp_Dataset = Standardization(Exp_Dataset)
            if i == "power":
                Exp_Dataset = Power(Exp_Dataset)
        return Exp_Dataset
    else:
        return print("pre_processing_order must be a list with valid values in ['log2','Standardiaztion','QN','power']")

