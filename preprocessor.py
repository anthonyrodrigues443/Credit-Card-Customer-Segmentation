import numpy as np 
import pandas as pd
import pickle

with open('preprocessing_values/preprocessing_data.pkl', 'rb') as pickle_file:
    loaded_dicts = pickle.load(pickle_file)

drop_cols, mean_dict, log_cols = loaded_dicts[0], loaded_dicts[1], loaded_dicts[2]

def col_dropper(data, cols = drop_cols):
    try : 
        transformed_data = data.drop(columns=cols).copy()
    except KeyError:
         transformed_data = data.copy()
    return transformed_data

def null_filler(data, mean_dict = mean_dict):
    transformed_data = data.copy()
    for col_mean in mean_dict.items():
        transformed_data[col_mean[0]].fillna(col_mean[1], inplace=True)
    return transformed_data

def log_transformer(data,cols = log_cols):
    transformed_data = data.copy()
    for col in log_cols :
            transformed_data[col] = np.log(1 + transformed_data[col])
    return transformed_data

def preprocessor(data):
    data = col_dropper(data)
    data1 = null_filler(data)
    data2 = log_transformer(data1)
    return data2