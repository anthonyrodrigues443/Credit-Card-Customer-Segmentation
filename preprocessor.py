import numpy as np 
import pandas as pd
import pickle

wit

def col_dropper(data, cols=['CUST_ID']):
    transformed_data = data.drop(columns=cols).copy()
    return transformed_data

def null_filler(data, col=['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']):
    transformed_data = data.copy()
    for i in col:
        transformed_data[i].fillna(transformed_data[i].median(), inplace=True)
    return transformed_data

def log_transformer(data,cols = log_cols):
    transformed_data = data.copy()
    for col in log_cols:
            transformed_data[col] = np.log(1 + transformed_data[col])
    return transformed_data

def preprocessor(data):
    data = col_dropper(data)
    data1 = null_filler(data)
    data2 = log_transformer(data1)
    data3 = col_dropper(data2, cols=['CASH_ADVANCE_FREQUENCY', 'INSTALLMENTS_PURCHASES', 'PURCHASES_FREQUENCY'])
    return data3