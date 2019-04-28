import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from statistics import mean
from sklearn import linear_model
from numpy.polynomial.polynomial import polyfit
from sklearn.ensemble import RandomForestRegressor
import pickle

def predict_price(data):
    # data = process_data(data)

    train_set = pd.read_csv("train.csv")
    train_set['YAS'] = train_set['YrSold'] - train_set['YearBuilt']
    train_set["GARAJ_YAS"] = train_set["YrSold"] - train_set["GarageYrBlt"]
    train_set["SINCEREMOD"] = train_set["YrSold"] - train_set["YearRemodAdd"]

    obj_cols = []
    num_cols = []
    for col in train_set:

        # get dtype for column
        #dt = train_set[col].dtype
        # index=index+1
        # check if it is a number
        if train_set[col].dtype == 'object':
            train_set[col] = train_set[col].fillna('YOK')
            obj_cols.append(col)
        # print(col)
            # imputer=SimpleImputer(strategy="constant",fill_value="yok")
            # train_set[col]=imputer.fit_transfrom(train_set[col])
            # Y[:,index:index+1]=imputer.transform(Y[:,index:index+1])
            # df6=pd.DataFrame(Y)
        if train_set[col].dtype != 'object':
            train_set[col] = train_set[col].fillna(round(train_set[col].mean(), 0))
            num_cols.append(col)

    temp_train = train_set.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl',
                                'Heating', 'LowQualFinSF', 'KitchenAbvGr', '3SsnPorch',
                                'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd'], axis=1)
    # bir değerin %95 üzerinde tekrar ettiği kolonlar ve garaj yapım yılı + konut yapım yılı + REMOD ÜZERİNE GEÇEN SÜRE(SINCEREMOD)
    #print(obj_cols)
    #print(num_cols)
    df_train = pd.get_dummies(temp_train)
    df_train.MSSubClass = df_train.MSSubClass.astype(object)
    df_train.MoSold = df_train.MoSold.astype(object)
    df_train.YrSold = df_train.YrSold.astype(object)
    df_train = pd.get_dummies(df_train)

    col_list = np.array(df_train.columns)

    
    
    
    # temp2_set = train_set.drop(['SalePrice','Id'], axis=1)
    
    data = process_data(data, train_set, obj_cols, num_cols)
    test_df_train = pd.DataFrame(data, index=['0'])
   
    test_df_train['YAS'] = test_df_train['YrSold'] - test_df_train['YearBuilt']
    test_df_train["GARAJ_YAS"] = test_df_train["YrSold"] - test_df_train["GarageYrBlt"]
    test_df_train["SINCEREMOD"] = test_df_train["YrSold"] - test_df_train["YearRemodAdd"]
    #print(test_df_train)
    temp1_train = test_df_train.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl',
                                'Heating', 'LowQualFinSF', 'KitchenAbvGr', '3SsnPorch',
                                'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd'], axis=1)
    
    df_test = pd.get_dummies(temp1_train)
    df_test.MSSubClass = df_test.MSSubClass.astype(object)
    df_test.MoSold = df_test.MoSold.astype(object)
    df_test.YrSold = df_test.YrSold.astype(object)
    df_test = pd.get_dummies(df_test)
    
    sample_list = np.array(df_test.columns)
    df_sample=pd.DataFrame()
   
    for col in col_list:
        if col in sample_list:
            df_sample[col]=df_test[col]
        else:
            df_sample[col]=0
    df_sample=df_sample.drop(['SalePrice', 'Id'], axis=1)
    
    saved_model = "saved_model.pk1"  
    with open(saved_model, 'rb') as file:  
        load_model = pickle.load(file)
    #load_model = pickle.load(open("model.pkl","r"))
    
    predictions = load_model.predict(df_sample)

    predictions = np.round(predictions, 0)
    return predictions  


def process_data(data, train_set, obj_cols, num_cols):
    train_set_dict = train_set.to_dict()
    # first_row_dict = train_set.iloc[[0]].to_dict()
    for col in train_set_dict:
        if col not in data:
            if col in obj_cols:
                the_most_occurred = find_the_most_occurred(train_set_dict, col)
                data[col] = the_most_occurred
            elif col in num_cols:
                mean = find_mean(train_set_dict, col)
                data[col] = mean
    return data


def find_mean(train_dict, column_name):
    # mean = round(list(train_dict[column_name].values()).mean(), 0)
    mean_value = round(mean(list(train_dict[column_name].values())), 0)
    return mean_value




def find_the_most_occurred(train_dict, column_name):
    count_dict = {}
    for value in train_dict[column_name]:
        if train_dict[column_name][value] not in count_dict:
            count_dict[train_dict[column_name][value]] = 0
        else:
            count_dict[train_dict[column_name][value]] += 1

    the_most_occurred = max(count_dict,key=count_dict.get)
    return the_most_occurred

data = {
     'MSSubClass':60 ,
     'OverallQual': 10,
     'LotArea':20000,
     'YrSold':2006,
   
     'OverallCond': 10,
     'YearBuilt':1990 ,
     'ExterQual':'Ex',
     'CentralAir':'Y',
     'KitchenQual':'Ex',
     'FireplaceQu': 'Ex',
     'GarageQual': 'NA',
     'GarageCond': 'NA',
     'PoolQC': 'Ex',
     'WoodDeckSF': 180,
     'YearRemodAdd':1996
 }


predictions = predict_price(data)
print(predictions)





