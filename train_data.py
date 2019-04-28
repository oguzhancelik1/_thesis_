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


def train_module():
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

    labels = np.array(df_train['SalePrice'])
    df_train_x = df_train.drop(['SalePrice', 'Id'], axis=1)

    # train_df_train, test_df_train, train_labels, test_labels = train_test_split(
        # df_train_x, labels, test_size=0.25, random_state=42)
    # print(type(test_labels))
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Train the model on training data
    # rf.fit(train_df_train, train_labels)
    rf.fit(df_train_x, labels)
    
    #pickle.dump(rf, open("model.pkl","wb"))
    

    saved_model = "saved_model.pk1"  
    with open(saved_model, 'wb') as file:  
        pickle.dump(rf, file)
    #///////////////////////////////////////////////////////////////////////
    # temp2_set = train_set.drop(['SalePrice','Id'], axis=1)
    
train_module()    
    