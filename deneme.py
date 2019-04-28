import requests
import json


data = {
    'MSSubClass':20 ,
    'OverallQual': 10,
    'LotArea':20000,
    'YrSold':2006,
    'OverallCond': 9,
    'YearBuilt':1900 ,S
    'ExterQual':'Ex',
    'CentralAir':'Y',
    'KitchenQual':'Ex',
    'FireplaceQu': 'Ex',
    'GarageQual': 'Ex',
    'GarageCond': 'Ex',
    'PoolQC': 'Ex',
    'WoodDeckSF': 180,
    'YearRemodAdd':1996
}

r = requests.post('http://192.168.43.198:5000/start', data={'data': json.dumps(data)})
print(r.text)
#****************************
'''
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
    predictions = rf.predict(df_sample)
    predictions = np.round(predictions, 0)
    return predictions #test_labels, 


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
    # the_most_occurred = {}
    # for count_key in count_dict:
    #     if count_dict[count_key] > the_most_occurred:
    #         the_most_occurred[]
            
def start_task(data):
    result = run(data)
    return result




def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))
                    ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
'''    
# data = {
#     'MSSubClass':20 ,
#     'OverallQual': 10,
#     'LotArea':20000,
#     'YrSold':2006,
   
#     'OverallCond': 9,
#     'YearBuilt':1900 ,
#     'ExterQual':'Ex',
#     'CentralAir':'Y',
#     'KitchenQual':'Ex',
#     'FireplaceQu': 'Ex',
#     'GarageQual': 'Ex',
#     'GarageCond': 'Ex',
#     'PoolQC': 'Ex',
#     'WoodDeckSF': 180,
#     'YearRemodAdd':1996
# }

# data = {
#     'MSSubClass': {
#         'asd': 1,
#         'asdadf': 3,
#         'asd12': 6
#     },
#     'OverallQual': 2
# }

# print(data['MSSubClass'].keys().split())


#test_labels, 
# predictions = start(data)
# print(predictions)

#print("Logaritmic Mean squared error: %.2f"
#   % rmsle(test_labels, predictions))
