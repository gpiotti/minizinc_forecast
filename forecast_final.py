import csv
import datetime
datetime.datetime.strptime
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import numpy.ma as ma

from datetime import datetime
from datetime import timedelta


train_start_date = datetime.strptime('01/01/2017  00:00', '%m/%d/%Y %H:%M')
predict_start_date = datetime.strptime(sys.argv[1], '%m/%d/%Y')
predict_end_date = predict_start_date + timedelta(days=6, hours=23, minutes=30)
train_end_date = predict_start_date - timedelta(weeks=2)


# 12/18/2017
# 12/25/2017
# 01/01/2018
# 01/08/2018

start_date = train_start_date
#train_end_date =  datetime.strptime('02/12/2018 23:30', '%m/%d/%Y %H:%M')

#train_start_date = train_end_date - timedelta(days=365)

##############################################################################
#predict_start_date = datetime.strptime('02/26/2018 00:00', '%m/%d/%Y %H:%M')
#predict_end_date = datetime.strptime('03/04/2018 23:30', '%m/%d/%Y %H:%M')
#############################################################################
    
print("train start: ", train_start_date)
print("train end: ", train_end_date)
print("predict start: ", predict_start_date)
print("predict end:", predict_end_date)

result = pd.DataFrame()
text = 'type_'
X_train = None
X_predict = None
y_train = None
y_predict = None
input_type = 1
max_input_type = 12


def create_dummies(data_train, data_predict):
    train_start = min(data_train.index)
    train_end = max(data_train.index)
    predict_start = min(data_predict.index)
    predict_end = max(data_predict.index)
    
    data_train.reset_index(drop=True)

    data_predict.reset_index(drop=True)
    concat = pd.concat([data_train, data_predict], axis=0, ignore_index=False)
    
    dataset_drop_first = pd.get_dummies(concat,
                            columns = [
                                
                              'timeblock',
                              'dow',
                              #'month',
                              #'month_week',
                              #'peak_month',
                              'dom',
                              #'week',                            
                              'type',
                              # 'week'
                            ],
                           drop_first=True).copy()


    dataset_drop_first.reset_index(drop=True)
    data_with_dummies = dataset_drop_first
    
    return data_with_dummies[train_start:train_end],  data_with_dummies[predict_start:predict_end]


# Cargar dataset


cache = {}

def cached_date_parser(s):
    if s in cache:
        return cache[s]
    dt = pd.to_datetime(s, format='%m/%d/%y %H:%M')
    cache[s] = dt
    return dt

dataset_original = pd.read_csv('dataset.csv',#pd.read_csv('new/dataset_april_with_outliers.csv',
                 
                 index_col= ['class_date'],
                 parse_dates=[0],
                 date_parser=cached_date_parser)



def filter_dataset(input_type):

    dataset_train = None
    new_dataset = None
    X_train = None
    X_predict = None
    y_train = None
    y_predict = None
    
    dataset_train = pd.DataFrame(dataset_original[dataset_original['type'] == input_type])

    dataset_train = dataset_train[[
    'dom',
     # 'q_py',
     'q_avg_14',
     #  week',
    'timeblock', 
    'q_current',
    'dow',
    'month',
    'peak_month',
    #'month_week',
    'type',
    'level_category', 'language', 'course_type'
    ]].copy()


    new_dataset = dataset_train[[
    'q_avg_14',
     #'q_py',
    'dom',
    'timeblock', 
        
    'dow',
    'month',
    #'week',
    'peak_month',
    # 'month_week',
    'type', 
    ]].copy()
    
    return new_dataset, dataset_train





#SCALE input data
def scale_and_train(X_train, y_train, X_predict, y_predict):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    minmax = MinMaxScaler(feature_range=(0, 1))
  
    minmax.fit(X_train)
    X_train = minmax.transform(X_train) 
  
    X_predict = minmax.transform(X_predict)   
    minmax = MinMaxScaler(feature_range=(0, 1))
    
    #TRAIN
  
    from sklearn.neural_network import MLPRegressor
    
    layers = 15
    alpha = 0.001
    learning_rate_i = 0.001
    learning_rate_type = 'adaptive'
    solver='sgd'
    
    if input_type  == 7:
        layers = (6,6)
        alpha = 0.001
    elif input_type == 5:
        layers = (20)
        alpha = 0.01
    elif input_type == 4:
        layers = (48, 48, 48, 48, 48, 48, 48)
        alpha = 0.0001
        learning_rate = 0.0001
        learning_rate_type = 'constant'
        solver='adam'
    elif input_type == 6:
        layers = (20)
        alpha = 2
    elif input_type == 9:
        layers=20
        alpha = 0.001
    
   
    
    model_neural = MLPRegressor(alpha=alpha,
                                early_stopping =False,validation_fraction =0.20,
                                    solver=solver,
                                    learning_rate = learning_rate_type,
                                    learning_rate_init = learning_rate_i,
                                    hidden_layer_sizes=layers,
                                    max_iter = 1500,
                                    verbose =False,
                                    random_state =1,
                                    tol=0.00000001,
                                    #power_t=0.5,
                                    activation='tanh',shuffle=True)
    
    model_neural.fit(X_train, y_train['q_current'].ravel())


    #PREDICT


    if input_type in (2,4,7,8,11,12):
        prediction = pd.DataFrame(np.maximum(np.round(model_neural.predict(X_predict),0), 0.) )
    
        
    else:
        prediction = pd.DataFrame(np.maximum(np.floor(model_neural.predict(X_predict)), 0.) )
        


    prediction = prediction.rename(columns={0:'q_predicted'})
    prediction.index =  data_predict.index
    
    return model_neural, prediction, X_train, X_predict

def export_data():
    start_date = predict_start_date.strftime('%m%d%Y')
    end_date =  predict_end_date.strftime('%m%d%Y')
    prediction_to_export = pd.concat([dataset_train[predict_start_date:predict_end_date],  prediction['q_predicted']], axis=1)

    del prediction_to_export['timeblock']
    del prediction_to_export['dow']
    del prediction_to_export['month']
    del prediction_to_export['dom']
      
    #prediction_to_export.to_csv(text + start_date + '_' + end_date +'.csv')
    
    print (text + start_date + '_' + end_date )
    print("total requests: ", sum(prediction_to_export['q_predicted']))

##    with open(r'scores.csv', 'a') as f:
##        fields=[predict_start_date.strftime('%m-%d-%Y'),
##                predict_end_date.strftime('%m-%d-%Y'),
##                model_neural.score(X_predict, y_predict),
##               text]
##        writer = csv.writer(f)
##        writer.writerow(fields)
    print("train score: ", model_neural.score(X_train, y_train))

    return prediction_to_export
    
    
 
while input_type <= max_input_type:
    data_train= None
    data_predict = None
    X_train = None
    y_train = None
    X_predict = None
    y_predict = None
    model_neural = None
    
    text = 'type_' + str(input_type) + "_"
    new_dataset, dataset_train  = filter_dataset(input_type)
    
    # create datasets
    data_train_original =  new_dataset[train_start_date:train_end_date]


    data_train, data_predict =  create_dummies(
        new_dataset[train_start_date:train_end_date],
        new_dataset[predict_start_date:predict_end_date],
        )

    target_name = ['q_current']

    # prepare inputs X_train, y_train, X_predict, y_predict

    data_train['q_current'] = dataset_train[train_start_date:train_end_date]['q_current'].values
    data_train.index =  dataset_train[train_start_date:train_end_date].index

    data_predict.index =  dataset_train[predict_start_date:predict_end_date].index

    X_train =  data_train
    y_train = data_train [target_name]

    X_predict = data_predict

    y_predict = pd.DataFrame(dataset_train[predict_start_date:predict_end_date]['q_current'].values)
    y_predict.index =    dataset_train[predict_start_date:predict_end_date].index
    y_predict = y_predict.rename(columns={0:'q_current'})
 
    del X_train['q_current']
   
    #model_neural = None
    prediction = None
    model_neural, prediction, X_train, X_predict = scale_and_train(X_train, y_train, X_predict, y_predict)
    to_export = export_data()
    result = pd.concat([result, to_export])
    start_date = start_date + timedelta(weeks=1)
   
    input_type += 1
    
result.to_csv('forecasted_requests.csv')

