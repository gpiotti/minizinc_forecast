
# coding: utf-8

# In[1]:


#%matplotlib inline
get_ipython().magic(u'matplotlib notebook')
import csv
import datetime
datetime.datetime.strptime
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


np.set_printoptions(precision=5, suppress=True)
plt.style.use('default')

plt.rcParams['agg.path.chunksize'] = 10000



# In[2]:


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
                                     'month',
                                     'dom',
                              'week',
                                'type',
                            ],
                           drop_first=True).copy()




    dataset_drop_first.reset_index(drop=True)
    data_with_dummies = dataset_drop_first
    
    return data_with_dummies[train_start:train_end],  data_with_dummies[predict_start:predict_end]


# Cargar dataset

# In[3]:


cache = {}

def cached_date_parser(s):
    if s in cache:
        return cache[s]
    dt = pd.to_datetime(s, format='%m/%d/%y %H:%M')
    cache[s] = dt
    return dt

dataset_original = pd.read_csv('new/dataset_no_outliers_april.csv',
                 index_col= ['class_date'],
                 parse_dates=[0],
                 date_parser=cached_date_parser)






# In[4]:


dataset_train = None
dataset_train = dataset_original




# In[5]:


dataset_train = dataset_train[[
'dom',
  'week',
'timeblock', 
'q_current',
'dow',
'month',
'type',
    'level_category', 'language', 'course_type'
]].copy()



# In[6]:



# train_start_date = '01/01/2017 00:00'
# train_end_date = '12/31/2017 23:30:00'

# predict_start_date = '01/08/2018 00:00'
# predict_end_date = '01/14/2018 23:30:00'



new_dataset = dataset_train[[
    'dom',
  'week',
    'timeblock',  
    'dow',
    'month',
    'type',
]].copy()





# In[ ]:


#SCALE input data
def scale_and_train(X_train, y_train, X_predict, y_predict):
    from sklearn.preprocessing import StandardScaler 
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    
    #scaler = StandardScaler()  
    #scaler.fit(X_predict)
    X_predict = scaler.transform(X_predict)

 
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics.scorer import make_scorer

    #TRAIN
    def my_custom_loss_func(ground_truth, pred):
        
        error =  0
        for i in range(0, len(ground_truth)):
            error_i = np.maximum(abs(ground_truth['q_current'][i] - pred[i]), 0.)
            error += error_i
        return error
        
        
    
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    



    model_neural = MLPRegressor(alpha= 0.01,
                                    solver='sgd',
                                    hidden_layer_sizes=(1),
                                    max_iter = 1000,
                                    verbose =False,
                                    random_state =1,
                                    tol=0.00001,
                                    learning_rate = 'adaptive',
                                    activation='tanh')
    
    from sklearn.model_selection import GridSearchCV
    parameters = {'hidden_layer_sizes':[11,20]}
    clf = GridSearchCV(model_neural, parameters,   )
    model_neural = clf.fit(X_train, y_train)
    
    #model_neural = clf.fit(X_predict, y_predict)
    
  
    print(clf.best_params_ )
    
    #model_neural.fit(X_train, y_train['q_current'])
    #model_neural = clf.fit(X_train, y_train['q_current'])


    #PREDICT

    prediction = pd.DataFrame(model_neural.predict(X_predict))
    prediction = pd.DataFrame(np.maximum(np.round(model_neural.predict(X_predict),0), 0.) )


    prediction = prediction.rename(columns={0:'q_predicted'})
    prediction.index =  data_predict.index
    
    return model_neural, prediction, X_train, X_predict

def export_data():
    start_date = predict_start_date.strftime('%m%d%Y')
    end_date =  predict_end_date.strftime('%m%d%Y')
    prediction_to_export = pd.concat([dataset_train[predict_start_date:predict_end_date],  prediction['q_predicted']], axis=1)


    del prediction_to_export['timeblock']
    del prediction_to_export['week']
    del prediction_to_export['dow']
    del prediction_to_export['month']
    del prediction_to_export['dom']
    
    text = 'new_scale_6,4_nodes_with_week_test_april_neural_prediction'
    print  (model_neural.score(X_predict, y_predict))
    prediction_to_export.to_csv(text + start_date + '_' + end_date +'.csv')
    print ("saved: " + text + start_date + '_' + end_date +'.csv')

    with open(r'scores.csv', 'ab') as f:
        fields=[predict_start_date.strftime('%m-%d-%Y'),
                predict_end_date.strftime('%m-%d-%Y'),
                model_neural.score(X_predict, y_predict),
               text]
        writer = csv.writer(f)
        writer.writerow(fields)
    print(model_neural.score(X_predict, y_predict))
    

 
    #graph = prediction_to_export [prediction_to_export['type'] == 1]
    #plt.plot(graph.index, graph['q_current'],'-', alpha = 1, color='red')
    #plt.plot(graph.index, graph['q_predicted'],'-', alpha=0.6, color = 'green')

    #plt.show();
  
    #plt.savefig('prediction_' +  start_date + '_' + end_date +'.png')


# In[ ]:


from datetime import datetime
from datetime import timedelta

start_date = datetime.strptime('01/01/2017 00:00', '%m/%d/%Y %H:%M')
last_date =  datetime.strptime('01/01/2018  00:00', '%m/%d/%Y %H:%M')


#train_start_date = datetime.strptime('01/01/2017  00:00', '%m/%d/%Y %H:%M')
train_end_date =  datetime.strptime('12/31/2017 23:30', '%m/%d/%Y %H:%M')
train_start_date = train_end_date - timedelta(days=365)
predict_start_date = datetime.strptime('01/08/2018 00:00', '%m/%d/%Y %H:%M')
predict_end_date = datetime.strptime('01/14/2018 00:00', '%m/%d/%Y %H:%M')
    
print(train_start_date)
print(train_end_date)
print(predict_start_date)
print(predict_end_date)

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
export_data()
start_date = start_date + timedelta(weeks=1)


# In[ ]:


model_neural.score(X_predict,y_predict)

