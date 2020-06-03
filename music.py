# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:27:08 2020

@author: ms_zh
"""

import pandas as pd
import numpy as np
members = pd.read_csv('members.csv')
songs = pd.read_csv('songs.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_a = train.merge(members, on = 'msno', how = 'left')

train_data = train_a.merge(songs, on = 'song_id', how = 'left')

test_a = test.merge(members, on = 'msno', how = 'left')

test_data = test_a.merge(songs, on = 'song_id', how = 'left')


import seaborn as sns
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(15,10))
fig.set(alpha=0.2)

plt.subplot2grid((2,2),(0,0))
train_data.gender.value_counts().plot(kind='bar')
plt.title('gender')
plt.ylabel('number')

plt.subplot2grid((2,2),(0,1))
train_data.source_system_tab.value_counts().plot(kind='bar')
plt.title('source_system_tab')
plt.ylabel('number')

plt.subplot2grid((2,2),(1,0))
train_data.source_screen_name.value_counts().plot(kind='bar')
plt.title('source_screen_name')
plt.ylabel('number')

plt.subplot2grid((2,2),(1,1))
train_data.source_type.value_counts().plot(kind='bar')
plt.title('source_type')
plt.ylabel('number')

plt.show()

def transform_type(data):
    col = data.columns
    for i in col:
        if i in ['id','song_length']:
            pass
        else:
            data[i] = data[i].astype('category')
            
            
def discretify(data,max_bins=5):
    data['song_length'] = data['song_length'].apply(lambda x: float(str(x)))
    data['song_length'] = pd.qcut(data['song_length'],q = max_bins, labels = False)

def get_registration_year(data):
    data['registration_year'] = data['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
    del data['registration_init_time']

def get_period(data):
    data['expiration_year'] = data['expiration_date'].apply(lambda x: int(str(x)[0:4]))
    data ['period'] = data ['expiration_year'] - data['registration_year']
    del data['expiration_date']


print('----- start processing feature --------')
transform_type(train_data) 
transform_type(test_data)

get_registration_year(train_data)
get_registration_year(test_data)

get_period(train_data)
get_period(test_data)

discretify(train_data)
discretify(test_data)


plt.figure(figsize=(14,10))
plt.suptitle('Feature Distributions', fontsize=22)

plt.subplot(221)
g = sns.countplot(x='registration_year',hue='target', data=train_data)
plt.legend(title='label', loc='upper center', labels=['0', '1'])

g.set_title("Year Distribution", fontsize=19)
g.set_xlabel("year", fontsize=17)
g.set_ylabel("Count", fontsize=17)


plt.subplot(222)
g1 = sns.countplot(x='period', hue='target', data=train_data)
plt.legend(title='label', loc='best', labels=['0', '1'])

g1.set_title("period", fontsize=19)
g1.set_xlabel("period", fontsize=17)
g1.set_ylabel("Count", fontsize=17)

plt.subplot(212)
g2 = sns.countplot(x='song_length', hue='target', data=train_data)
plt.legend(title='label', loc='best', labels=['0','1'])

g2.set_title("song_length", fontsize=20)
g2.set_xlabel("split", fontsize=17)
g2.set_ylabel("count", fontsize=17)

plt.subplots_adjust(hspace = 0.6, top = 0.85)

plt.show()

print('----- model --------')

import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_bin': 255,
    'learning_rate': 0.1,
    'num_leaves': 64,
    'max_depth': -1, 
    'feature_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_fraction': 0.8,    
    'min_data_in_leaf': 21,
    'min_sum_hessian_in_leaf': 3.0
}

def train_model(dataset, model_file):
    d_x = dataset.drop(['target'],axis=1)
    d_y = dataset['target'].astype(int)

    train_X, valid_X, train_Y, valid_Y = train_test_split(d_x, d_y, test_size = 0.2)

    lgb_train = lgb.Dataset(train_X, label=train_Y)
    lgb_eval = lgb.Dataset(valid_X, label=valid_Y, reference=lgb_train)
    print("Training...")
    bst = lgb.train(
        params,
        lgb_train,
        categorical_feature=list(range(len(d_x.columns))),
        valid_sets=[lgb_eval],
        early_stopping_rounds=20,
        num_boost_round=5000)
    print("Saving Model...")
    bst.save_model(str(model_file))
    
def predict(dataset, model_file):
    print("Predicting...")
    bst = lgb.Booster(model_file=model_file)
    predict_result = bst.predict(dataset)
    return predict_result  

model_path = 'model.txt'
train_model(train_data,model_path)
x_test = test_data.drop(['id'],axis = 1)
ids = test_data['id'].values
prob = predict(x_test,model_path)

bst = lgb.Booster(model_file=model_path)
lgb.plot_importance(bst, max_num_features=10)  
plt.show()

print('----- output --------')

result = pd.DataFrame(prob,index = ids)
result.to_csv('result.txt', sep='\t')