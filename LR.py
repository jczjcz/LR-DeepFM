

import numpy as np
from sklearn.linear_model import LogisticRegression    #逻辑斯蒂回归
from sklearn.model_selection import train_test_split   #用于分割数据
from sklearn import metrics                 
from sklearn.metrics import mean_squared_error
import csv
import pandas as pd




file_2 = "./frappe.train2.libfm"
file_ = "./frappe.test2.libfm"
fp=pd.read_table(file_2,sep=' ',engine='python',names=['label','user','item','cnt', 'daytime','weekday','isweekend','homework','cost','weather','country'])  
fp.to_csv('./frappe_new.train2.csv',index=False)




data = pd.read_csv(file_,sep=' ',engine='python',names=['label','user','item','cnt', 'daytime','weekday','isweekend','homework','cost','weather','country'])  
data.to_csv("./frappe_new.test2.csv",index = False)


a,b = data.shape
print(a,b)




list(data.columns.values)
data.shape


file = "./frappe_new.train2.csv"
file3 = "./frappe_new.test2.csv"
train = pd.read_csv(file)
test = pd.read_csv(file3)




def train_model():    
    print("Start Train Model...")
    label = "label"
    ID = "user"
    x_columns = [x for x in train.columns if x not in [label, ID]]    #.columns返回列索引
    x_train = train[x_columns]
    y_train = train[label]
    # 定义模型
    lr = LogisticRegression(penalty="l2", tol=1e-4, fit_intercept=True)
    lr.fit(x_train, y_train)
    return lr



def evaluate(lr):
    label = "label"
    ID = "user"
    x_columns = [x for x in test.columns if x not in [label, ID]]
    x_test = test[x_columns]
    y_test = test[label]
    y_pred = lr.predict(x_test)
    new_y_pred = y_pred
    mse = mean_squared_error(y_test, new_y_pred)
    print("MSE: %.4f" % mse)
    accuracy = metrics.accuracy_score(y_test.values, new_y_pred)
    print("Accuracy : %.4g" % accuracy)
    auc = metrics.roc_auc_score(y_test.values, new_y_pred)
    print("AUC Score : %.4g" % auc)





lr = train_model()
evaluate(lr)





