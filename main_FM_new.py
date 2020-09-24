#!/usr/bin/env python
# coding: utf-8



import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression    #逻辑斯蒂回归
from sklearn.model_selection import train_test_split   #用于分割数据
from sklearn import metrics                 
from sklearn.metrics import mean_squared_error
import csv
import pandas as pd
from random import normalvariate #正态分布
from sklearn.preprocessing import MinMaxScaler as MM #可将特征缩放到给定的最小值和最大值之间
import pandas as pd




file = './frappe_new.train.csv'
file3 = './frappe_new.test.csv'
train = pd.read_csv(file)
test = pd.read_csv(file3)



def preprocessing(data_input):
    standardopt = MM()
    feature = data_input.iloc[:,2:]#除了前两列都是特征
    feature = standardopt.fit_transform(feature)
    feature = np.mat(feature)
    #print(feature)
    label = np.array(data_input.iloc[:,1]) #第二列是标签
    return feature,label




def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))



def sgd_fm(datamatrix,label,k,iter,alpha):
    '''
    k：分解矩阵的长度
    datamatrix：数据集特征
    label：数据集标签
    iter:迭代次数
    alpha:学习率
    '''
    m, n = np.shape(datamatrix) #m:数据集特征的行数，n:数据集特征的列数
    w0 = 0.0 #初始化w0为0
    w = np.zeros((n, 1)) #初始化w
    v = normalvariate(0, 0.2) * np.ones((n, k))
    for it in range(iter):   #迭代次数
        for i in range(m):  #遍历所有行
            # inner1 = datamatrix[i] * w
            inner1 = datamatrix[i] * v #对应公式进行计算，计算和平方
            inner2 = np.multiply(datamatrix[i], datamatrix[i]) * np.multiply(v, v)#计算平方和
            jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
            ypredict = w0 + datamatrix[i] * w + jiaocha   #计算预测值
            # print(np.shape(ypredict))
            # print(ypredict[0, 0])
            yp = sigmoid(label[i]*ypredict[0, 0])      #[0,0]可加可不加 
            loss = 1 - (-(np.log(yp)))    #准确率高，yp大，np.log(yp)大，loss大
            w0 = w0 - alpha * (yp - 1) * label[i] * 1    #反向传播
            #处理V的梯度
            for j in range(n):
                if datamatrix[i, j] != 0:
                    w[j] = w[j] - alpha * (yp - 1) * label[i] * datamatrix[i, j]
                    for k in range(k):
                        v[j, k] = v[j, k] - alpha * ((yp - 1) * label[i] * (datamatrix[i, j] * inner1[0, k] - v[j, k] *                               datamatrix[i, j] * datamatrix[i, j]))
        print('第%s次训练的误差为：%f' % (it, loss))
        #print(ypredict)
        #print("datamatrix =",datamatrix)
        #print("jiaocha = ",jiaocha)
    return w0, w, v



def predict(w0, w, v, x, thold):
    inner1 = x * v
    inner2 = np.multiply(x, x) * np.multiply(v, v)
    jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
    ypredict = w0 + x * w + jiaocha
    y0 = sigmoid(ypredict[0,0])
    if y0 > thold:
        yp = 1
    else:
        yp = -1
    return yp



def calaccuracy(datamatrix, label, w0, w, v, thold):  #准确率
    error = 0
    for i in range(np.shape(datamatrix)[0]):
        yp = predict(w0, w, v, datamatrix[i], thold)
        if yp != label[i]:
            error += 1
    accuray = 1.0 - error/np.shape(datamatrix)[0]
    return accuray




data_train,label_train = preprocessing(train)
data_test,label_test = preprocessing(test)
w0, w, v = sgd_fm(data_train, label_train, 20, 20, 0.1)
maxaccuracy = 0.0 
tmpthold = 0.0
accuracy_test = calaccuracy(data_test, label_test, w0, w, v, 0.5)
# for i in np.linspace(0.4, 0.6, 201):  #选择最好的间隔
#     #print(i)
#     accuracy_test = calaccuracy(data_test, label_test, w0, w, v, i)
#     if accuracy_test > maxaccuracy:
#         maxaccuracy = accuracy_test
#         tmpthold = i
print("准确率:",accuracy_test)





