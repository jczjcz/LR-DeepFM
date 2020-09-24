# #!/usr/bin/env python
# # coding: utf-8


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
import torch.utils.data as Data
import torch.optim as optim



#读取数据
file = './frappe_new.train2.csv'
file3 = './frappe_new.test.csv'
train = pd.read_csv(file)
test = pd.read_csv(file3)





def preprocessing(data_input):
    standardopt = MM()
    feature = data_input.iloc[:,1:]#除了第一列都是特征
    feature = standardopt.fit_transform(feature)
    feature = np.mat(feature)
    #print(feature)
    label = np.array(data_input.iloc[:,0]) #第一列是标签
    return feature,label




#构建FM模型，x \in batch_size*n
class FM_model(nn.Module):
    def __init__(self):
        super(FM_model,self).__init__()
        self.n = 10
        self.k = 20
        self.v = nn.Parameter(torch.randn(self.k,self.n))
        self.linear = nn.Linear(self.n,1,bias=True)
    def forward(self,x):
        fm_1 = torch.mm(x,self.v.t())
        fm_1 = torch.pow(fm_1,2)
        fm_2 = torch.mm(torch.pow(x,2),torch.pow(self.v.t(),2))
        output =  torch.sigmoid(self.linear(x)+0.5*torch.sum(fm_1-fm_2))
        return output
model = FM_model()  #取k=20




data_train,label_train = preprocessing(train)
data_test,label_test = preprocessing(test)
label_train = label_train.reshape(-1,1)
data_train = torch.Tensor(data_train)
label_train = torch.Tensor(label_train)
data_test= torch.Tensor(data_test)
print(data_train.size())
print(label_train.size())
label_test = torch.Tensor(label_test)
train_dataset = Data.TensorDataset(data_train,label_train)
test_dataset = Data.TensorDataset(data_test,label_test)
BATCH_SIZE = 1
loader_train = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
loader_test = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)



optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(loader_train):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #print(inputs.size())
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        #print("outputs = ",outputs)
        #print("labels = ",labels)
        loss = (outputs-labels).sum()
        #print("loss = ",loss)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 5000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
print('Finished Training')



PATH = './FM_model.pth'
torch.save(model.state_dict(), PATH)




model = FM_model()
model.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    m = 0
    for data in loader_test:
        inputs, labels = data
        labels = labels.reshape(1,1)
        #print(inputs,labels)
        outputs = model(inputs)
        for i in range(1):
            if outputs[i][0]>0: outputs[i][0]=1
            else: outputs[i][0]= -1
        #print(outputs,labels,'------------')
        total += labels.size(0)
        if m == 0 :
            print("outputs = ",outputs)
            print("labels = ",labels)
            m += 1
        correct += (outputs== labels).sum().item()   

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))






