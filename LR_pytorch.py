#!/usr/bin/env python
# coding: utf-8



import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd



file = "./frappe_new.train2.csv"
file3 = "./frappe_new.test2.csv"
train = pd.read_csv(file)
test = pd.read_csv(file3)



label = "label"
ID = "user"
num = "num"
item = "item"
cnt = "cnt"
daytime = "daytime"
x_columns = [x for x in train.columns if x not in [label ,num,ID]]
x_train = train[x_columns]
y_train = train[label]
x_test = test[x_columns]
y_test = test[label]

# print("y_train.shape=",y_train.shape)

for i in range(202027):
    if y_train[i] == -1:
        y_train[i] = 0
for i in range(28860):
    if y_test[i] == -1:
        y_test[i] = 0

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = torch.tensor(x_train).type(torch.FloatTensor)
y_train = torch.tensor(y_train).type(torch.FloatTensor)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = torch.tensor(x_test).type(torch.FloatTensor)
y_test = torch.tensor(y_test).type(torch.FloatTensor)



y_train = torch.squeeze(y_train)
y_test = torch.squeeze(y_test)

# print("type(y_train)=",type(y_train))
# print("y_train.shape = ",y_train.shape)

# print("x.shape",x_train.shape)
# print("y.shape",y_train.shape)
print(type(x_train),type(y_train))





class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.lr = nn.Linear(9,1)
        self.sm = nn.Sigmoid()
    def forward(self,x):
        x = self.lr(x)
        #print("x_1 =",x)
        x = self.sm(x)
        #print("x_2 =",x)
        return x
model = LR()



#定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)




for epoch in range(1000):
    x_data = Variable(x_train)
    y_data = Variable(y_train)

    out = model(x_data)   #根据逻辑斯蒂拟合出来的y值
    # print("out=",out)
    # print("y_data=",y_data)
    loss = criterion(out,y_data) #计算损失函数
    print_loss = loss.data.item() #得出损失函数值
    mask = out.ge(0.5).float()
    #correct = (mask == y_data).sum()  #计算正确预测的样本数
    correct = (mask[:, 0] == y_data).sum()
    acc = correct.item()/x_data.size(0)  #计算准确率
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #每20轮打印当前的误差和精度
    if (epoch+1)%100 == 0:
        print('*'*10)
        print('epoch {}'.format(epoch+1))  #误差
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))  #精度


PATH = './FM_model.pth'
torch.save(model.state_dict(), PATH)

x_data = Variable(x_test)
y_data = Variable(y_test)
out = model(x_data)   #根据逻辑斯蒂拟合出来的y值
loss = criterion(out,y_data) #计算损失函数
print_loss = loss.data.item() #得出损失函数值
mask = out.ge(0.5).float()
correct = (mask[:, 0] == y_data).sum()
acc = correct.item()/x_data.size(0)  #计算准确率
print("last_acc = ",acc)

# model.load_state_dict(torch.load(PATH))

# correct = 0
# total = 0
# with torch.no_grad():
#     m = 0
#     for data in loader_test:
#         inputs, labels = data
#         labels = labels.reshape(1,1)
#         #print(inputs,labels)
#         outputs = model(inputs)
#         for i in range(1):
#             if outputs[i][0]>0: outputs[i][0]=1
#             else: outputs[i][0]= -1
#         #print(outputs,labels,'------------')
#         total += labels.size(0)
#         if m == 0 :
#             print("outputs = ",outputs)
#             print("labels = ",labels)
#             m += 1
#         correct += (outputs== labels).sum().item()   

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

