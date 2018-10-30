# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:14:42 2018

@author: Rahul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split (x,y,test_size=0.20,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier=Sequential()

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



y_pred=classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)





import torch
from torch.autograd import Variable

x_data=Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data=Variable=(torch.Tensor([[2.0],[4.0],[6.0]]))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear=torch.nn.Linear(1,1)
        
        
    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred
    
model=Model()
    
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01) 


for epoch in range (500):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   



























